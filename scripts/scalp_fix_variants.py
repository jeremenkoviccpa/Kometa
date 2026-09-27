"""Research: the scalper's three candidate fixes, declared 2026-09-27 before any was run (research_log).

The diagnosis (2017-2020): before costs the breakout has no edge (-0.007R a trade); costs of 0.109R a trade
make it lose. The candidates attack costs or signal quality through existing parameters only. Each runs on
the exploration window 2017-2020; the best by R per trade then gets one run on the test window 2021-2022.
2023 on (the holdout) is never read. Every run is a trial.
    uv run python scripts/scalp_fix_variants.py data/research_2017_2023
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from autotrader.core.hashing import hash_obj
from autotrader.core.ledger import JsonlLedger
from autotrader.core.settings import Settings
from autotrader.data.instruments import load_instruments
from autotrader.engine.backtest import run_backtest
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.inputs import prepare
from autotrader.validation.runner import profit_factor
from autotrader.validation.store import Trial, TrialRegistry

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "strategies" / "library" / "scalp_session_breakout"
VARIANTS: dict[str, dict[str, object]] = {
    "current": {},
    "A spread <= 5% of risk": {"max_spread_r": 0.05},
    "B breakout 15% of the range": {"buffer": 0.15},
    "C target 2R": {"reward_risk": 2.0},
}
WINDOWS = {
    "explore 2017-2020": (datetime(2017, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)),
    "test 2021-2022": (datetime(2021, 1, 1, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)),
}


def run(job: tuple[str, str, str]) -> tuple[str, str, dict[str, float]]:
    data, name, window = job
    ls = load_strategy(STRATEGY)
    a, b = WINDOWS[window]
    frame = pl.read_parquet(Path(data) / "XAUUSD_M1.parquet").filter(
        (pl.col("open_time") >= a) & (pl.col("open_time") < b)
    )
    inst, _ = load_instruments(ROOT / "config" / "instruments.yaml")
    inp = prepare({"XAUUSD": frame}, ls.manifest, {"XAUUSD": inst["XAUUSD"]})
    res = run_backtest(ls.cls, inp.m1, inp.series, inp.instruments, inp.cost_model, params=VARIANTS[name])  # type: ignore[arg-type]
    r = np.array([t.r_multiple for t in res.trades], dtype=np.float64)
    risk = np.array([t.money_at_risk for t in res.trades])
    cost = (
        np.array([t.spread_cost + t.slippage_cost + t.commission for t in res.trades]) / risk if r.size else r
    )
    return (
        name,
        window,
        {
            "trades": float(r.size),
            "win": float((r > 0).mean()) if r.size else 0.0,
            "avg_r": float(r.mean()) if r.size else 0.0,
            "gross_r": float((r + cost).mean()) if r.size else 0.0,
            "cost_r": float(cost.mean()) if r.size else 0.0,
            "total_r": float(r.sum()),
            "pf": profit_factor(r) if r.size else 0.0,
        },
    )


def record(reg: TrialRegistry, name: str, window: str, st: dict[str, float]) -> None:
    ls = load_strategy(STRATEGY)
    a, b = WINDOWS[window]
    reg.record(
        Trial(
            family=ls.manifest.family,
            strategy_id=ls.manifest.id,
            version=ls.manifest.version,
            params_hash=hash_obj({**ls.manifest.param_values(), **VARIANTS[name]}),
            kind="research",
            data_version=hash_obj({"XAUUSD": window}),
            window_start=a.isoformat(),
            window_end=b.isoformat(),
            sharpe=st["avg_r"],
            trades=int(st["trades"]),
            code_hash=ls.code_hash,
            note=f"scalper fix: {name} on {window}",
        )
    )


def line(name: str, window: str, st: dict[str, float]) -> str:
    return (
        f"{window:18} {name:28} trades {st['trades']:4.0f}  win {st['win']:5.1%}  avg {st['avg_r']:+.3f}R  "
        f"(before costs {st['gross_r']:+.3f}R, costs {st['cost_r']:.3f}R)  "
        f"total {st['total_r']:+6.1f}R  PF {st['pf']:.2f}"
    )


def main(data: str) -> None:
    reg = TrialRegistry(JsonlLedger(Settings().ledger_path))
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as pool:
        explore = list(pool.map(run, [(data, n, "explore 2017-2020") for n in VARIANTS]))
    for name, window, st in explore:
        record(reg, name, window, st)
        print(line(name, window, st))
    best = max((x for x in explore if x[0] != "current"), key=lambda x: x[2]["avg_r"])
    print(
        f"best on the exploration window: {best[0]}; one run on the test window, next to the current version:"
    )
    for name in ("current", best[0]):
        n, w, st = run((data, name, "test 2021-2022"))
        record(reg, n, w, st)
        print(line(n, w, st))


if __name__ == "__main__":
    main(sys.argv[1])
