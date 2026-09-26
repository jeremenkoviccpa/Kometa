"""Research: the sniper variants declared on 2026-09-26 before any of them was run (docs/research_log.md).

Runs each variant on the research data (the holdout year is never in the folder passed in), records every
run as a trial so the family's deflated Sharpe pays for it, and prints one line per variant. Usage:
    uv run python scripts/sniper_variants.py data/research_2017_2023
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from autotrader.core.hashing import hash_obj
from autotrader.core.ledger import JsonlLedger
from autotrader.core.series import from_ns
from autotrader.core.settings import Settings
from autotrader.data.instruments import load_instruments
from autotrader.engine.backtest import run_backtest
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.dsr import sharpe
from autotrader.validation.inputs import prepare
from autotrader.validation.runner import daily_r_returns, profit_factor
from autotrader.validation.store import Trial, TrialRegistry

ROOT = Path(__file__).resolve().parents[1]
VARIANTS: dict[str, dict[str, object]] = {
    "A current": {},
    "B stop at sweep low, widest stop cap": {"stop_at": "sweep", "max_sl_atr": 2.0},
    "C wider buffer under the pullback low": {"stop_buffer_atr": 1.0},
    "D 8 hours for the retest": {"window_m": 480},
    "E trade WATCH too (score 70+)": {"min_score": 70, "offsession_score": 80},
}


def run(name: str, data: str) -> tuple[str, dict[str, float], int, int, int]:
    ls = load_strategy(ROOT / "strategies" / "library" / "smc_sniper")
    frame = pl.read_parquet(Path(data) / "XAUUSD_M1.parquet")
    inst, _ = load_instruments(ROOT / "config" / "instruments.yaml")
    inp = prepare({"XAUUSD": frame}, ls.manifest, {"XAUUSD": inst["XAUUSD"]})
    params = VARIANTS[name]
    res = run_backtest(ls.cls, inp.m1, inp.series, inp.instruments, inp.cost_model, params=params)  # type: ignore[arg-type]
    r = np.array([t.r_multiple for t in res.trades], dtype=np.float64)
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq)) if r.size else 0.0
    start, end = int(inp.m1["XAUUSD"].open_time[0]), int(inp.m1["XAUUSD"].close_time[-1])
    stats = {
        "trades": float(r.size),
        "win": float((r > 0).mean()) if r.size else 0.0,
        "avg_r": float(r.mean()) if r.size else 0.0,
        "total_r": float(r.sum()),
        "pf": profit_factor(r) if r.size else 0.0,
        "max_dd_r": dd,
        "sharpe_d": sharpe(daily_r_returns(res.trades, start, end, 0.005)),
    }
    return name, stats, start, end, hash(ls.code_hash)


def main(data: str) -> None:
    ls = load_strategy(ROOT / "strategies" / "library" / "smc_sniper")
    reg = TrialRegistry(JsonlLedger(Settings().ledger_path))
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as pool:
        results = list(pool.map(run, VARIANTS, [data] * len(VARIANTS)))
    for name, st, start, end, _ in results:
        reg.record(
            Trial(
                family=ls.manifest.family,
                strategy_id=ls.manifest.id,
                version=ls.manifest.version,
                params_hash=hash_obj({**ls.manifest.param_values(), **VARIANTS[name]}),
                kind="research",
                data_version=hash_obj({"XAUUSD": data}),
                window_start=from_ns(start).isoformat(),
                window_end=from_ns(end).isoformat(),
                sharpe=st["sharpe_d"],
                trades=int(st["trades"]),
                code_hash=ls.code_hash,
                note=f"sniper variant: {name}",
            )
        )
        print(
            f"{name:42} trades {st['trades']:4.0f}  win {st['win']:5.1%}  avgR {st['avg_r']:+.3f}  "
            f"totalR {st['total_r']:+6.1f}  PF {st['pf']:5.2f}  maxDD {st['max_dd_r']:4.1f}R"
        )


if __name__ == "__main__":
    main(sys.argv[1])
