"""Research: the owner's sniper method on more markets (owner's choice, 2026-09-26; docs/research_log.md).

The rules are not gold-specific (every distance is in ATR, the spread limit is a share of the risk), so the
same code runs on each symbol as written, and once more with the score threshold off to count the setups
that pass the hard rules. Every run is recorded as a trial. Usage:
    uv run python scripts/sniper_markets.py data/research_2017_2023 XAUUSD EURUSD GBPUSD
"""

from __future__ import annotations

import sys
from collections import Counter
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
from autotrader.validation.runner import daily_r_returns, profit_factor, with_manifest
from autotrader.validation.store import Trial, TrialRegistry

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"as written": {}, "threshold off": {"min_score": 0, "offsession_score": 0}}


def run(job: tuple[str, str, str]) -> tuple[str, str, dict[str, float], str, int, int]:
    data, symbol, run_name = job
    ls = load_strategy(ROOT / "strategies" / "library" / "smc_sniper")
    cls = with_manifest(ls.cls, symbols=[symbol])
    frame = pl.read_parquet(Path(data) / f"{symbol}_M1.parquet")
    inst, _ = load_instruments(ROOT / "config" / "instruments.yaml")
    inp = prepare({symbol: frame}, cls.manifest, {symbol: inst[symbol]})
    res = run_backtest(cls, inp.m1, inp.series, inp.instruments, inp.cost_model, params=RUNS[run_name])
    r = np.array([t.r_multiple for t in res.trades], dtype=np.float64)
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq)) if r.size else 0.0
    start, end = int(inp.m1[symbol].open_time[0]), int(inp.m1[symbol].close_time[-1])
    quality = Counter(dict(t.tags).get("quality", "?") for t in res.trades)
    stats = {
        "trades": float(r.size),
        "years": (end - start) / (365.25 * 86_400e9),
        "win": float((r > 0).mean()) if r.size else 0.0,
        "avg_r": float(r.mean()) if r.size else 0.0,
        "total_r": float(r.sum()),
        "pf": profit_factor(r) if r.size else 0.0,
        "max_dd_r": dd,
        "sharpe_d": sharpe(daily_r_returns(res.trades, start, end, 0.005)),
    }
    return symbol, run_name, stats, ", ".join(f"{k} {v}" for k, v in sorted(quality.items())), start, end


def main(data: str, symbols: list[str]) -> None:
    ls = load_strategy(ROOT / "strategies" / "library" / "smc_sniper")
    reg = TrialRegistry(JsonlLedger(Settings().ledger_path))
    jobs = [(data, s, r) for s in symbols for r in RUNS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(run, jobs))
    for symbol, run_name, st, qual, start, end in results:
        reg.record(
            Trial(
                family=ls.manifest.family,
                strategy_id=ls.manifest.id,
                version=ls.manifest.version,
                params_hash=hash_obj({**ls.manifest.param_values(), **RUNS[run_name]}),
                kind="research",
                data_version=hash_obj({symbol: data}),
                window_start=from_ns(start).isoformat(),
                window_end=from_ns(end).isoformat(),
                sharpe=st["sharpe_d"],
                trades=int(st["trades"]),
                code_hash=ls.code_hash,
                note=f"sniper markets: {symbol} {run_name}",
            )
        )
        print(
            f"{symbol} {run_name:13} {st['years']:.1f}y trades {st['trades']:3.0f} "
            f"({st['trades'] / st['years']:4.1f}/yr) win {st['win']:5.1%} avgR {st['avg_r']:+.3f} "
            f"totalR {st['total_r']:+6.1f} PF {st['pf']:4.2f} maxDD {st['max_dd_r']:4.1f}R  [{qual}]"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
