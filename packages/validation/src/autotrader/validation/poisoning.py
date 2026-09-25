"""Future poisoning test (spec section 6): the structural proof of no lookahead.

Run the strategy on the real data, then again with every bar after T replaced
by random garbage. Every signal created at or before T must be identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from autotrader.core.models import Instrument
from autotrader.core.timeutil import ensure_utc
from autotrader.engine.backtest import BacktestConfig, run_backtest
from autotrader.strategies_api.base import Strategy
from autotrader.strategies_api.manifest import ParamValue
from autotrader.validation.inputs import prepare


@dataclass(frozen=True)
class PoisoningReport:
    passed: bool
    cut: datetime
    signals_checked: int
    first_difference: str | None


def poison_after(df: pl.DataFrame, cut: datetime, seed: int = 0) -> pl.DataFrame:
    """Replace prices of every bar with open_time >= cut by a random walk at a random scale."""
    cut = ensure_utc(cut)
    after = (df["open_time"] >= cut).to_numpy()
    n = int(after.sum())
    if n == 0:
        return df
    rng = np.random.default_rng(seed)
    base = float(df["bid_c"].to_numpy()[~after][-1]) if (~after).any() else 100.0
    walk = base * rng.uniform(0.3, 3.0) * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    wick = np.abs(rng.normal(0, 0.01, (2, n))) * walk
    spread = np.abs(rng.normal(0, 0.002, n)) * walk + 1e-6
    o = np.r_[walk[0], walk[:-1]]
    cols = {
        "bid_o": o,
        "bid_c": walk,
        "bid_h": np.maximum(o, walk) + wick[0],
        "bid_l": np.minimum(o, walk) - wick[1],
    }
    for k in ("o", "h", "l", "c"):
        cols[f"ask_{k}"] = cols[f"bid_{k}"] + spread
    cols["volume"] = rng.uniform(1, 1000, n)
    out = {c: df[c].to_numpy().copy() for c in df.columns if c != "open_time"}
    for c, v in cols.items():
        out[c][after] = v
    return pl.DataFrame({"open_time": df["open_time"], **out})


def future_poisoning_test(
    strategy_cls: type[Strategy],
    frames: Mapping[str, pl.DataFrame],
    instruments: Mapping[str, Instrument],
    cut: datetime,
    *,
    params: Mapping[str, ParamValue] | None = None,
    config: BacktestConfig | None = None,
    seed: int = 0,
) -> PoisoningReport:
    manifest = strategy_cls.manifest
    poisoned_frames = {s: poison_after(f, cut, seed + i) for i, (s, f) in enumerate(sorted(frames.items()))}
    # the cost model must not leak either: spreads come from the clean data before the cut only
    spread_src = {s: f.filter(pl.col("open_time") < ensure_utc(cut)) for s, f in frames.items()}
    clean_in = prepare(frames, manifest, instruments, spread_frames=spread_src)
    pois_in = prepare(poisoned_frames, manifest, instruments, spread_frames=spread_src)
    a = run_backtest(
        strategy_cls,
        clean_in.m1,
        clean_in.series,
        clean_in.instruments,
        clean_in.cost_model,
        params=params,
        config=config,
    )
    b = run_backtest(
        strategy_cls,
        pois_in.m1,
        pois_in.series,
        pois_in.instruments,
        pois_in.cost_model,
        params=params,
        config=config,
    )
    cut_utc = ensure_utc(cut)
    sa = [s.model_dump_json() for s in a.signals if s.created_at <= cut_utc]
    sb = [s.model_dump_json() for s in b.signals if s.created_at <= cut_utc]
    diff = None
    for x, y in zip(sa, sb, strict=False):
        if x != y:
            diff = f"clean={x}\npoisoned={y}"
            break
    if diff is None and len(sa) != len(sb):
        diff = f"signal count differs up to cut: clean={len(sa)} poisoned={len(sb)}"
    return PoisoningReport(diff is None, cut_utc, len(sa), diff)
