"""Turn polars M1 frames into engine inputs: M1 arrays, higher timeframe series, cost model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from autotrader.core.models import CalendarEvent, Instrument, Timeframe
from autotrader.core.series import BarsArray, to_ns
from autotrader.data.aggregate import resample
from autotrader.data.convert import median_spread_by_hour, to_bars_array
from autotrader.data.versioning import data_version
from autotrader.engine.costs import CostModel, InstrumentCosts, SpreadModel, news_by_symbol, rollover_times
from autotrader.strategies_api.manifest import StrategyManifest


@dataclass(frozen=True)
class EngineInputs:
    m1: dict[str, BarsArray]
    series: dict[tuple[str, Timeframe], BarsArray]
    instruments: dict[str, InstrumentCosts]
    cost_model: CostModel
    data_versions: dict[str, str]


@dataclass(frozen=True)
class CostOverride:
    """A calibrated cost model (learning L8): spread by hour of week per symbol (None = keep the data's
    median for that hour) and slippage as a multiple of the spread per symbol."""

    version_id: str
    spread_by_hour: Mapping[str, Sequence[float | None]]
    slippage_mult: Mapping[str, float]


def prepare(
    frames: Mapping[str, pl.DataFrame],
    manifest: StrategyManifest,
    instruments: Mapping[str, Instrument],
    *,
    events: Sequence[CalendarEvent] = (),
    spread_frames: Mapping[str, pl.DataFrame] | None = None,
    synthetic: bool = False,
    slippage_mult: float = 0.2,
    costs: CostOverride | None = None,
) -> EngineInputs:
    """`spread_frames` lets broker data define spreads while another source supplies prices."""
    m1: dict[str, BarsArray] = {}
    series: dict[tuple[str, Timeframe], BarsArray] = {}
    medians = {}
    versions = {}
    for sym in manifest.symbols:
        df = frames[sym].unique(subset="open_time", keep="first").sort("open_time")
        m1[sym] = to_bars_array(df)
        medians[sym] = median_spread_by_hour((spread_frames or {}).get(sym, df))
        versions[sym] = data_version(df, sym, "M1", synthetic=synthetic)
        for tf in manifest.timeframes:
            if tf != Timeframe.M1:
                series[(sym, tf)] = to_bars_array(resample(df, tf))
    start = min(pl.Series(frames[s]["open_time"]).min() for s in manifest.symbols)  # type: ignore[type-var]
    end = max(pl.Series(frames[s]["open_time"]).max() for s in manifest.symbols)  # type: ignore[type-var]
    rolls = np.asarray([t for t, _ in rollover_times(start, end)], dtype=np.int64)  # type: ignore[arg-type]
    by_ccy: dict[str, list[int]] = {}
    for e in events:
        if e.impact == "high":
            by_ccy.setdefault(e.currency, []).append(to_ns(e.time))
    news = news_by_symbol({s: (instruments[s].base, instruments[s].quote) for s in manifest.symbols}, by_ccy)
    for sym, hours in (costs.spread_by_hour if costs is not None else {}).items():
        if sym in medians and len(hours) == 168:
            base = medians[sym].copy()
            for h, v in enumerate(hours):
                if v is not None:
                    base[h] = v
            medians[sym] = base
    spreads = SpreadModel(median_by_hour=medians, news_ns=news, rollover_ns=rolls)
    return EngineInputs(
        m1=m1,
        series=series,
        instruments={s: InstrumentCosts.from_instrument(instruments[s]) for s in manifest.symbols},
        cost_model=CostModel(
            spreads,
            slippage_mult=slippage_mult,
            slippage_mult_by_symbol=dict(costs.slippage_mult) if costs is not None else {},
        ),
        data_versions=versions,
    )
