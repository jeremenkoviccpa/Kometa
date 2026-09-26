"""The feature snapshot at signal time (spec 14.2): what the market looked like when a strategy signalled.

`snapshot()` takes bars that may run past `now` and drops every bar that closes after it itself, so no
caller can leak the future into a feature (tests/unit/test_journal.py poisons the future to prove it).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict

from autotrader.core.indicators import (
    EventIndex,
    atr,
    ema,
    levels,
    percentile_rank,
    sessions_of,
    slope,
)
from autotrader.core.models import Timeframe
from autotrader.core.series import BarsArray, to_ns

TRENDS = (Timeframe.H1, Timeframe.H4, Timeframe.D1)


class FeatureSnapshot(BaseModel):
    """None means "not enough history to know", never zero. Distances and slopes are in ATR units."""

    model_config = ConfigDict(frozen=True)

    atr_h1: float | None
    atr_pct_h1: float | None  # where today's H1 ATR sits in its last 250 values, 0..1
    slope_h1: float | None  # EMA(20) slope per bar over 5 bars / ATR of that timeframe
    slope_h4: float | None
    slope_d1: float | None
    level_dist_atr: float | None  # to the nearest unbroken H4 level
    realized_vol_d: float | None  # std of M5 log returns over the last day, scaled to a day
    spread_ratio: float | None  # spread now / median M5 close spread over the last day
    minutes_to_event: float | None  # next high-impact event in either currency of the symbol
    session: tuple[str, ...]
    hour: int
    weekday: int

    def row(self) -> dict[str, float]:
        """Numbers only, for models (missing = NaN; sessions one-hot)."""
        out: dict[str, float] = {}
        for k, v in self.model_dump().items():
            if k == "session":
                for s in ("asia", "london", "new_york"):
                    out[f"session_{s}"] = 1.0 if s in v else 0.0
            else:
                out[k] = math.nan if v is None else float(v)
        return out


def _closed(b: BarsArray, now_ns: int) -> BarsArray:
    keep = b.close_time <= now_ns
    if bool(keep.all()):
        return b
    return BarsArray(**{f: getattr(b, f)[keep] for f in b.__dataclass_fields__})


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def snapshot(
    bars: Mapping[Timeframe, BarsArray],
    now: datetime,
    *,
    spread: float,
    currencies: tuple[str, str] | None = None,
    events: EventIndex | None = None,
) -> FeatureSnapshot:
    now_ns = to_ns(now)
    b = {tf: _closed(x, now_ns) for tf, x in bars.items()}

    def last(a: np.ndarray) -> float | None:
        return _finite(float(a[-1])) if a.size else None

    h1 = b.get(Timeframe.H1)
    a_h1 = atr(h1.bid_h, h1.bid_l, h1.bid_c, 14) if h1 is not None and len(h1) else np.array([])
    atr_h1 = last(a_h1)
    trend: dict[Timeframe, float | None] = {}
    for tf in TRENDS:
        x = b.get(tf)
        if x is None or len(x) < 30:
            trend[tf] = None
            continue
        a = last(atr(x.bid_h, x.bid_l, x.bid_c, 14))
        s = last(slope(ema(x.bid_c, 20), 5))
        trend[tf] = s / a if a and s is not None else None
    level_dist = None
    h4 = b.get(Timeframe.H4)
    if h4 is not None and len(h4) >= 30:
        a4 = last(atr(h4.bid_h, h4.bid_l, h4.bid_c, 14))
        book = levels(h4.bid_h, h4.bid_l, h4.bid_c)[-1]
        px = float(h4.bid_c[-1])
        live = [abs(px - lv.price) for lv in book if not lv.broken]
        if a4 and live:
            level_dist = min(live) / a4
    vol = ratio = None
    m5 = b.get(Timeframe.M5)
    if m5 is not None and len(m5) >= 60:
        day = m5.bid_c[-288:]
        r = np.diff(np.log(day))
        vol = _finite(float(np.std(r)) * math.sqrt(288))
        med = float(np.median((m5.ask_c - m5.bid_c)[-288:]))
        ratio = spread / med if med > 0 else None
    mte = None
    if events is not None and currencies is not None:
        mins = [m for c in currencies if (m := events.minutes_to_next(now, c)) is not None]
        mte = min(mins) if mins else None
    return FeatureSnapshot(
        atr_h1=atr_h1,
        atr_pct_h1=last(percentile_rank(a_h1, 250)) if a_h1.size >= 250 else None,
        slope_h1=trend[Timeframe.H1],
        slope_h4=trend[Timeframe.H4],
        slope_d1=trend[Timeframe.D1],
        level_dist_atr=level_dist,
        realized_vol_d=vol,
        spread_ratio=ratio,
        minutes_to_event=mte,
        session=sessions_of(now),
        hour=now.hour,
        weekday=now.weekday(),
    )
