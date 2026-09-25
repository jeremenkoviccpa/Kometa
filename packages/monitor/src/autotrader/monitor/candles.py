"""Candles for the hub's chart, built from the quotes on the bus (bid prices, like the strategies' bars).

M1 candles cover [minute, minute + 1m); higher timeframes are aggregated with `core.timeframes`, the
alignment the historical resampler and the live bar builder use, so an H4 or D1 candle here is the
bar a strategy saw (days close at 17:00 New York). About a week of M1 is kept per symbol.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from autotrader.core.models import Timeframe
from autotrader.core.series import NS_PER_MINUTE, from_ns, to_ns
from autotrader.core.timeframes import bucket_open_ns

M1_KEEP = 7500


@dataclass
class Candle:
    open_ns: int
    o: float
    h: float
    l: float  # noqa: E741
    c: float
    ticks: int = 1
    spread: float = 0.0  # last spread seen in the candle

    def add(self, px: float, spread: float) -> None:
        self.h, self.l, self.c = max(self.h, px), min(self.l, px), px
        self.ticks += 1
        self.spread = spread


@dataclass
class Candles:
    m1: dict[str, deque[Candle]] = field(default_factory=dict)

    def quote(self, symbol: str, at: datetime, bid: float, ask: float) -> None:
        t = to_ns(at)
        minute = t - t % NS_PER_MINUTE
        q = self.m1.setdefault(symbol, deque(maxlen=M1_KEEP))
        if q and q[-1].open_ns == minute:
            q[-1].add(bid, ask - bid)
        elif not q or minute > q[-1].open_ns:
            q.append(Candle(minute, bid, bid, bid, bid, 1, ask - bid))
        # an older quote than the current minute is dropped: candles never change once passed

    def series(self, symbol: str, tf: Timeframe, limit: int) -> list[Candle]:
        src = list(self.m1.get(symbol, ()))
        if tf == Timeframe.M1:
            return src[-limit:]
        out: list[Candle] = []
        for c in src:
            start = bucket_open_ns(c.open_ns, tf)
            if out and out[-1].open_ns == start:
                last = out[-1]
                last.h, last.l, last.c = max(last.h, c.h), min(last.l, c.l), c.c
                last.ticks += c.ticks
                last.spread = c.spread
            else:
                out.append(Candle(start, c.o, c.h, c.l, c.c, c.ticks, c.spread))
        return out[-limit:]


def candle_json(c: Candle) -> dict[str, float | int | str]:
    return {
        "time": from_ns(c.open_ns).isoformat(),
        "t": c.open_ns // 1_000_000_000,
        "o": c.o,
        "h": c.h,
        "l": c.l,
        "c": c.c,
        "ticks": c.ticks,
        "spread": c.spread,
    }
