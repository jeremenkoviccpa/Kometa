"""Support and resistance levels from clustered swing points (spec section 6).

A confirmed swing (high or low) within `k * ATR` of an existing level merges
into it (touch-weighted price, touch count +1); otherwise it starts a new
level. A level is *broken* when a bar closes beyond it by more than
`break_atr * ATR`, and *flipped* when, after the break, price comes back within
`k * ATR` of it from the new side and the bar closes on the new side.

Everything is knowable at the bar where it is reported: swings are added at
their confirmation bar, and `last_touch` is the confirmation bar index.

`levels()` (batch, driven by the vectorized swing and ATR arrays) and
`LevelTracker` (incremental) share `_LevelBook` for the clustering rules and
are tested to agree at every bar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from autotrader.core.indicators._base import as_f64
from autotrader.core.indicators.structure import SwingDetector, swings
from autotrader.core.indicators.volatility import ATR, atr


@dataclass(frozen=True)
class Level:
    price: float
    touches: int
    first_touch: int  # bar index where the first swing was confirmed
    last_touch: int
    kind: str  # "support" or "resistance", by the side price was last on
    broken: bool = False
    broken_at: int = -1
    flipped: bool = False
    flipped_at: int = -1


class _LevelBook:
    def __init__(self, k: float, break_atr: float, max_levels: int) -> None:
        self.k = k
        self.break_atr = break_atr
        self.max_levels = max_levels
        self.levels: list[Level] = []

    def add_swing(self, price: float, is_high: bool, t: int, a: float) -> None:
        best, best_d = -1, np.inf
        for i, lv in enumerate(self.levels):
            d = abs(lv.price - price)
            if d <= self.k * a and d < best_d:
                best, best_d = i, d
        if best >= 0:
            lv = self.levels[best]
            n = lv.touches + 1
            self.levels[best] = replace(
                lv, price=(lv.price * lv.touches + price) / n, touches=n, last_touch=t
            )
        else:
            kind = "resistance" if is_high else "support"
            self.levels.append(Level(price, 1, t, t, kind))
            if len(self.levels) > self.max_levels:
                # drop the level touched longest ago
                oldest = min(range(len(self.levels)), key=lambda i: (self.levels[i].last_touch, i))
                del self.levels[oldest]

    def on_bar(self, high: float, low: float, close: float, t: int, a: float) -> None:
        for i, lv in enumerate(self.levels):
            if not lv.broken:
                up = lv.kind == "resistance" and close > lv.price + self.break_atr * a
                down = lv.kind == "support" and close < lv.price - self.break_atr * a
                if up or down:
                    self.levels[i] = replace(
                        lv, broken=True, broken_at=t, kind="support" if up else "resistance"
                    )
            elif not lv.flipped and t > lv.broken_at:
                if lv.kind == "support":  # broken upward, retest from above
                    retest = low <= lv.price + self.k * a and close > lv.price
                else:
                    retest = high >= lv.price - self.k * a and close < lv.price
                if retest:
                    self.levels[i] = replace(lv, flipped=True, flipped_at=t)

    def snapshot(self) -> tuple[Level, ...]:
        return tuple(sorted(self.levels, key=lambda lv: lv.price))


def levels(
    high: object,
    low: object,
    close: object,
    *,
    left: int = 3,
    right: int = 3,
    atr_n: int = 14,
    k: float = 0.5,
    break_atr: float = 0.25,
    max_levels: int = 20,
) -> list[tuple[Level, ...]]:
    """Level snapshot after every bar (index i reflects bars [0, i])."""
    h, lo, c = as_f64(high), as_f64(low), as_f64(close)
    sw = swings(h, lo, left, right)
    a = atr(h, lo, c, atr_n)
    book = _LevelBook(k, break_atr, max_levels)
    out: list[tuple[Level, ...]] = []
    for t in range(h.size):
        at = float(a[t])
        if not np.isnan(at):
            book.on_bar(float(h[t]), float(lo[t]), float(c[t]), t, at)
            if not np.isnan(sw.high[t]):
                book.add_swing(float(sw.high[t]), True, t, at)
            if not np.isnan(sw.low[t]):
                book.add_swing(float(sw.low[t]), False, t, at)
        out.append(book.snapshot())
    return out


class LevelTracker:
    def __init__(
        self,
        left: int = 3,
        right: int = 3,
        atr_n: int = 14,
        k: float = 0.5,
        break_atr: float = 0.25,
        max_levels: int = 20,
    ) -> None:
        self._sw = SwingDetector(left, right)
        self._atr = ATR(atr_n)
        self._book = _LevelBook(k, break_atr, max_levels)
        self._t = -1

    def update(self, high: float, low: float, close: float) -> tuple[Level, ...]:
        self._t += 1
        sh, sl = self._sw.update(high, low)
        a = self._atr.update(high, low, close)
        if not np.isnan(a):
            self._book.on_bar(high, low, close, self._t, a)
            if not np.isnan(sh):
                self._book.add_swing(sh, True, self._t, a)
            if not np.isnan(sl):
                self._book.add_swing(sl, False, self._t, a)
        return self._book.snapshot()
