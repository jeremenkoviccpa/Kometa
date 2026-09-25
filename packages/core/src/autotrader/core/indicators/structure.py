"""Swing points and trend state.

A swing at bar j needs `right` later bars to confirm it. To make lookahead
impossible, outputs are aligned to the CONFIRMATION bar j + right, never to j.
A swing high is strictly above the `left` bars before it and >= the `right`
bars after it, so a flat double top yields one swing, not two.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from autotrader.core.indicators._base import NAN, FloatArray, as_f64, nan_array, windows


class Swings(NamedTuple):
    high: FloatArray  # price of the swing high confirmed at this bar, else NaN
    low: FloatArray


def _is_swing_high(w: FloatArray, left: int) -> bool:
    c = w[left]
    return bool(np.all(w[:left] < c) and np.all(w[left + 1 :] <= c))


def _is_swing_low(w: FloatArray, left: int) -> bool:
    c = w[left]
    return bool(np.all(w[:left] > c) and np.all(w[left + 1 :] >= c))


def swings(high: object, low: object, left: int = 2, right: int = 2) -> Swings:
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")
    h, lo = as_f64(high), as_f64(low)
    sh, sl = nan_array(h.size), nan_array(h.size)
    span = left + right + 1
    if h.size < span:
        return Swings(sh, sl)
    wh, wl = windows(h, span), windows(lo, span)
    ch, cl = wh[:, left : left + 1], wl[:, left : left + 1]
    is_hi = np.all(wh[:, :left] < ch, axis=1) & np.all(wh[:, left + 1 :] <= ch, axis=1)
    is_lo = np.all(wl[:, :left] > cl, axis=1) & np.all(wl[:, left + 1 :] >= cl, axis=1)
    sh[span - 1 :] = np.where(is_hi, ch[:, 0], np.nan)
    sl[span - 1 :] = np.where(is_lo, cl[:, 0], np.nan)
    return Swings(sh, sl)


def _trend(last_hi: float, prev_hi: float, last_lo: float, prev_lo: float) -> float:
    if np.isnan(prev_hi) or np.isnan(prev_lo):
        return 0.0
    if last_hi > prev_hi and last_lo > prev_lo:
        return 1.0
    if last_hi < prev_hi and last_lo < prev_lo:
        return -1.0
    return 0.0


def trend_state(high: object, low: object, left: int = 2, right: int = 2) -> FloatArray:
    """+1 higher highs and higher lows, -1 lower highs and lower lows, 0 otherwise."""
    s = swings(high, low, left, right)
    out = np.zeros(s.high.size, dtype=np.float64)
    hi = [NAN, NAN]  # prev, last
    lo = [NAN, NAN]
    for t in range(s.high.size):
        if not np.isnan(s.high[t]):
            hi = [hi[1], float(s.high[t])]
        if not np.isnan(s.low[t]):
            lo = [lo[1], float(s.low[t])]
        out[t] = _trend(hi[1], hi[0], lo[1], lo[0])
    return out


class SwingDetector:
    def __init__(self, left: int = 2, right: int = 2) -> None:
        if left < 1 or right < 1:
            raise ValueError("left and right must be >= 1")
        self.left = left
        span = left + right + 1
        self._h: deque[float] = deque(maxlen=span)
        self._l: deque[float] = deque(maxlen=span)

    def update(self, high: float, low: float) -> tuple[float, float]:
        self._h.append(float(high))
        self._l.append(float(low))
        if len(self._h) < (self._h.maxlen or 0):
            return NAN, NAN
        h = np.asarray(self._h, dtype=np.float64)
        lo = np.asarray(self._l, dtype=np.float64)
        return (
            float(h[self.left]) if _is_swing_high(h, self.left) else NAN,
            float(lo[self.left]) if _is_swing_low(lo, self.left) else NAN,
        )


class TrendState:
    def __init__(self, left: int = 2, right: int = 2) -> None:
        self._sw = SwingDetector(left, right)
        self._hi = [NAN, NAN]
        self._lo = [NAN, NAN]

    def update(self, high: float, low: float) -> float:
        sh, sl = self._sw.update(high, low)
        if not np.isnan(sh):
            self._hi = [self._hi[1], sh]
        if not np.isnan(sl):
            self._lo = [self._lo[1], sl]
        return _trend(self._hi[1], self._hi[0], self._lo[1], self._lo[0])
