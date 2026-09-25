"""Candle anatomy and simple two-bar patterns. Ratios are NaN on zero-range bars."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from autotrader.core.indicators._base import NAN, BoolArray, FloatArray, as_f64


class Anatomy(NamedTuple):
    body: FloatArray
    upper_wick: FloatArray
    lower_wick: FloatArray
    range: FloatArray
    body_ratio: FloatArray
    upper_ratio: FloatArray
    lower_ratio: FloatArray


class Patterns(NamedTuple):
    inside: BoolArray
    outside: BoolArray
    bull_engulfing: BoolArray
    bear_engulfing: BoolArray


def anatomy(open_: object, high: object, low: object, close: object) -> Anatomy:
    o, h, lo, c = as_f64(open_), as_f64(high), as_f64(low), as_f64(close)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - lo
    rng = h - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(rng > 0, rng, np.nan)
        return Anatomy(body, upper, lower, rng, body / safe, upper / safe, lower / safe)


def patterns(open_: object, high: object, low: object, close: object) -> Patterns:
    o, h, lo, c = as_f64(open_), as_f64(high), as_f64(low), as_f64(close)
    n = o.size
    inside = np.zeros(n, dtype=np.bool_)
    outside = np.zeros(n, dtype=np.bool_)
    bull = np.zeros(n, dtype=np.bool_)
    bear = np.zeros(n, dtype=np.bool_)
    if n > 1:
        ph, pl, po, pc = h[:-1], lo[:-1], o[:-1], c[:-1]
        inside[1:] = (h[1:] <= ph) & (lo[1:] >= pl)
        outside[1:] = (h[1:] > ph) & (lo[1:] < pl)
        bull[1:] = (pc < po) & (c[1:] > o[1:]) & (o[1:] <= pc) & (c[1:] >= po)
        bear[1:] = (pc > po) & (c[1:] < o[1:]) & (o[1:] >= pc) & (c[1:] <= po)
    return Patterns(inside, outside, bull, bear)


class CandleTracker:
    """Incremental form: feed bars one at a time, get anatomy and patterns for the latest bar."""

    def __init__(self) -> None:
        self._prev: tuple[float, float, float, float] | None = None

    def update(
        self, o: float, h: float, lo: float, c: float
    ) -> tuple[tuple[float, float, float, float, float, float, float], tuple[bool, bool, bool, bool]]:
        body, upper, lower, rng = abs(c - o), h - max(o, c), min(o, c) - lo, h - lo
        ratios = (body / rng, upper / rng, lower / rng) if rng > 0 else (NAN, NAN, NAN)
        pats = (False, False, False, False)
        if self._prev is not None:
            po, ph, pl, pc = self._prev
            pats = (
                h <= ph and lo >= pl,
                h > ph and lo < pl,
                pc < po and c > o and o <= pc and c >= po,
                pc > po and c < o and o >= pc and c <= po,
            )
        self._prev = (o, h, lo, c)
        return (body, upper, lower, rng, *ratios), pats


class Reversals(NamedTuple):
    """Single- and three-bar reversal candles, each true at the bar that completes the pattern."""

    bull_pin: BoolArray  # hammer: long lower wick, small body in the upper part of the range
    bear_pin: BoolArray  # shooting star: long upper wick, small body in the lower part
    doji: BoolArray  # open and close nearly equal
    morning_star: BoolArray  # strong down bar, small bar, up bar closing above the first bar's midpoint
    evening_star: BoolArray  # the mirror image


# Textbook proportions, fixed so every strategy and test means the same thing by "pin bar".
PIN_WICK, PIN_BODY, PIN_NOSE = 0.6, 0.35, 0.2  # wick >= 60% of range, body <= 35%, other wick <= 20%
DOJI_BODY = 0.1
STAR_BIG, STAR_SMALL = 0.5, 0.3


def _single(o: float, h: float, lo: float, c: float) -> tuple[bool, bool, bool]:
    rng = h - lo
    if not rng > 0:
        return False, False, False
    body, up, dn = abs(c - o) / rng, (h - max(o, c)) / rng, (min(o, c) - lo) / rng
    bull = dn >= PIN_WICK and body <= PIN_BODY and up <= PIN_NOSE
    bear = up >= PIN_WICK and body <= PIN_BODY and dn <= PIN_NOSE
    return bull, bear, body <= DOJI_BODY


def _stars(bars: tuple[tuple[float, float, float, float], ...]) -> tuple[bool, bool]:
    (o1, h1, l1, c1), (o2, h2, l2, c2), (o3, _h3, _l3, c3) = bars
    r1, r2 = h1 - l1, h2 - l2
    if not (r1 > 0 and r2 > 0):
        return False, False
    big1, small2 = abs(c1 - o1) / r1 >= STAR_BIG, abs(c2 - o2) / r2 <= STAR_SMALL
    mid1 = (o1 + c1) / 2
    morning = big1 and small2 and c1 < o1 and c3 > o3 and c3 > mid1
    evening = big1 and small2 and c1 > o1 and c3 < o3 and c3 < mid1
    return morning, evening


def reversals(open_: object, high: object, low: object, close: object) -> Reversals:
    o, h, lo, c = as_f64(open_), as_f64(high), as_f64(low), as_f64(close)
    n = o.size
    rng = h - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(rng > 0, rng, np.nan)
        body, up, dn = np.abs(c - o) / safe, (h - np.maximum(o, c)) / safe, (np.minimum(o, c) - lo) / safe
        valid = rng > 0
        bull = valid & (dn >= PIN_WICK) & (body <= PIN_BODY) & (up <= PIN_NOSE)
        bear = valid & (up >= PIN_WICK) & (body <= PIN_BODY) & (dn <= PIN_NOSE)
        doji = valid & (body <= DOJI_BODY)
        morning = np.zeros(n, dtype=np.bool_)
        evening = np.zeros(n, dtype=np.bool_)
        if n > 2:
            b1, b2 = body[:-2], body[1:-1]
            ok = valid[:-2] & valid[1:-1] & (b1 >= STAR_BIG) & (b2 <= STAR_SMALL)
            mid1 = (o[:-2] + c[:-2]) / 2
            o3, c3 = o[2:], c[2:]
            morning[2:] = ok & (c[:-2] < o[:-2]) & (c3 > o3) & (c3 > mid1)
            evening[2:] = ok & (c[:-2] > o[:-2]) & (c3 < o3) & (c3 < mid1)
    return Reversals(bull, bear, doji, morning, evening)


class ReversalTracker:
    """Incremental form of `reversals`."""

    def __init__(self) -> None:
        self._last: list[tuple[float, float, float, float]] = []

    def update(self, o: float, h: float, lo: float, c: float) -> tuple[bool, bool, bool, bool, bool]:
        bull, bear, doji = _single(o, h, lo, c)
        self._last = [*self._last[-2:], (o, h, lo, c)]
        morning, evening = _stars(tuple(self._last)) if len(self._last) == 3 else (False, False)
        return bull, bear, doji, morning, evening
