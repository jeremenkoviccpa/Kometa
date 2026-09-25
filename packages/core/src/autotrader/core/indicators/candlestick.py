"""The classic candlestick pattern set, ported from github.com/cm45t3r/candlestick v3.0.0 (MIT, Copyright (c)
2016-present cm45t3r; notice in THIRD_PARTY_NOTICES.md).

Same definitions and thresholds as the original, with one deliberate difference: the original reports a
multi-candle pattern at its FIRST candle, which is lookahead for a trading system (the pattern is not known
until its last candle closes). Here every pattern is true at the bar that completes it.
tests/unit/test_candlestick_port.py checks both forms against the original library's output
(tests/golden/candlestick_oracle.json) and against each other.

Patterns that need a gap (kickers, hanging man, shooting star, piercing line, dark cloud cover) are rare in
markets that trade around the clock, such as gold and FX: they mostly appear after weekend gaps.
"""

from __future__ import annotations

import numpy as np

from autotrader.core.indicators._base import BoolArray, FloatArray, as_f64

EPS = 1e-8  # the original's tolerance for inverted hammers

# name -> number of candles; the order is the public order of the pattern set
PATTERNS: dict[str, int] = {
    "hammer": 1,
    "bullish_hammer": 1,
    "bearish_hammer": 1,
    "inverted_hammer": 1,
    "bullish_inverted_hammer": 1,
    "bearish_inverted_hammer": 1,
    "doji": 1,
    "marubozu": 1,
    "bullish_marubozu": 1,
    "bearish_marubozu": 1,
    "spinning_top": 1,
    "bullish_spinning_top": 1,
    "bearish_spinning_top": 1,
    "bullish_engulfing": 2,
    "bearish_engulfing": 2,
    "bullish_harami": 2,
    "bearish_harami": 2,
    "bullish_kicker": 2,
    "bearish_kicker": 2,
    "hanging_man": 2,
    "shooting_star": 2,
    "piercing_line": 2,
    "dark_cloud_cover": 2,
    "tweezers_top": 2,
    "tweezers_bottom": 2,
    "morning_star": 3,
    "evening_star": 3,
    "three_white_soldiers": 3,
    "three_black_crows": 3,
}
BULLISH = frozenset(
    {
        "bullish_hammer",
        "hammer",
        "bullish_inverted_hammer",
        "inverted_hammer",
        "bullish_engulfing",
        "bullish_harami",
        "bullish_kicker",
        "piercing_line",
        "tweezers_bottom",
        "morning_star",
        "three_white_soldiers",
        "bullish_marubozu",
    }
)
BEARISH = frozenset(
    {
        "bearish_engulfing",
        "bearish_harami",
        "bearish_kicker",
        "hanging_man",
        "shooting_star",
        "dark_cloud_cover",
        "tweezers_top",
        "evening_star",
        "three_black_crows",
        "bearish_marubozu",
    }
)


# ---------------------------------------------------------------- vectorized


class _C:
    """Per-bar geometry, optionally lagged by k bars (lagged entries are NaN, so every test is False)."""

    def __init__(self, o: FloatArray, h: FloatArray, lo: FloatArray, c: FloatArray, k: int = 0) -> None:
        if k:
            o, h, lo, c = (
                np.concatenate([np.full(k, np.nan), a[:-k]]) if a.size > k else np.full(a.size, np.nan)
                for a in (o, h, lo, c)
            )
        self.o, self.h, self.l, self.c = o, h, lo, c
        self.body = np.abs(o - c)
        self.wick = h - np.maximum(o, c)
        self.tail = np.minimum(o, c) - lo
        self.rng = h - lo
        self.bull = o < c
        self.bear = o > c
        self.top = np.where(o <= c, c, o)
        self.bottom = np.where(o <= c, o, c)


def _hammer(x: _C) -> BoolArray:
    return (
        (x.rng > 0)
        & (x.tail >= 2 * x.body)
        & (x.wick <= x.body)
        & (np.maximum(x.o, x.c) > x.l + (x.rng * 2) / 3)
    )


def _inv_hammer(x: _C) -> BoolArray:
    return (
        (x.rng > 0)
        & (x.wick >= 2 * x.body)
        & (x.tail <= x.body + EPS)
        & (np.minimum(x.o, x.c) <= x.h - (x.rng * 2) / 3 + EPS)
    )


def _ratio(a: FloatArray, b: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def candlestick_patterns(open_: object, high: object, low: object, close: object) -> dict[str, BoolArray]:
    """Every pattern in PATTERNS as a boolean array, true at the bar that completes the pattern."""
    o, h, lo, c = as_f64(open_), as_f64(high), as_f64(low), as_f64(close)
    x, p, q = _C(o, h, lo, c), _C(o, h, lo, c, 1), _C(o, h, lo, c, 2)  # current, previous, two back
    nz = x.rng != 0
    hammer, inv = _hammer(x), _inv_hammer(x)
    maru = nz & (_ratio(x.body, x.rng) >= 0.7) & (x.wick <= x.body * 0.1) & (x.tail <= x.body * 0.1)
    spin = (
        nz & (_ratio(x.body, x.rng) < 0.3) & (_ratio(x.wick, x.rng) >= 0.2) & (_ratio(x.tail, x.rng) >= 0.2)
    )
    out: dict[str, BoolArray] = {
        "hammer": hammer,
        "bullish_hammer": x.bull & hammer,
        "bearish_hammer": x.bear & hammer,
        "inverted_hammer": inv,
        "bullish_inverted_hammer": x.bull & inv,
        "bearish_inverted_hammer": x.bear & inv,
        "doji": (x.rng > 0) & (_ratio(x.body, x.rng) < 0.1),
        "marubozu": maru,
        "bullish_marubozu": x.bull & maru,
        "bearish_marubozu": x.bear & maru,
        "spinning_top": spin,
        "bullish_spinning_top": x.bull & spin,
        "bearish_spinning_top": x.bear & spin,
    }
    # two candles: p = first, x = second
    p_in_x = (p.top <= x.top) & (p.bottom >= x.bottom)  # x's body engulfs p's
    x_in_p = (x.top <= p.top) & (x.bottom >= p.bottom)
    out["bullish_engulfing"] = p.bear & x.bull & p_in_x
    out["bearish_engulfing"] = p.bull & x.bear & p_in_x
    out["bullish_harami"] = p.bear & x.bull & x_in_p
    out["bearish_harami"] = p.bull & x.bear & x_in_p
    not_pin = ~(hammer | inv)
    out["bullish_kicker"] = p.bear & x.bull & (p.top < x.bottom) & not_pin
    out["bearish_kicker"] = p.bull & x.bear & (p.bottom > x.top) & not_pin
    out["hanging_man"] = p.bull & (x.bear & hammer) & (x.o > p.h)
    out["shooting_star"] = p.bull & (x.bear & inv) & (x.o > p.h)
    strong_p = (p.rng != 0) & ~(p.body < p.rng * 0.5)
    strong_x = (x.rng != 0) & ~(x.body < x.rng * 0.5)
    mid_p = (p.top + p.bottom) / 2
    out["piercing_line"] = (
        p.bear & strong_p & x.bull & strong_x & ~(x.o >= p.l) & ~(x.c <= mid_p) & ~(x.c >= p.top)
    )
    out["dark_cloud_cover"] = (
        p.bull & strong_p & x.bear & strong_x & ~(x.o <= p.h) & ~(x.c >= mid_p) & ~(x.c <= p.bottom)
    )
    avg = (p.h - p.l + (x.h - x.l)) / 2
    bodies_ok = ~(_ratio(p.body, p.h - p.l) < 0.4) & ~(_ratio(x.body, x.h - x.l) < 0.4)
    tol = avg * 0.01
    out["tweezers_top"] = (avg != 0) & ~(np.abs(p.h - x.h) > tol) & p.bull & x.bear & bodies_ok
    out["tweezers_bottom"] = (avg != 0) & ~(np.abs(p.l - x.l) > tol) & p.bear & x.bull & bodies_ok
    # three candles: q = first, p = second, x = third
    q_big = (q.rng != 0) & ~(q.body < q.rng * 0.6)
    p_small = (p.rng != 0) & ~(p.body > p.rng * 0.3)
    x_big = (x.rng != 0) & ~(x.body < x.rng * 0.6)
    mid_q = (q.top + q.bottom) / 2
    out["morning_star"] = q.bear & q_big & p_small & ~(p.top >= q.bottom) & x.bull & x_big & ~(x.c < mid_q)
    out["evening_star"] = q.bull & q_big & p_small & ~(p.bottom <= q.top) & x.bear & x_big & ~(x.c > mid_q)

    def big(y: _C) -> BoolArray:
        ok: BoolArray = (y.rng != 0) & ~(y.body < y.rng * 0.6)
        return ok

    out["three_white_soldiers"] = (
        q.bull & p.bull & x.bull & big(q) & big(p) & big(x)
        & ~((p.c <= q.c) | (x.c <= p.c))
        & ~((p.o < q.o) | (p.o > q.c))
        & ~((x.o < p.o) | (x.o > p.c))
        & ~(q.wick > q.body * 0.3) & ~(p.wick > p.body * 0.3) & ~(x.wick > x.body * 0.3)
    )  # fmt: skip
    out["three_black_crows"] = (
        q.bear & p.bear & x.bear & big(q) & big(p) & big(x)
        & ~((p.c >= q.c) | (x.c >= p.c))
        & ~((p.o > q.o) | (p.o < q.c))
        & ~((x.o > p.o) | (x.o < p.c))
        & ~(q.tail > q.body * 0.3) & ~(p.tail > p.body * 0.3) & ~(x.tail > x.body * 0.3)
    )  # fmt: skip
    return {name: out[name] for name in PATTERNS}


# ---------------------------------------------------------------- incremental


Bar = tuple[float, float, float, float]


def _geo(b: Bar) -> tuple[float, float, float, float, bool, bool, float, float]:
    o, h, lo, c = b
    return (
        abs(o - c),
        h - max(o, c),
        min(o, c) - lo,
        h - lo,
        o < c,
        o > c,
        (c if o <= c else o),
        (o if o <= c else c),
    )


def _is_hammer(b: Bar) -> bool:
    o, _, lo, c = b
    body, wick, tail, rng, *_ = _geo(b)
    return rng > 0 and tail >= 2 * body and wick <= body and max(o, c) > lo + (rng * 2) / 3


def _is_inv(b: Bar) -> bool:
    o, h, _lo, c = b
    body, wick, tail, rng, *_ = _geo(b)
    return rng > 0 and wick >= 2 * body and tail <= body + EPS and min(o, c) <= h - (rng * 2) / 3 + EPS


def _single(b: Bar) -> dict[str, bool]:
    body, wick, tail, rng, bull, bear, _, _ = _geo(b)
    ham, inv = _is_hammer(b), _is_inv(b)
    maru = rng != 0 and body / rng >= 0.7 and not (wick > body * 0.1 or tail > body * 0.1)
    spin = rng != 0 and body / rng < 0.3 and wick / rng >= 0.2 and tail / rng >= 0.2
    return {
        "hammer": ham,
        "bullish_hammer": bull and ham,
        "bearish_hammer": bear and ham,
        "inverted_hammer": inv,
        "bullish_inverted_hammer": bull and inv,
        "bearish_inverted_hammer": bear and inv,
        "doji": rng > 0 and body / rng < 0.1,
        "marubozu": maru,
        "bullish_marubozu": bull and maru,
        "bearish_marubozu": bear and maru,
        "spinning_top": spin,
        "bullish_spinning_top": bull and spin,
        "bearish_spinning_top": bear and spin,
    }


def _ratio_ok(body: float, rng: float, need: float) -> bool:
    """JS `!(body / rng < need)`: a zero range gives NaN or infinity, which do not fail the test."""
    if rng == 0:
        return True
    return not body / rng < need


def _pair(pb: Bar, xb: Bar) -> dict[str, bool]:
    _, ph, pl, _ = pb
    xo, xh, xl, xc = xb
    pbody, _pw, _pt, prng, pbull, pbear, ptop, pbot = _geo(pb)
    xbody, _xw, _xt, xrng, xbull, xbear, xtop, xbot = _geo(xb)
    p_in_x = ptop <= xtop and pbot >= xbot
    x_in_p = xtop <= ptop and xbot >= pbot
    not_pin = not (_is_hammer(xb) or _is_inv(xb))
    strong_p = prng != 0 and not pbody < prng * 0.5
    strong_x = xrng != 0 and not xbody < xrng * 0.5
    mid = (ptop + pbot) / 2
    avg = (ph - pl + (xh - xl)) / 2
    bodies = _ratio_ok(pbody, ph - pl, 0.4) and _ratio_ok(xbody, xh - xl, 0.4)
    return {
        "bullish_engulfing": pbear and xbull and p_in_x,
        "bearish_engulfing": pbull and xbear and p_in_x,
        "bullish_harami": pbear and xbull and x_in_p,
        "bearish_harami": pbull and xbear and x_in_p,
        "bullish_kicker": pbear and xbull and ptop < xbot and not_pin,
        "bearish_kicker": pbull and xbear and pbot > xtop and not_pin,
        "hanging_man": pbull and xbear and _is_hammer(xb) and xo > ph,
        "shooting_star": pbull and xbear and _is_inv(xb) and xo > ph,
        "piercing_line": pbear and strong_p and xbull and strong_x and not xo >= pl and not xc <= mid
        and not xc >= ptop,
        "dark_cloud_cover": pbull and strong_p and xbear and strong_x and not xo <= ph and not xc >= mid
        and not xc <= pbot,
        "tweezers_top": avg != 0 and not abs(ph - xh) > avg * 0.01 and pbull and xbear and bodies,
        "tweezers_bottom": avg != 0 and not abs(pl - xl) > avg * 0.01 and pbear and xbull and bodies,
    }  # fmt: skip


def _triple(qb: Bar, pb: Bar, xb: Bar) -> dict[str, bool]:
    q, p, x = _geo(qb), _geo(pb), _geo(xb)

    def big(g: tuple[float, ...]) -> bool:
        return g[3] != 0 and not g[0] < g[3] * 0.6

    q_small_p = p[3] != 0 and not p[0] > p[3] * 0.3
    mid = (q[6] + q[7]) / 2
    qo, _, _, qc = qb
    po, _, _, pc = pb
    xo, _, _, xc = xb
    return {
        "morning_star": bool(
            q[5] and big(q) and q_small_p and not p[6] >= q[7] and x[4] and big(x) and not xc < mid
        ),
        "evening_star": bool(
            q[4] and big(q) and q_small_p and not p[7] <= q[6] and x[5] and big(x) and not xc > mid
        ),
        "three_white_soldiers": bool(
            q[4] and p[4] and x[4] and big(q) and big(p) and big(x)
            and not (pc <= qc or xc <= pc) and not (po < qo or po > qc) and not (xo < po or xo > pc)
            and not q[1] > q[0] * 0.3 and not p[1] > p[0] * 0.3 and not x[1] > x[0] * 0.3
        ),
        "three_black_crows": bool(
            q[5] and p[5] and x[5] and big(q) and big(p) and big(x)
            and not (pc >= qc or xc >= pc) and not (po > qo or po < qc) and not (xo > po or xo < pc)
            and not q[2] > q[0] * 0.3 and not p[2] > p[0] * 0.3 and not x[2] > x[0] * 0.3
        ),
    }  # fmt: skip


class CandlestickTracker:
    """Incremental form of `candlestick_patterns`: feed closed bars, get the patterns they complete."""

    def __init__(self) -> None:
        self._last: list[Bar] = []

    def update(self, o: float, h: float, lo: float, c: float) -> dict[str, bool]:
        b: Bar = (o, h, lo, c)
        self._last = [*self._last[-2:], b]
        out = _single(b)
        pair = _pair(self._last[-2], b) if len(self._last) >= 2 else dict.fromkeys(_PAIR_NAMES, False)
        triple = (
            _triple(self._last[-3], self._last[-2], b)
            if len(self._last) == 3
            else dict.fromkeys(_TRIPLE_NAMES, False)
        )
        merged = {**out, **pair, **triple}
        return {name: merged[name] for name in PATTERNS}


_PAIR_NAMES = [n for n, k in PATTERNS.items() if k == 2]
_TRIPLE_NAMES = [n for n, k in PATTERNS.items() if k == 3]


def last_bar_patterns(open_: object, high: object, low: object, close: object) -> list[str]:
    """Names of the patterns completed by the most recent bar (what a strategy usually asks)."""
    pats = candlestick_patterns(open_, high, low, close)
    return [n for n, v in pats.items() if v.size and bool(v[-1])]
