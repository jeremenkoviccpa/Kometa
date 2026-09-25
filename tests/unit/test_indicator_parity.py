"""Every indicator: vectorized == incremental, and output[:T] ignores data after T."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from autotrader.core import indicators as ind

RTOL = 1e-9
ATOL = 1e-9


def ohlc(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.05, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    # sprinkle flat bars and exact repeats to exercise ties
    h[::37] = np.maximum(o[::37], c[::37])
    lo[::37] = h[::37]
    o[::37] = c[::37] = h[::37]
    return o, h, lo, c


def _candle_step(s: Any, o: float, h: float, l: float, c: float) -> tuple[float, ...]:
    vals, pats = s.update(o, h, l, c)
    return (*vals, *(float(x) for x in pats))


# name -> (vectorized(o,h,l,c) -> tuple of arrays, factory, step(obj, o,h,l,c) -> tuple of floats)
Case = tuple[Callable[..., tuple[Any, ...]], Callable[[], Any], Callable[..., tuple[float, ...]]]

CASES: dict[str, Case] = {
    "sma": (lambda o, h, l, c: (ind.sma(c, 10),), lambda: ind.SMA(10), lambda s, o, h, l, c: (s.update(c),)),
    "ema": (lambda o, h, l, c: (ind.ema(c, 10),), lambda: ind.EMA(10), lambda s, o, h, l, c: (s.update(c),)),
    "wma": (lambda o, h, l, c: (ind.wma(c, 7),), lambda: ind.WMA(7), lambda s, o, h, l, c: (s.update(c),)),
    "slope": (
        lambda o, h, l, c: (ind.slope(c, 9),),
        lambda: ind.Slope(9),
        lambda s, o, h, l, c: (s.update(c),),
    ),
    "true_range": (
        lambda o, h, l, c: (ind.true_range(h, l, c),),
        ind.TrueRange,
        lambda s, o, h, l, c: (s.update(h, l, c),),
    ),
    "atr": (
        lambda o, h, l, c: (ind.atr(h, l, c, 14),),
        lambda: ind.ATR(14),
        lambda s, o, h, l, c: (s.update(h, l, c),),
    ),
    "rolling_std": (
        lambda o, h, l, c: (ind.rolling_std(c, 20),),
        lambda: ind.RollingStd(20),
        lambda s, o, h, l, c: (s.update(c),),
    ),
    "atr_percentile": (
        lambda o, h, l, c: (ind.atr_percentile(h, l, c, 14, 50),),
        lambda: ind.ATRPercentile(14, 50),
        lambda s, o, h, l, c: (s.update(h, l, c),),
    ),
    "rsi": (lambda o, h, l, c: (ind.rsi(c, 14),), lambda: ind.RSI(14), lambda s, o, h, l, c: (s.update(c),)),
    "macd": (lambda o, h, l, c: tuple(ind.macd(c)), ind.MACD, lambda s, o, h, l, c: s.update(c)),
    "stochastic": (
        lambda o, h, l, c: tuple(ind.stochastic(h, l, c, 14, 3)),
        lambda: ind.Stochastic(14, 3),
        lambda s, o, h, l, c: s.update(h, l, c),
    ),
    "bollinger": (
        lambda o, h, l, c: tuple(ind.bollinger(c)),
        ind.Bollinger,
        lambda s, o, h, l, c: s.update(c),
    ),
    "keltner": (
        lambda o, h, l, c: tuple(ind.keltner(h, l, c)),
        ind.Keltner,
        lambda s, o, h, l, c: s.update(h, l, c),
    ),
    "donchian": (
        lambda o, h, l, c: tuple(ind.donchian(h, l)),
        ind.Donchian,
        lambda s, o, h, l, c: s.update(h, l),
    ),
    "swings": (
        lambda o, h, l, c: tuple(ind.swings(h, l, 3, 2)),
        lambda: ind.SwingDetector(3, 2),
        lambda s, o, h, l, c: s.update(h, l),
    ),
    "trend_state": (
        lambda o, h, l, c: (ind.trend_state(h, l, 2, 2),),
        lambda: ind.TrendState(2, 2),
        lambda s, o, h, l, c: (s.update(h, l),),
    ),
    "candles": (
        lambda o, h, l, c: (*ind.anatomy(o, h, l, c), *(p.astype(float) for p in ind.patterns(o, h, l, c))),
        ind.CandleTracker,
        _candle_step,
    ),
    "reversals": (
        lambda o, h, l, c: tuple(p.astype(float) for p in ind.reversals(o, h, l, c)),
        ind.ReversalTracker,
        lambda s, o, h, l, c: tuple(float(x) for x in s.update(o, h, l, c)),
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_vectorized_matches_incremental(name: str, seed: int) -> None:
    vec, factory, step = CASES[name]
    o, h, lo, c = ohlc(400, seed)
    expected = vec(o, h, lo, c)
    obj = factory()
    got = np.array([step(obj, *map(float, bar)) for bar in zip(o, h, lo, c, strict=True)]).T
    assert len(expected) == got.shape[0]
    for e, g in zip(expected, got, strict=True):
        np.testing.assert_allclose(g, e, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_lookahead(name: str) -> None:
    """Future poisoning at indicator level: garbage after T must not change output up to T."""
    vec, _, _ = CASES[name]
    o, h, lo, c = ohlc(300, 7)
    t = 180
    clean = vec(o, h, lo, c)
    rng = np.random.default_rng(99)
    po, ph, pl, pc = (a.copy() for a in (o, h, lo, c))
    for a in (po, ph, pl, pc):
        a[t + 1 :] = rng.uniform(-1e6, 1e6, a.size - t - 1)
    poisoned = vec(po, ph, pl, pc)
    for a, b in zip(clean, poisoned, strict=True):
        np.testing.assert_array_equal(np.asarray(a)[: t + 1], np.asarray(b)[: t + 1], err_msg=name)
