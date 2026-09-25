"""Indicators against hand-computed reference values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autotrader.core import indicators as ind


def test_sma_ema_wma_small() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(ind.sma(x, 3), [np.nan, np.nan, 2.0, 3.0, 4.0], equal_nan=True)
    # EMA(3): alpha 0.5, seed mean(1,2,3)=2 -> 3 -> 4
    np.testing.assert_allclose(ind.ema(x, 3), [np.nan, np.nan, 2.0, 3.0, 4.0], equal_nan=True)
    # WMA(3) at i=2: (1*1+2*2+3*3)/6 = 14/6
    assert ind.wma(x, 3)[2] == pytest.approx(14 / 6)


def test_slope_of_line() -> None:
    x = 2.5 * np.arange(20) + 7
    assert ind.slope(x, 5)[-1] == pytest.approx(2.5)


def test_true_range_uses_previous_close() -> None:
    h, lo, c = [10.0, 12.0], [9.0, 11.5], [9.5, 12.0]
    np.testing.assert_allclose(ind.true_range(h, lo, c), [1.0, 2.5])


def test_atr_constant_range() -> None:
    n = 50
    c = np.full(n, 100.0)
    atr = ind.atr(c + 1, c - 1, c, 14)
    assert math.isnan(atr[12])
    np.testing.assert_allclose(atr[13:], 2.0)


def test_rsi_extremes() -> None:
    up = np.arange(30, dtype=float)
    assert ind.rsi(up, 14)[-1] == 100.0
    assert ind.rsi(up[::-1], 14)[-1] == pytest.approx(0.0)
    assert ind.rsi(np.full(30, 5.0), 14)[-1] == 50.0


def test_rsi_first_value_hand_computed() -> None:
    # n=2, changes +1, -1 -> ag=0.5, al=0.5 -> RSI 50; then +2: ag=1.25, al=0.25 -> 100-100/6
    r = ind.rsi([10.0, 11.0, 10.0, 12.0], 2)
    assert r[2] == pytest.approx(50.0)
    assert r[3] == pytest.approx(100 - 100 / 6)


def test_stochastic_bounds_and_flat() -> None:
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(size=200))
    k, _ = ind.stochastic(c + 1, c - 1, c)
    finite = k[~np.isnan(k)]
    assert finite.min() >= 0 and finite.max() <= 100
    kf, _ = ind.stochastic(np.ones(20), np.ones(20), np.ones(20))
    assert kf[-1] == 50.0


def test_bands_ordering() -> None:
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(size=300))
    for band in (ind.bollinger(c), ind.keltner(c + 1, c - 1, c), ind.donchian(c + 1, c - 1)):
        m = ~np.isnan(band.upper)
        assert np.all(band.upper[m] >= band.mid[m]) and np.all(band.mid[m] >= band.lower[m])


def test_swing_confirmed_late_and_single_on_flat_top() -> None:
    h = np.array([1, 2, 5, 5, 2, 1, 1], dtype=float)
    lo = h - 0.5
    s = ind.swings(h, lo, left=2, right=2)
    # bar 2 (first 5) is the swing high, confirmed at bar 4; the equal bar 3 is not a second swing
    assert s.high[4] == 5.0
    assert np.count_nonzero(~np.isnan(s.high)) == 1


def test_trend_state_uptrend() -> None:
    # zig-zag with rising peaks and troughs
    base = np.tile([0, 2, 4, 2], 12).astype(float) + np.repeat(np.arange(12), 4) * 1.0
    ts = ind.trend_state(base + 0.1, base - 0.1, 1, 1)
    assert ts[-1] == 1.0
    ts_down = ind.trend_state(-base + 0.1, -base - 0.1, 1, 1)
    assert ts_down[-1] == -1.0


def test_candle_patterns() -> None:
    o = [10.0, 9.0]
    c = [9.5, 10.5]
    h = [10.2, 10.6]
    lo = [9.4, 8.9]
    p = ind.patterns(o, h, lo, c)
    assert p.bull_engulfing[1] and p.outside[1] and not p.inside[1]
    a = ind.anatomy([1.0], [1.0], [1.0], [1.0])
    assert math.isnan(a.body_ratio[0])


def test_invalid_periods() -> None:
    with pytest.raises(ValueError, match="n must be"):
        ind.sma([1.0], 0)
    with pytest.raises(ValueError, match="fast must be"):
        ind.macd([1.0], 26, 12)


def test_reversal_candles_on_textbook_shapes() -> None:
    """Each pattern on a hand-built bar sequence, true exactly at the completing bar; both forms agree."""
    bars = [  # (open, high, low, close)
        (100.0, 100.5, 96.0, 100.2),  # 0 hammer: 4.5 range, lower wick 4.0 (89%), body 0.2, nose 0.3
        (100.0, 104.0, 99.8, 100.1),  # 1 shooting star
        (100.0, 101.0, 99.0, 100.05),  # 2 doji
        (104.0, 104.2, 100.0, 100.2),  # 3 big down bar
        (100.0, 100.6, 99.4, 100.1),  # 4 small bar
        (100.2, 103.5, 100.1, 103.2),  # 5 up bar above the 3rd bar's midpoint (102.1): morning star
        (100.0, 104.2, 99.8, 104.0),  # 6 big up bar
        (104.0, 104.6, 103.4, 104.1),  # 7 small
        (103.8, 103.9, 100.5, 100.9),  # 8 down bar below the midpoint (102.0): evening star
    ]
    o, h, lo, c = (np.array(x) for x in zip(*bars, strict=True))
    r = ind.reversals(o, h, lo, c)
    assert np.flatnonzero(r.bull_pin).tolist() == [0]
    assert np.flatnonzero(r.bear_pin).tolist() == [1]
    assert 2 in np.flatnonzero(r.doji).tolist()
    assert np.flatnonzero(r.morning_star).tolist() == [5]
    assert np.flatnonzero(r.evening_star).tolist() == [8]
    t = ind.ReversalTracker()
    inc = np.array([t.update(*b) for b in bars]).T
    for vec, got in zip(r, inc, strict=True):
        assert vec.tolist() == got.tolist()
