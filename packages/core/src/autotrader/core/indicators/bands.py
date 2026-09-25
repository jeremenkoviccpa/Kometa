"""Bollinger, Keltner and Donchian channels."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

from autotrader.core.indicators._base import NAN, FloatArray, as_f64, check_period, nan_array, windows
from autotrader.core.indicators.ma import EMA, SMA, ema, sma
from autotrader.core.indicators.volatility import ATR, RollingStd, atr, rolling_std


class Band(NamedTuple):
    mid: FloatArray
    upper: FloatArray
    lower: FloatArray


def bollinger(close: object, n: int = 20, k: float = 2.0) -> Band:
    mid, sd = sma(close, n), rolling_std(close, n)
    return Band(mid, mid + k * sd, mid - k * sd)


def keltner(
    high: object, low: object, close: object, n: int = 20, atr_n: int = 10, mult: float = 2.0
) -> Band:
    mid, a = ema(close, n), atr(high, low, close, atr_n)
    return Band(mid, mid + mult * a, mid - mult * a)


def donchian(high: object, low: object, n: int = 20) -> Band:
    check_period(n)
    h, lo = as_f64(high), as_f64(low)
    up, dn = nan_array(h.size), nan_array(h.size)
    if h.size >= n:
        up[n - 1 :] = windows(h, n).max(axis=1)
        dn[n - 1 :] = windows(lo, n).min(axis=1)
    return Band((up + dn) / 2.0, up, dn)


class Bollinger:
    def __init__(self, n: int = 20, k: float = 2.0) -> None:
        self._sma, self._sd, self.k = SMA(n), RollingStd(n), k

    def update(self, close: float) -> tuple[float, float, float]:
        m, s = self._sma.update(close), self._sd.update(close)
        return m, m + self.k * s, m - self.k * s


class Keltner:
    def __init__(self, n: int = 20, atr_n: int = 10, mult: float = 2.0) -> None:
        self._ema, self._atr, self.mult = EMA(n), ATR(atr_n), mult

    def update(self, high: float, low: float, close: float) -> tuple[float, float, float]:
        m, a = self._ema.update(close), self._atr.update(high, low, close)
        return m, m + self.mult * a, m - self.mult * a


class Donchian:
    def __init__(self, n: int = 20) -> None:
        check_period(n)
        self.n = n
        self._h: deque[float] = deque(maxlen=n)
        self._l: deque[float] = deque(maxlen=n)

    def update(self, high: float, low: float) -> tuple[float, float, float]:
        self._h.append(float(high))
        self._l.append(float(low))
        if len(self._h) < self.n:
            return NAN, NAN, NAN
        up, dn = max(self._h), min(self._l)
        return (up + dn) / 2.0, up, dn
