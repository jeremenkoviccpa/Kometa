"""RSI (Wilder), MACD, stochastic."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from autotrader.core.indicators._base import NAN, FloatArray, as_f64, check_period, nan_array, windows
from autotrader.core.indicators.ma import EMA, SMA, ema, sma


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def rsi(close: object, n: int = 14) -> FloatArray:
    check_period(n)
    c = as_f64(close)
    out = nan_array(c.size)
    if c.size <= n:
        return out
    d = np.diff(c)
    gains, losses = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = float(gains[:n].mean()), float(losses[:n].mean())
    vals = [_rsi_value(ag, al)]
    for g, lo in zip(gains[n:].tolist(), losses[n:].tolist(), strict=True):
        ag = (ag * (n - 1) + g) / n
        al = (al * (n - 1) + lo) / n
        vals.append(_rsi_value(ag, al))
    out[n:] = vals
    return out


class MACDResult(NamedTuple):
    macd: FloatArray
    signal: FloatArray
    hist: FloatArray


def macd(close: object, fast: int = 12, slow: int = 26, signal: int = 9) -> MACDResult:
    if fast >= slow:
        raise ValueError("fast must be < slow")
    c = as_f64(close)
    line = ema(c, fast) - ema(c, slow)
    sig = nan_array(c.size)
    start = slow - 1
    if c.size > start:
        sig[start:] = ema(line[start:], signal)
    return MACDResult(line, sig, line - sig)


class StochResult(NamedTuple):
    k: FloatArray
    d: FloatArray


def _stoch_k(c: float, hh: float, ll: float) -> float:
    rng = hh - ll
    return 50.0 if rng == 0.0 else 100.0 * (c - ll) / rng


def stochastic(high: object, low: object, close: object, k: int = 14, d: int = 3) -> StochResult:
    check_period(k, "k")
    check_period(d, "d")
    h, lo, c = as_f64(high), as_f64(low), as_f64(close)
    kk = nan_array(c.size)
    if c.size >= k:
        hh, ll = windows(h, k).max(axis=1), windows(lo, k).min(axis=1)
        rng = hh - ll
        with np.errstate(divide="ignore", invalid="ignore"):
            kk[k - 1 :] = np.where(rng == 0.0, 50.0, 100.0 * (c[k - 1 :] - ll) / rng)
    dd = nan_array(c.size)
    if c.size >= k:
        dd[k - 1 :] = sma(kk[k - 1 :], d)
    return StochResult(kk, dd)


class RSI:
    def __init__(self, n: int = 14) -> None:
        check_period(n)
        self.n = n
        self._prev = NAN
        self._g: list[float] = []
        self._l: list[float] = []
        self._ag = NAN
        self._al = NAN

    def update(self, close: float) -> float:
        prev, self._prev = self._prev, float(close)
        if np.isnan(prev):
            return NAN
        change = float(close) - prev
        g, lo = max(change, 0.0), max(-change, 0.0)
        if len(self._g) < self.n:
            self._g.append(g)
            self._l.append(lo)
            if len(self._g) < self.n:
                return NAN
            self._ag = float(np.mean(np.asarray(self._g, dtype=np.float64)))
            self._al = float(np.mean(np.asarray(self._l, dtype=np.float64)))
        else:
            self._ag = (self._ag * (self.n - 1) + g) / self.n
            self._al = (self._al * (self.n - 1) + lo) / self.n
        return _rsi_value(self._ag, self._al)


class MACD:
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self._fast, self._slow, self._sig = EMA(fast), EMA(slow), EMA(signal)

    def update(self, close: float) -> tuple[float, float, float]:
        f, s = self._fast.update(close), self._slow.update(close)
        line = f - s
        if np.isnan(line):
            return NAN, NAN, NAN
        sig = self._sig.update(line)
        return line, sig, line - sig


class Stochastic:
    def __init__(self, k: int = 14, d: int = 3) -> None:
        check_period(k, "k")
        self.k = k
        self._h: deque[float] = deque(maxlen=k)
        self._l: deque[float] = deque(maxlen=k)
        self._d = SMA(d)

    def update(self, high: float, low: float, close: float) -> tuple[float, float]:
        self._h.append(float(high))
        self._l.append(float(low))
        if len(self._h) < self.k:
            return NAN, NAN
        kv = _stoch_k(float(close), max(self._h), min(self._l))
        return kv, self._d.update(kv)
