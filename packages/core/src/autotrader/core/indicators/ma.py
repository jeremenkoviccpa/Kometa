"""Moving averages and slope. Output index i uses inputs [0, i] only; warmup is NaN."""

from __future__ import annotations

from collections import deque

import numpy as np

from autotrader.core.indicators._base import NAN, FloatArray, as_f64, check_period, nan_array, windows


def sma(x: object, n: int) -> FloatArray:
    check_period(n)
    a = as_f64(x)
    out = nan_array(a.size)
    if a.size >= n:
        out[n - 1 :] = windows(a, n).mean(axis=1)
    return out


def ema(x: object, n: int) -> FloatArray:
    """EMA with alpha = 2 / (n + 1), seeded by the SMA of the first n values."""
    check_period(n)
    a = as_f64(x)
    out = nan_array(a.size)
    if a.size < n:
        return out
    alpha = 2.0 / (n + 1.0)
    prev = float(a[:n].mean())
    vals = [prev]
    for v in a[n:].tolist():  # plain floats: same IEEE arithmetic as numpy scalars, ~5x faster
        prev = prev + alpha * (v - prev)
        vals.append(prev)
    out[n - 1 :] = vals
    return out


def wma(x: object, n: int) -> FloatArray:
    check_period(n)
    a = as_f64(x)
    out = nan_array(a.size)
    if a.size >= n:
        w = np.arange(1, n + 1, dtype=np.float64)
        out[n - 1 :] = windows(a, n) @ w / w.sum()
    return out


def slope(x: object, n: int) -> FloatArray:
    """Least-squares slope per bar over the last n values."""
    if n < 2:
        raise ValueError("slope needs n >= 2")
    a = as_f64(x)
    out = nan_array(a.size)
    if a.size >= n:
        t = np.arange(n, dtype=np.float64)
        tc = t - t.mean()
        out[n - 1 :] = windows(a, n) @ tc / float(tc @ tc)
    return out


class SMA:
    def __init__(self, n: int) -> None:
        check_period(n)
        self.n = n
        self._buf: deque[float] = deque(maxlen=n)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        if len(self._buf) < self.n:
            return NAN
        return float(np.mean(np.fromiter(self._buf, dtype=np.float64, count=self.n)))


class EMA:
    def __init__(self, n: int) -> None:
        check_period(n)
        self.n = n
        self.alpha = 2.0 / (n + 1.0)
        self._seed: list[float] = []
        self.value = NAN

    def update(self, x: float) -> float:
        x = float(x)
        if len(self._seed) < self.n:
            self._seed.append(x)
            if len(self._seed) == self.n:
                self.value = float(np.mean(np.asarray(self._seed, dtype=np.float64)))
            return self.value
        self.value = self.value + self.alpha * (x - self.value)
        return self.value


class WMA:
    def __init__(self, n: int) -> None:
        check_period(n)
        self.n = n
        self._w = np.arange(1, n + 1, dtype=np.float64)
        self._buf: deque[float] = deque(maxlen=n)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        if len(self._buf) < self.n:
            return NAN
        a = np.fromiter(self._buf, dtype=np.float64, count=self.n)
        return float(a @ self._w / self._w.sum())


class Slope:
    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError("slope needs n >= 2")
        self.n = n
        t = np.arange(n, dtype=np.float64)
        self._tc = t - t.mean()
        self._den = float(self._tc @ self._tc)
        self._buf: deque[float] = deque(maxlen=n)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        if len(self._buf) < self.n:
            return NAN
        a = np.fromiter(self._buf, dtype=np.float64, count=self.n)
        return float(a @ self._tc / self._den)
