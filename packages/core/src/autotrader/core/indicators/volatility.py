"""True range, ATR (Wilder), rolling standard deviation, ATR percentile."""

from __future__ import annotations

from collections import deque

import numpy as np

from autotrader.core.indicators._base import NAN, FloatArray, as_f64, check_period, nan_array, windows


def true_range(high: object, low: object, close: object) -> FloatArray:
    h, lo, c = as_f64(high), as_f64(low), as_f64(close)
    tr = h - lo
    if tr.size > 1:
        prev_c = c[:-1]
        tr[1:] = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - prev_c), np.abs(lo[1:] - prev_c)])
    return tr


def wilder(x: object, n: int) -> FloatArray:
    """Wilder smoothing: seed with the mean of the first n values, then (prev*(n-1)+x)/n."""
    check_period(n)
    a = as_f64(x)
    out = nan_array(a.size)
    if a.size < n:
        return out
    prev = float(a[:n].mean())
    vals = [prev]
    for v in a[n:].tolist():
        prev = (prev * (n - 1) + v) / n
        vals.append(prev)
    out[n - 1 :] = vals
    return out


def atr(high: object, low: object, close: object, n: int = 14) -> FloatArray:
    return wilder(true_range(high, low, close), n)


def rolling_std(x: object, n: int) -> FloatArray:
    """Population standard deviation (ddof=0), the Bollinger convention."""
    check_period(n)
    a = as_f64(x)
    out = nan_array(a.size)
    if a.size >= n:
        out[n - 1 :] = windows(a, n).std(axis=1)
    return out


def percentile_rank(x: object, lookback: int, chunk: int = 65536) -> FloatArray:
    """Fraction of the last `lookback` values (current included) that are <= current.

    NaN until `lookback` consecutive finite values are available; a NaN resets the count.
    """
    check_period(lookback, "lookback")
    a = as_f64(x)
    out = nan_array(a.size)
    finite = ~np.isnan(a)
    # contiguous finite runs
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
    for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
        seg = a[start:stop]
        if seg.size < lookback:
            continue
        w = windows(seg, lookback)
        for c0 in range(0, w.shape[0], chunk):
            wc = w[c0 : c0 + chunk]
            ranks = np.count_nonzero(wc <= wc[:, -1:], axis=1) / lookback
            out[start + lookback - 1 + c0 : start + lookback - 1 + c0 + wc.shape[0]] = ranks
    return out


def atr_percentile(high: object, low: object, close: object, n: int = 14, lookback: int = 250) -> FloatArray:
    return percentile_rank(atr(high, low, close, n), lookback)


class TrueRange:
    def __init__(self) -> None:
        self._prev_close = NAN

    def update(self, high: float, low: float, close: float) -> float:
        tr = high - low
        if not np.isnan(self._prev_close):
            tr = max(tr, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        return float(tr)


class Wilder:
    def __init__(self, n: int) -> None:
        check_period(n)
        self.n = n
        self._seed: list[float] = []
        self.value = NAN

    def update(self, x: float) -> float:
        if len(self._seed) < self.n:
            self._seed.append(float(x))
            if len(self._seed) == self.n:
                self.value = float(np.mean(np.asarray(self._seed, dtype=np.float64)))
            return self.value
        self.value = (self.value * (self.n - 1) + float(x)) / self.n
        return self.value


class ATR:
    def __init__(self, n: int = 14) -> None:
        self._tr = TrueRange()
        self._w = Wilder(n)

    def update(self, high: float, low: float, close: float) -> float:
        return self._w.update(self._tr.update(high, low, close))


class RollingStd:
    def __init__(self, n: int) -> None:
        check_period(n)
        self.n = n
        self._buf: deque[float] = deque(maxlen=n)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        if len(self._buf) < self.n:
            return NAN
        return float(np.fromiter(self._buf, dtype=np.float64, count=self.n).std())


class PercentileRank:
    def __init__(self, lookback: int) -> None:
        check_period(lookback, "lookback")
        self.lookback = lookback
        self._buf: deque[float] = deque(maxlen=lookback)

    def update(self, x: float) -> float:
        if np.isnan(x):
            self._buf.clear()
            return NAN
        self._buf.append(float(x))
        if len(self._buf) < self.lookback:
            return NAN
        w = np.fromiter(self._buf, dtype=np.float64, count=self.lookback)
        return float(np.count_nonzero(w <= x)) / self.lookback


class ATRPercentile:
    def __init__(self, n: int = 14, lookback: int = 250) -> None:
        self._atr = ATR(n)
        self._rank = PercentileRank(lookback)

    def update(self, high: float, low: float, close: float) -> float:
        return self._rank.update(self._atr.update(high, low, close))
