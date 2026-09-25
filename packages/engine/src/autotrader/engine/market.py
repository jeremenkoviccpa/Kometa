"""MarketView backed by series buffers that only ever expose closed bars.

The same `SeriesBuffer` serves backtest (preloaded arrays, visibility advanced
by the clock) and live (rows appended as bars close). In both modes `last(n)`
returns a copy of at most the `visible` bars, so there is no API that reaches
a bar whose close_time is after `now`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from datetime import datetime

import numpy as np
import numpy.typing as npt

from autotrader.core.models import Timeframe
from autotrader.core.series import BarsArray, from_ns


class SeriesBuffer:
    def __init__(self, data: BarsArray | None = None, capacity: int = 1024) -> None:
        if data is not None:
            self._cols = {f.name: getattr(data, f.name) for f in fields(BarsArray)}
            self._size = len(data)
            self.visible = 0
        else:
            empty = BarsArray.empty()
            self._cols = {
                f.name: np.empty(capacity, dtype=getattr(empty, f.name).dtype) for f in fields(BarsArray)
            }
            self._size = 0
            self.visible = 0

    @property
    def size(self) -> int:
        return self._size

    def close_times(self) -> npt.NDArray[np.int64]:
        """Close times of all loaded bars (engine scheduling only; never handed to strategies)."""
        return np.asarray(self._cols["close_time"][: self._size], dtype=np.int64).copy()

    def close_time(self, i: int) -> int:
        return int(self._cols["close_time"][i])

    def advance_to(self, count: int) -> None:
        if count < self.visible or count > self._size:
            raise ValueError("visibility can only move forward within loaded data")
        self.visible = count

    def append(self, row: dict[str, float | int]) -> None:
        """Live mode: add one closed bar and make it visible."""
        if self._size == self._cols["open_time"].size:
            for k, v in self._cols.items():
                self._cols[k] = np.resize(v, max(16, v.size * 2))
        if self._size and int(row["close_time"]) <= int(self._cols["close_time"][self._size - 1]):
            raise ValueError("bars must be appended in close_time order")
        for k in self._cols:
            self._cols[k][self._size] = row[k]
        self._size += 1
        self.visible = self._size

    def last(self, n: int) -> BarsArray:
        if n < 0:
            raise ValueError("n must be >= 0")
        start = max(0, self.visible - n)
        return BarsArray(**{k: v[start : self.visible].copy() for k, v in self._cols.items()})

    def row(self, i: int) -> dict[str, float | int]:
        if not 0 <= i < self.visible:
            raise IndexError("bar not visible")
        return {k: v[i].item() for k, v in self._cols.items()}


class EngineMarketView:
    def __init__(self, spread_fn: Callable[[str, int], float]) -> None:
        self._series: dict[tuple[str, Timeframe], SeriesBuffer] = {}
        self._now_ns = 0
        self._spread_fn = spread_fn

    def add_series(self, symbol: str, tf: Timeframe, buf: SeriesBuffer) -> None:
        self._series[(symbol, tf)] = buf

    def series(self, symbol: str, tf: Timeframe) -> SeriesBuffer:
        return self._series[(symbol, tf)]

    def set_now(self, ns: int) -> None:
        if ns < self._now_ns:
            raise ValueError("market time cannot move backwards")
        self._now_ns = ns

    @property
    def now_ns(self) -> int:
        return self._now_ns

    @property
    def now(self) -> datetime:
        return from_ns(self._now_ns)

    def bars(self, symbol: str, tf: Timeframe, n: int) -> BarsArray:
        try:
            buf = self._series[(symbol, tf)]
        except KeyError:
            raise KeyError(f"not subscribed to {symbol} {tf}") from None
        return buf.last(n)

    def spread(self, symbol: str) -> float:
        return float(self._spread_fn(symbol, self._now_ns))
