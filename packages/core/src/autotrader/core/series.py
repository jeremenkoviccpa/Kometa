"""Columnar bar arrays: the numpy form of a bar series used by the engine and strategies.

Times are int64 nanoseconds since the Unix epoch, UTC. Instances handed to
strategies are always copies, so no strategy can reach data beyond what the
MarketView exposes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]

NS_PER_MINUTE = 60_000_000_000


def to_ns(t: datetime) -> int:
    if t.tzinfo is None:
        raise ValueError("naive datetime")
    delta = t.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def from_ns(ns: int) -> datetime:
    s, rem = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(s, tz=UTC).replace(microsecond=rem // 1_000)


@dataclass(frozen=True)
class BarsArray:
    open_time: I64
    close_time: I64
    bid_o: F64
    bid_h: F64
    bid_l: F64
    bid_c: F64
    ask_o: F64
    ask_h: F64
    ask_l: F64
    ask_c: F64
    volume: F64

    def __len__(self) -> int:
        return int(self.open_time.size)

    def __post_init__(self) -> None:
        n = self.open_time.size
        for f in fields(self):
            if getattr(self, f.name).shape != (n,):
                raise ValueError(f"column {f.name} has wrong shape")

    def slice(self, start: int, stop: int) -> BarsArray:
        """Copying slice (never a view)."""
        return BarsArray(*(getattr(self, f.name)[start:stop].copy() for f in fields(self)))

    @property
    def mid_c(self) -> F64:
        return (self.bid_c + self.ask_c) / 2.0

    @staticmethod
    def empty() -> BarsArray:
        i, f = np.empty(0, np.int64), np.empty(0, np.float64)
        return BarsArray(i, i, f, f, f, f, f, f, f, f, f)
