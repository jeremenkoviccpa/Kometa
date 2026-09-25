"""Clocks. The engine never calls datetime.now() directly; it asks its clock."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from autotrader.core.timeutil import ensure_utc


class Clock(Protocol):
    def now(self) -> datetime: ...


class SimClock:
    """Deterministic clock driven by the event loop. Time can only move forward."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, t: datetime) -> None:
        t = ensure_utc(t)
        if t < self._now:
            raise ValueError(f"SimClock cannot move backwards: {t} < {self._now}")
        self._now = t


class LiveClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
