"""Session and calendar helpers (spec section 6)."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from autotrader.core.models import CalendarEvent, SessionWindow
from autotrader.core.timeutil import ensure_utc

DEFAULT_SESSIONS: tuple[SessionWindow, ...] = (
    SessionWindow.model_validate({"name": "asia", "tz": "Asia/Tokyo", "start": "09:00", "end": "18:00"}),
    SessionWindow.model_validate({"name": "london", "tz": "Europe/London", "start": "08:00", "end": "17:00"}),
    SessionWindow.model_validate(
        {"name": "new_york", "tz": "America/New_York", "start": "08:00", "end": "17:00"}
    ),
)


@lru_cache(maxsize=64)
def _zone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def in_session(ts: datetime, window: SessionWindow) -> bool:
    local = ensure_utc(ts).astimezone(_zone(window.tz))
    if local.weekday() not in window.weekdays:
        return False
    t = local.time().replace(tzinfo=None)
    if window.start <= window.end:
        return window.start <= t < window.end
    return t >= window.start or t < window.end  # crosses midnight


def sessions_of(ts: datetime, windows: Iterable[SessionWindow] = DEFAULT_SESSIONS) -> tuple[str, ...]:
    return tuple(w.name for w in windows if in_session(ts, w))


class EventIndex:
    """Sorted event times per currency for fast 'minutes to next event' lookups."""

    def __init__(self, events: Sequence[CalendarEvent], impact: str = "high") -> None:
        by_ccy: dict[str, list[datetime]] = {}
        for e in events:
            if e.impact == impact:
                by_ccy.setdefault(e.currency, []).append(e.time)
        self._times = {k: sorted(v) for k, v in by_ccy.items()}

    def minutes_to_next(self, ts: datetime, currency: str) -> float | None:
        times = self._times.get(currency)
        if not times:
            return None
        i = bisect_left(times, ensure_utc(ts))
        return None if i == len(times) else (times[i] - ts).total_seconds() / 60.0

    def minutes_since_last(self, ts: datetime, currency: str) -> float | None:
        times = self._times.get(currency)
        if not times:
            return None
        i = bisect_left(times, ensure_utc(ts))
        # an event exactly at ts counts as "next" (0 minutes), not "last"
        return None if i == 0 else (ts - times[i - 1]).total_seconds() / 60.0

    def in_blackout(self, ts: datetime, currencies: Iterable[str], minutes: float) -> bool:
        for c in currencies:
            nxt, last = self.minutes_to_next(ts, c), self.minutes_since_last(ts, c)
            if (nxt is not None and nxt <= minutes) or (last is not None and last <= minutes):
                return True
        return False
