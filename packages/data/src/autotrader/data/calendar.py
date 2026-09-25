"""Economic calendar sources (spec section 7). ForexFactory's weekly export is the first live provider
(open question 3); CSV and static sources serve backtests and tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
import polars as pl

from autotrader.core.models import CalendarEvent
from autotrader.core.timeutil import ensure_utc


class CalendarSource(Protocol):
    name: str

    def events(self, start: datetime, end: datetime) -> list[CalendarEvent]: ...


class StaticCalendar:
    """In-memory events; used by tests and backtests with a pre-fetched calendar."""

    name = "static"

    def __init__(self, events: list[CalendarEvent]) -> None:
        self._events = sorted(events, key=lambda e: (e.time, e.currency, e.name))

    def events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        s, e = ensure_utc(start), ensure_utc(end)
        return [ev for ev in self._events if s <= ev.time < e]


class CsvCalendar(StaticCalendar):
    """CSV with columns time (ISO 8601 with offset), currency, impact, name."""

    name = "csv"

    def __init__(self, path: Path) -> None:
        df = pl.read_csv(path)
        events = [
            CalendarEvent(
                time=datetime.fromisoformat(r["time"]),
                currency=r["currency"],
                impact=r["impact"],
                name=r["name"],
            )
            for r in df.iter_rows(named=True)
        ]
        super().__init__(events)


FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_IMPACT = {"High": "high", "Medium": "medium", "Low": "low"}  # Holiday and Non-Economic are not events


class ForexFactoryCalendar:
    """ForexFactory's weekly calendar export (the feed ForexFactory publishes for tools; the website itself
    is never scraped). The feed covers the current week and asks callers not to poll it often, so it is
    fetched at most once per `min_interval` and cached on disk; `fresh` tells whether the cache is recent
    enough to trade on. A failed fetch keeps the previous cache."""

    name = "forexfactory"

    def __init__(
        self,
        cache: Path,
        *,
        max_age: timedelta = timedelta(hours=12),
        min_interval: timedelta = timedelta(minutes=30),
        fetch: Callable[[], bytes] | None = None,
    ) -> None:
        self.cache = cache
        self.max_age = max_age
        self.min_interval = min_interval
        self._fetch = fetch or _http_get
        self.fetched_at: datetime | None = None
        self._last_try: datetime | None = None
        self.rows: list[dict[str, Any]] = []
        self._events: list[CalendarEvent] = []
        if cache.exists():
            try:
                doc = json.loads(cache.read_text())
                self._load(doc["feed"], datetime.fromisoformat(doc["fetched_at"]))
            except (OSError, ValueError, KeyError):
                pass  # a broken cache is just a missing one

    def refresh(self, now: datetime) -> bool:
        """Fetch the feed (rate limited). True when the calendar was updated."""
        now = ensure_utc(now)
        if self._last_try is not None and now - self._last_try < self.min_interval:
            return False
        self._last_try = now
        try:
            raw = json.loads(self._fetch())
            if not isinstance(raw, list):
                raise ValueError("feed is not a list")
            self._load(raw, now)
        except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
            return False
        tmp = self.cache.with_suffix(".tmp")
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"fetched_at": now.isoformat(), "feed": raw}))  # as fetched: one parser
        tmp.replace(self.cache)
        return True

    def _load(self, raw: list[dict[str, Any]], fetched_at: datetime) -> None:
        rows, events = [], []
        for r in raw:
            ccy, impact = str(r.get("country", "")), FF_IMPACT.get(str(r.get("impact", "")))
            if len(ccy) != 3 or not ccy.isalpha() or not ccy.isupper():
                continue
            t = ensure_utc(datetime.fromisoformat(str(r["date"])))  # the feed carries its offset
            rows.append(
                {
                    "time": t.isoformat(),
                    "currency": ccy,
                    "impact": impact or str(r.get("impact", "")).lower(),
                    "name": str(r.get("title", "")),
                    "forecast": str(r.get("forecast", "")),
                    "previous": str(r.get("previous", "")),
                }
            )
            if impact is not None:
                events.append(
                    CalendarEvent(time=t, currency=ccy, impact=impact, name=str(r.get("title", "")))
                )
        self.rows = sorted(rows, key=lambda x: x["time"])
        self._events = sorted(events, key=lambda e: (e.time, e.currency, e.name))
        self.fetched_at = ensure_utc(fetched_at)

    def fresh(self, now: datetime) -> bool:
        return self.fetched_at is not None and ensure_utc(now) - self.fetched_at <= self.max_age

    def events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        s, e = ensure_utc(start), ensure_utc(end)
        return [ev for ev in self._events if s <= ev.time < e]


def _http_get() -> bytes:
    r = httpx.get(
        FF_URL, timeout=15, headers={"User-Agent": "kometa-autotrader/0.1 (calendar export reader)"}
    )
    r.raise_for_status()
    return r.content


_REGISTRY: dict[str, Callable[..., CalendarSource]] = {
    "static": StaticCalendar,
    "csv": CsvCalendar,
    "forexfactory": ForexFactoryCalendar,
}


def make_calendar(name: str, **kwargs: object) -> CalendarSource:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"unknown calendar source {name!r}; known: {sorted(_REGISTRY)}") from None
