"""MT5 timestamps are broker server wall-clock seconds written as if they were UTC.

The server timezone is never guessed (Phase 1 lesson): the bridge refuses to start without it.
Accepted forms:
- an IANA zone, e.g. "Europe/Athens" or "Etc/GMT-2";
- "NY+<h>", the common "New York close" convention: server time = New York local time + h hours,
  so the server day ends at 17:00 New York all year (UTC+2 in winter, UTC+3 in summer with "NY+7").
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from autotrader.core.timeutil import ensure_utc

_NY = ZoneInfo("America/New_York")
_EPOCH = datetime.fromtimestamp(0, UTC).replace(tzinfo=None)  # naive, for server wall-clock seconds
_NY_RE = re.compile(r"^NY\+(\d{1,2})$")


class ServerTimeZone:
    def __init__(self, spec: str) -> None:
        if not spec:
            raise ValueError("MT5 server timezone is required (e.g. 'NY+7' or an IANA zone)")
        self.spec = spec
        m = _NY_RE.match(spec)
        # "NY+h" = New York wall clock shifted by h hours; otherwise the IANA zone's wall clock
        self._shift = timedelta(hours=int(m.group(1))) if m else timedelta(0)
        self._zone = _NY if m else ZoneInfo(spec)

    def to_utc(self, server_seconds: float) -> datetime:
        wall = _EPOCH + timedelta(seconds=server_seconds)  # naive server wall clock
        return (wall - self._shift).replace(tzinfo=self._zone).astimezone(UTC)

    def to_server(self, t: datetime) -> datetime:
        """UTC -> naive server wall clock, for MT5 request arguments."""
        return ensure_utc(t).astimezone(self._zone).replace(tzinfo=None) + self._shift

    def to_server_seconds(self, t: datetime) -> int:
        return int((self.to_server(t) - _EPOCH).total_seconds())
