"""Higher-timeframe bucket alignment, shared by historical aggregation and the live bar builder.

A trading day closes at a configurable local time (17:00 New York by default). Local time is shifted
so the day close becomes midnight, truncated to the timeframe (relative to the epoch, like polars
`dt.truncate` on naive datetimes), and shifted back. Both paths must produce identical buckets
(tests/unit/test_live_bars.py checks it against data.aggregate.resample, DST switches included).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

from autotrader.core.models import Timeframe
from autotrader.core.series import NS_PER_MINUTE, from_ns

DEFAULT_DAY_CLOSE = "17:00"
DEFAULT_DAY_TZ = "America/New_York"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC).replace(tzinfo=None)


@lru_cache(maxsize=16)
def _shift(day_close: str) -> timedelta:
    hh, mm = (int(x) for x in day_close.split(":"))
    return timedelta(days=1) - timedelta(hours=hh, minutes=mm)


def bucket_open_ns(
    open_ns: int, tf: Timeframe, *, day_close: str = DEFAULT_DAY_CLOSE, tz: str = DEFAULT_DAY_TZ
) -> int:
    """Open time (ns) of the `tf` bucket containing the M1 bar that opens at `open_ns`."""
    if tf == Timeframe.M1:
        return open_ns - open_ns % NS_PER_MINUTE
    local = from_ns(open_ns).astimezone(ZoneInfo(tz)).replace(tzinfo=None) + _shift(day_close)
    width = timedelta(minutes=tf.minutes)
    offset = (local - _EPOCH) % width
    return open_ns - (offset.days * 86_400 + offset.seconds) * 1_000_000_000 - offset.microseconds * 1_000
