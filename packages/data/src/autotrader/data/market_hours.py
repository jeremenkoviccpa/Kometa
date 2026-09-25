"""Weekly market hours in a local timezone, e.g. FX: Sunday 17:00 to Friday 17:00 New York."""

from __future__ import annotations

from datetime import time

import polars as pl
from pydantic import BaseModel, ConfigDict

_DAYS = {"MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6, "SUN": 7}  # polars weekday numbering


def _minute_of_week(weekday: int, t: time) -> int:
    """Minutes since Monday 00:00, with polars weekday numbering (Mon=1)."""
    return (weekday - 1) * 1440 + t.hour * 60 + t.minute


class MarketHours(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tz: str = "America/New_York"
    weekly_open: str = "SUN 17:00"
    weekly_close: str = "FRI 17:00"
    daily_breaks: tuple[tuple[time, time], ...] = ()

    def _parse(self, spec: str) -> int:
        day, hhmm = spec.split()
        h, m = hhmm.split(":")
        return _minute_of_week(_DAYS[day.upper()], time(int(h), int(m)))

    def is_open_expr(self, col: str = "open_time") -> pl.Expr:
        """Polars expression: True where the minute starting at `col` is inside market hours."""
        local = pl.col(col).dt.convert_time_zone(self.tz)
        mow = (
            (local.dt.weekday().cast(pl.Int32) - 1) * 1440
            + local.dt.hour().cast(pl.Int32) * 60
            + (local.dt.minute().cast(pl.Int32))
        )
        o, c = self._parse(self.weekly_open), self._parse(self.weekly_close)
        # week wraps (open Sunday, close Friday): open if after open OR before close
        is_open = (mow >= o) | (mow < c) if o > c else (mow >= o) & (mow < c)
        mod = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)
        for start, end in self.daily_breaks:
            s, e = start.hour * 60 + start.minute, end.hour * 60 + end.minute
            is_open = is_open & ~((mod >= s) & (mod < e))
        return is_open


FX_HOURS = MarketHours()
METAL_HOURS = MarketHours(daily_breaks=((time(17, 0), time(18, 0)),))
