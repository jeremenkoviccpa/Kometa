"""Build higher timeframes from M1 in memory (the DB uses continuous aggregates).

Buckets are aligned to a "trading day" that closes at a configurable local
time, 17:00 New York by default: we shift local time so the day close becomes
midnight, truncate, and shift back. Incomplete buckets at the end of the data
are dropped so a bar is never visible before its period has ended.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from autotrader.core.models import Timeframe
from autotrader.data.schema import validate_frame

_EVERY = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}


def resample(
    m1: pl.DataFrame,
    tf: Timeframe,
    *,
    day_close: str = "17:00",
    tz: str = "America/New_York",
    data_end: datetime | None = None,
) -> pl.DataFrame:
    """Aggregate M1 bars; `data_end` is the close_time of the last M1 bar (default: inferred)."""
    df = validate_frame(m1).unique(subset="open_time", keep="first").sort("open_time")
    if tf == Timeframe.M1 or df.height == 0:
        return df.with_columns((pl.col("open_time") + pl.duration(minutes=1)).alias("close_time"))
    hh, mm = (int(x) for x in day_close.split(":"))
    shift = timedelta(days=1) - timedelta(hours=hh, minutes=mm)
    shifted = pl.col("open_time").dt.convert_time_zone(tz).dt.replace_time_zone(None) + shift
    offset_in_bucket = shifted - shifted.dt.truncate(_EVERY[tf])
    out = (
        df.with_columns((pl.col("open_time") - offset_in_bucket).alias("_bucket"))
        .group_by("_bucket", maintain_order=True)
        .agg(
            pl.col("bid_o").first(),
            pl.col("bid_h").max(),
            pl.col("bid_l").min(),
            pl.col("bid_c").last(),
            pl.col("ask_o").first(),
            pl.col("ask_h").max(),
            pl.col("ask_l").min(),
            pl.col("ask_c").last(),
            pl.col("volume").sum(),
        )
        .rename({"_bucket": "open_time"})
        .with_columns((pl.col("open_time") + pl.duration(minutes=tf.minutes)).alias("close_time"))
    )
    end = data_end or (df["open_time"][-1] + timedelta(minutes=1))
    return out.filter(pl.col("close_time") <= end)
