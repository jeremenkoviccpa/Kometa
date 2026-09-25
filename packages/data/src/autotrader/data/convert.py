"""Polars bar frames -> core BarsArray, and spread statistics by hour of week."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from autotrader.core.series import BarsArray
from autotrader.data.schema import PRICE_COLS, validate_frame


def to_bars_array(df: pl.DataFrame, bar_minutes: int = 1) -> BarsArray:
    """Frame with open_time (and optionally close_time) -> BarsArray."""
    close = (
        df["close_time"] if "close_time" in df.columns else (df["open_time"] + timedelta(minutes=bar_minutes))
    )
    frame = validate_frame(df)
    ot = frame["open_time"].dt.epoch("ns").to_numpy().astype(np.int64)
    ct = close.dt.epoch("ns").to_numpy().astype(np.int64)
    cols = {c: frame[c].to_numpy().astype(np.float64) for c in (*PRICE_COLS, "volume")}
    return BarsArray(open_time=ot, close_time=ct, **cols)


def spread_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Median and 90th percentile close spread per hour of week (0 = Monday 00:00 UTC)."""
    frame = validate_frame(df)
    return (
        frame.with_columns(
            (
                (pl.col("open_time").dt.weekday().cast(pl.Int32) - 1) * 24
                + pl.col("open_time").dt.hour().cast(pl.Int32)
            ).alias("hour_of_week"),
            (pl.col("ask_c") - pl.col("bid_c")).alias("spread"),
        )
        .group_by("hour_of_week")
        .agg(
            pl.col("spread").median().alias("median_spread"),
            pl.col("spread").quantile(0.9, "nearest").alias("p90_spread"),
            pl.len().alias("samples"),
        )
        .sort("hour_of_week")
    )


def median_spread_by_hour(df: pl.DataFrame) -> np.ndarray:
    """168-array of median spreads; hours without data get the overall median."""
    stats = spread_stats(df)
    frame = validate_frame(df)
    med = (frame["ask_c"] - frame["bid_c"]).median()
    overall = float(med) if isinstance(med, (int, float)) else 0.0
    out = np.full(168, overall, dtype=np.float64)
    out[stats["hour_of_week"].to_numpy()] = stats["median_spread"].to_numpy()
    return out
