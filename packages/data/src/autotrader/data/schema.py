"""Canonical in-memory bar frame (polars). Mirrors the `bars_m1` hypertable."""

from __future__ import annotations

import polars as pl

PRICE_COLS = ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c")

BAR_SCHEMA: dict[str, pl.DataType] = {
    "open_time": pl.Datetime("us", "UTC"),
    **{c: pl.Float64() for c in PRICE_COLS},
    "volume": pl.Float64(),
}


def validate_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce to the canonical schema and column order; raise on missing columns."""
    missing = [c for c in BAR_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"bar frame missing columns: {missing}")
    if df.schema["open_time"] != BAR_SCHEMA["open_time"]:
        dtype = df.schema["open_time"]
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            raise ValueError("open_time must be timezone-aware (UTC); naive timestamps are rejected")
    return df.select([pl.col(c).cast(t) for c, t in BAR_SCHEMA.items()])
