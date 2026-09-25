"""data_version: a content hash that pins exactly which bars a result was computed on."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from autotrader.core.hashing import hash_obj
from autotrader.data.schema import validate_frame

SYNTHETIC_PREFIX = "synthetic-"


def data_version(df: pl.DataFrame, symbol: str, timeframe: str, *, synthetic: bool = False) -> str:
    """Hash over (symbol, timeframe, first/last time, rows, content hash per UTC month)."""
    df = validate_frame(df).sort("open_time")
    months: dict[str, str] = {}
    if df.height:
        keyed = df.with_columns(pl.col("open_time").dt.strftime("%Y-%m").alias("_m"))
        for (month,), part in keyed.group_by("_m", maintain_order=True):
            h = hashlib.sha256()
            for col in part.drop("_m").columns:
                arr = part[col].to_physical().to_numpy()
                h.update(col.encode())
                h.update(np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<")).tobytes())
            months[str(month)] = h.hexdigest()
    header = {
        "symbol": symbol,
        "timeframe": timeframe,
        "first": df["open_time"][0].isoformat() if df.height else None,
        "last": df["open_time"][-1].isoformat() if df.height else None,
        "rows": df.height,
        "months": months,
    }
    digest = hash_obj(header)[:32]
    return f"{SYNTHETIC_PREFIX}{digest}" if synthetic else digest


def is_synthetic(version: str) -> bool:
    return version.startswith(SYNTHETIC_PREFIX)
