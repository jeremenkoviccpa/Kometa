"""DataSource interface and the CSV/Parquet importer. Sources are registered by name (section 21)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from autotrader.core.models import Timeframe
from autotrader.core.timeutil import ensure_utc
from autotrader.data.schema import validate_frame


class DataSource(Protocol):
    name: str

    def m1_bars(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Canonical bar frame with open_time in [start, end)."""
        ...


class FileSource:
    """Reads `<root>/<SYMBOL>_M1.{csv,parquet}` with a column mapping.

    Timestamps in files must carry an explicit timezone or `assume_tz` must be set;
    silently guessing a timezone is a classic source of lookahead bugs.
    """

    name = "file"

    def __init__(
        self, root: Path, columns: dict[str, str] | None = None, assume_tz: str | None = None
    ) -> None:
        self.root = root
        self.columns = columns or {}
        self.assume_tz = assume_tz

    def _path(self, symbol: str) -> Path:
        for ext in ("parquet", "csv"):
            p = self.root / f"{symbol}_{Timeframe.M1.value}.{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no M1 file for {symbol} in {self.root}")

    def m1_bars(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        p = self._path(symbol)
        raw = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, try_parse_dates=True)
        raw = raw.rename({k: v for k, v in self.columns.items() if k in raw.columns})
        ot = raw.schema.get("open_time")
        if ot == pl.String:
            raw = raw.with_columns(pl.col("open_time").str.to_datetime(time_unit="us"))
            ot = raw.schema["open_time"]
        if isinstance(ot, pl.Datetime) and ot.time_zone is None:
            if self.assume_tz is None:
                raise ValueError(f"{p}: timestamps have no timezone and assume_tz is not set")
            raw = raw.with_columns(pl.col("open_time").dt.replace_time_zone(self.assume_tz))
        raw = raw.with_columns(pl.col("open_time").dt.convert_time_zone("UTC"))
        s, e = ensure_utc(start), ensure_utc(end)
        return (
            validate_frame(raw)
            .filter((pl.col("open_time") >= s) & (pl.col("open_time") < e))
            .sort("open_time")
        )


_REGISTRY: dict[str, Callable[..., DataSource]] = {"file": FileSource}


def register_source(name: str, factory: Callable[..., DataSource]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"data source {name!r} already registered")
    _REGISTRY[name] = factory


def make_source(name: str, **kwargs: object) -> DataSource:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"unknown data source {name!r}; known: {sorted(_REGISTRY)}") from None
