"""Broker history to the file layout validation reads (`<out>/<SYMBOL>_M1.parquet`), month by month.

Months are cached under `<out>/raw/<SYMBOL>/<YYYY-MM>.parquet`, so a second run only fetches what is new
(the current month is always refreshed). Works with any BrokerAdapter's `history_bars`; used for OANDA.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from autotrader.core.models import Bar, Timeframe
from autotrader.execution.adapter import BrokerAdapter

COLUMNS = ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "volume")


def bars_frame(bars: list[Bar]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": pl.Series([b.open_time for b in bars], dtype=pl.Datetime("us", "UTC")),
            **{c: [float(getattr(b, c)) for b in bars] for c in COLUMNS},
        }
    )


def _months(start: date, end: date) -> list[tuple[datetime, datetime]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        a = datetime(y, m, 1, tzinfo=UTC)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((a, datetime(y, m, 1, tzinfo=UTC)))
    return out


async def fetch_history(
    adapter: BrokerAdapter,
    symbol: str,
    start: date,
    end: date,
    out: Path,
    progress: Callable[[str], None] = lambda _m: None,
) -> int:
    raw = out / "raw" / symbol
    raw.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    for a, b in _months(start, end):
        p = raw / f"{a:%Y-%m}.parquet"
        if p.exists() and b <= now:
            continue  # a finished month never changes
        bars = await adapter.history_bars(symbol, Timeframe.M1, a, min(b, now))
        bars_frame(bars).write_parquet(p)
        progress(f"{symbol} {a:%Y-%m}: {len(bars):,} bars")
    frames = [pl.read_parquet(p) for p in sorted(raw.glob("*.parquet"))]
    df = pl.concat([f for f in frames if f.height]).unique("open_time").sort("open_time") if frames else None
    if df is None or df.is_empty():
        return 0
    df.write_parquet(out / f"{symbol}_M1.parquet")
    return df.height
