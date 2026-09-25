"""Historical M1 bid/ask bars from Dukascopy's public datafeed (open question 2: the owner chose Dukascopy).

Per symbol and UTC day the feed has two LZMA-compressed files, `BID_candles_min_1.bi5` and
`ASK_candles_min_1.bi5`, each 1440 records of big-endian (seconds into the day, open, close, low, high as
int32 in 1/scale units, volume as float32). Months in the URL are 0-based. Minutes when the market is
closed come as flat zero-volume candles and are dropped, so a closed market has no bars, like everywhere
else in the system.

Downloads are polite (a few at a time, backoff on 429/503) and resumable: every day's raw files are kept
under `raw/`, so a second run only fetches what is missing. The result is `<out>/<SYMBOL>_M1.parquet` in
the FileSource layout, with a real (non-synthetic) data_version.
SPEC-QUESTION: Dukascopy's terms allow personal use of the feed; the owner confirms before commercial use
(docs/open_questions.md 2).
"""

from __future__ import annotations

import asyncio
import lzma
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl

BASE = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">5if")
# price scale by symbol: metals and JPY crosses are quoted in thousandths, the rest in 1e-5
SCALE = {"XAUUSD": 1000, "XAGUSD": 1000, "USDJPY": 1000, "EURJPY": 1000, "GBPJPY": 1000}
DEFAULT_SCALE = 100_000


def scale_of(symbol: str) -> int:
    return SCALE.get(symbol, DEFAULT_SCALE)


def day_url(symbol: str, day: date, side: str) -> str:
    return f"{BASE}/{symbol}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def decode_day(raw: bytes, day: date, scale: int) -> pl.DataFrame:
    """One side's file -> (open_time, o, h, l, c, volume). An empty file means no data for the day."""
    if not raw:
        return pl.DataFrame(
            schema={"open_time": pl.Datetime("us", "UTC"), **dict.fromkeys("ohlcv", pl.Float64)}
        )
    data = lzma.decompress(raw)
    n = len(data) // RECORD.size
    arr = np.frombuffer(data[: n * RECORD.size], dtype=np.dtype(">i4, >i4, >i4, >i4, >i4, >f4"))
    secs = arr["f0"].astype(np.int64)
    start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 1_000_000
    return pl.DataFrame(
        {
            "open_time": pl.Series(start + secs * 1_000_000).cast(pl.Datetime("us", "UTC")),
            "o": arr["f1"] / scale,
            "c": arr["f2"] / scale,
            "l": arr["f3"] / scale,
            "h": arr["f4"] / scale,
            "v": arr["f5"].astype(np.float64),
        }
    )


def merge_sides(bid: pl.DataFrame, ask: pl.DataFrame) -> pl.DataFrame:
    """Bid and ask candles -> the system's M1 bar frame. Drops closed-market minutes (flat, no volume)."""
    b = bid.rename({"o": "bid_o", "h": "bid_h", "l": "bid_l", "c": "bid_c", "v": "volume"})
    a = ask.rename({"o": "ask_o", "h": "ask_h", "l": "ask_l", "c": "ask_c", "v": "ask_volume"})
    df = b.join(a, on="open_time", how="inner")
    traded = (pl.col("volume") > 0) | (pl.col("ask_volume") > 0) | (pl.col("bid_h") != pl.col("bid_l"))
    return (
        df.filter(traded)
        .select(
            "open_time",
            "bid_o",
            "bid_h",
            "bid_l",
            "bid_c",
            "ask_o",
            "ask_h",
            "ask_l",
            "ask_c",
            (pl.col("volume") + pl.col("ask_volume")).alias("volume"),
        )
        .sort("open_time")
    )


@dataclass
class FetchReport:
    days: int = 0
    downloaded: int = 0
    cached: int = 0
    failed: list[str] = field(default_factory=list)
    bars: int = 0


class DukascopyFetcher:
    def __init__(
        self,
        out: Path,
        *,
        concurrency: int = 4,
        retries: int = 6,
        client: httpx.AsyncClient | None = None,
        progress: Callable[[str], None] | None = None,
        backoff: float = 1.0,
        pause: float = 0.5,
    ) -> None:
        self.backoff = backoff
        self.pause = pause  # between requests per worker: the feed throttles bursts by IP
        self.out = out
        self.concurrency = concurrency
        self.retries = retries
        self.client = client
        self.progress = progress or (lambda _msg: None)

    def raw_path(self, symbol: str, day: date, side: str) -> Path:
        return self.out / "raw" / symbol / f"{day:%Y}" / f"{day:%m%d}_{side}.bi5"

    async def _get(self, client: httpx.AsyncClient, url: str) -> bytes | None:
        """The file's bytes; b"" when the feed has no file for that day (404); None after all retries."""
        delay = self.backoff
        for _ in range(self.retries):
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return r.content
                if r.status_code == 404:
                    return b""
            except httpx.HTTPError:
                pass
            await asyncio.sleep(delay)  # 429/503/network: back off, the feed throttles bursts
            delay = min(delay * 2, 30.0)
        return None

    async def fetch(self, symbol: str, start: date, end: date) -> FetchReport:
        """Download (or reuse) every weekday in [start, end] and rebuild <SYMBOL>_M1.parquet."""
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        days = [d for d in days if d.weekday() != 5]  # Saturday never trades; Sunday evening does
        rep = FetchReport(days=len(days))
        sem = asyncio.Semaphore(self.concurrency)
        client = self.client or httpx.AsyncClient(
            timeout=30, headers={"User-Agent": "kometa-autotrader/0.1 (historical data)"}
        )

        async def one(day: date, side: str) -> None:
            p = self.raw_path(symbol, day, side)
            if p.exists():
                rep.cached += 1
                return
            async with sem:
                data = await self._get(client, day_url(symbol, day, side))
                await asyncio.sleep(self.pause)
            if data is None:
                rep.failed.append(f"{day} {side}")
                return
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
            rep.downloaded += 1

        try:
            for i in range(0, len(days), 60):
                chunk = days[i : i + 60]
                await asyncio.gather(*(one(d, s) for d in chunk for s in ("BID", "ASK")))
                self.progress(f"{symbol}: {min(i + 60, len(days))}/{len(days)} days")
        finally:
            if self.client is None:
                await client.aclose()
        rep.bars = self.build(symbol, days)
        return rep

    def build(self, symbol: str, days: list[date]) -> int:
        scale = scale_of(symbol)
        frames = []
        for d in days:
            pb, pa = self.raw_path(symbol, d, "BID"), self.raw_path(symbol, d, "ASK")
            if pb.exists() and pa.exists():
                frames.append(
                    merge_sides(decode_day(pb.read_bytes(), d, scale), decode_day(pa.read_bytes(), d, scale))
                )
        if not frames:
            return 0
        df = pl.concat(frames).unique("open_time", keep="first").sort("open_time")
        self.out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.out / f"{symbol}_M1.parquet")
        return df.height
