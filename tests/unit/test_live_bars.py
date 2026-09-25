"""Live bar builder (spec section 7) against the historical resampler, DST switches included."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from autotrader.core.broker import Quote
from autotrader.core.events import BarClosed
from autotrader.core.models import Bar, Timeframe
from autotrader.core.series import from_ns, to_ns
from autotrader.core.timeframes import bucket_open_ns
from autotrader.core.timeutil import utc
from autotrader.data.aggregate import resample
from autotrader.data.synthetic import SyntheticSpec, generate
from autotrader.engine.live_bars import LiveBarBuilder

COLS = ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c")


def quotes_for(df: pl.DataFrame, symbol: str = "SYNTH") -> list[Quote]:
    """Four quotes per M1 bar that reproduce it exactly: open, high, low, close."""
    out = []
    for r in df.iter_rows(named=True):
        t0 = r["open_time"]
        for k, (b, a) in enumerate(
            (("bid_o", "ask_o"), ("bid_h", "ask_h"), ("bid_l", "ask_l"), ("bid_c", "ask_c"))
        ):
            out.append(
                Quote(
                    symbol=symbol,
                    bid=Decimal(str(r[b])),
                    ask=Decimal(str(r[a])),
                    time=t0 + timedelta(seconds=5 + 15 * k),
                )
            )
    return out


def build(df: pl.DataFrame, tfs: list[Timeframe]) -> tuple[list[BarClosed], list[Bar]]:
    m1: list[Bar] = []
    b = LiveBarBuilder([("SYNTH", tf) for tf in tfs], on_m1=m1.append)
    events: list[BarClosed] = []
    for q in quotes_for(df):
        events += b.on_quote(q)
    events += b.on_time(df["open_time"][-1] + timedelta(days=3))
    return events, m1


@pytest.fixture(scope="module", params=[datetime(2020, 3, 5, tzinfo=UTC), datetime(2020, 10, 29, tzinfo=UTC)])
def dst_frame(request: pytest.FixtureRequest) -> pl.DataFrame:
    # a week around the US clock changes (2020-03-08 and 2020-11-01, both Sundays)
    return generate(SyntheticSpec(start=request.param, days=7, seed=5))


@pytest.mark.parametrize("tf", [Timeframe.M5, Timeframe.H1, Timeframe.H4, Timeframe.D1])
def test_bucket_alignment_matches_resampler(dst_frame: pl.DataFrame, tf: Timeframe) -> None:
    hist = resample(dst_frame, tf)
    opens = {to_ns(t) for t in hist["open_time"]}
    mine = {bucket_open_ns(to_ns(t), tf) for t in dst_frame["open_time"]}
    assert opens <= mine and len(mine - opens) <= 1  # the resampler drops only the incomplete last bucket


@pytest.fixture(scope="module")
def dst_built(dst_frame: pl.DataFrame) -> tuple[list[BarClosed], list[Bar]]:
    return build(dst_frame, [Timeframe.M1, Timeframe.H1, Timeframe.H4, Timeframe.D1])


@pytest.mark.parametrize("tf", [Timeframe.H1, Timeframe.H4, Timeframe.D1])
def test_live_bars_equal_historical_bars(
    dst_frame: pl.DataFrame, dst_built: tuple[list[BarClosed], list[Bar]], tf: Timeframe
) -> None:
    events, m1 = dst_built
    # M1 bars rebuilt from quotes are the original bars
    assert len(m1) == dst_frame.height
    first = dst_frame.row(0, named=True)
    assert m1[0].open_time == first["open_time"] and all(getattr(m1[0], c) == first[c] for c in COLS)
    hist = resample(dst_frame.with_columns(pl.lit(4.0).alias("volume")), tf)
    live = [e.bar for e in events if e.timeframe == tf]
    assert [b.open_time for b in live[: hist.height]] == list(hist["open_time"])
    for b, r in zip(live, hist.iter_rows(named=True), strict=False):
        assert b.close_time == r["close_time"]
        assert all(getattr(b, c) == pytest.approx(r[c], abs=1e-12) for c in COLS)
        assert b.volume == r["volume"]


def test_batches_follow_backtest_order() -> None:
    b = LiveBarBuilder([("B", Timeframe.M1), ("B", Timeframe.H1), ("A", Timeframe.M1), ("A", Timeframe.H1)])
    t = utc(2026, 1, 7, 12, 59, 30)
    for s in ("A", "B"):
        b.on_quote(Quote(symbol=s, bid=Decimal(1), ask=Decimal(2), time=t))
    out = b.on_time(utc(2026, 1, 7, 13, 0, 2))
    assert [(e.symbol, e.timeframe) for e in out] == [
        ("B", Timeframe.M1),
        ("B", Timeframe.H1),
        ("A", Timeframe.M1),
        ("A", Timeframe.H1),
    ]
    assert all(e.at == utc(2026, 1, 7, 13) for e in out)


def test_grace_and_late_quotes() -> None:
    b = LiveBarBuilder([("A", Timeframe.M1)])
    b.on_quote(Quote(symbol="A", bid=Decimal(1), ask=Decimal(2), time=utc(2026, 1, 7, 12, 0, 59)))
    assert b.on_time(utc(2026, 1, 7, 12, 1)) == []  # within the 1 s grace
    out = b.on_time(utc(2026, 1, 7, 12, 1, 1))
    assert len(out) == 1 and out[0].bar.close_time == utc(2026, 1, 7, 12, 1)
    b.on_quote(Quote(symbol="A", bid=Decimal(9), ask=Decimal(10), time=utc(2026, 1, 7, 12, 0, 59)))
    assert b.late_quotes == 1  # never rewrites an emitted bar


def test_minute_without_quotes_has_no_bar() -> None:
    b = LiveBarBuilder([("A", Timeframe.M1)])
    out = b.on_quote(Quote(symbol="A", bid=Decimal(1), ask=Decimal(2), time=utc(2026, 1, 7, 12, 0, 10)))
    out += b.on_quote(Quote(symbol="A", bid=Decimal(1), ask=Decimal(2), time=utc(2026, 1, 7, 12, 5, 10)))
    out += b.on_time(utc(2026, 1, 7, 13))
    assert [e.bar.open_time for e in out] == [utc(2026, 1, 7, 12, 0), utc(2026, 1, 7, 12, 5)]
    assert from_ns(to_ns(out[1].at)) == utc(2026, 1, 7, 12, 6)
