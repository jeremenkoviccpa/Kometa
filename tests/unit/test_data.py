from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from autotrader.core.models import Timeframe
from autotrader.core.timeutil import utc
from autotrader.data.aggregate import resample
from autotrader.data.calendar import ForexFactoryCalendar
from autotrader.data.instruments import load_instruments
from autotrader.data.market_hours import FX_HOURS
from autotrader.data.quality import IssueType, QualityIssue, Severity, check_bars, quality_alerts
from autotrader.data.sources import FileSource, make_source
from autotrader.data.synthetic import SyntheticSpec, generate, inject_faults
from autotrader.data.versioning import data_version, is_synthetic

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return generate(SyntheticSpec(days=21, seed=3))


def test_synthetic_is_deterministic_and_sane(bars: pl.DataFrame) -> None:
    again = generate(SyntheticSpec(days=21, seed=3))
    assert bars.equals(again)
    assert not bars.equals(generate(SyntheticSpec(days=21, seed=4)))
    assert (bars["ask_c"] > bars["bid_c"]).all()
    assert (bars["bid_h"] >= bars[["bid_o", "bid_c"]].max_horizontal()).all()
    assert (bars["bid_l"] <= bars[["bid_o", "bid_c"]].min_horizontal()).all()
    assert bars["open_time"].is_sorted() and bars["open_time"].n_unique() == bars.height


def test_synthetic_respects_weekend(bars: pl.DataFrame) -> None:
    ny = bars["open_time"].dt.convert_time_zone("America/New_York")
    wd, hr = ny.dt.weekday(), ny.dt.hour()
    assert not (wd == 6).any()  # Saturday
    assert not ((wd == 7) & (hr < 17)).any()  # Sunday before 17:00
    assert not ((wd == 5) & (hr >= 17)).any()  # Friday after 17:00


def test_quality_clean_data_has_no_high_issues(bars: pl.DataFrame) -> None:
    issues = check_bars(bars, "SYNTH", FX_HOURS)
    assert [i for i in issues if i.severity == Severity.HIGH] == []


def test_quality_detects_every_injected_fault(bars: pl.DataFrame) -> None:
    faulty, truth = inject_faults(bars)
    issues = check_bars(faulty, "SYNTH", FX_HOURS)
    found = {(i.issue_type, i.start) for i in issues}
    for f in truth:
        assert (IssueType(f.kind), f.start) in found, f"missed {f.kind} at {f.start}"
    assert {f.kind for f in truth} == {t.value for t in IssueType}


def test_quality_issue_on_a_traded_symbol_is_a_warning(bars: pl.DataFrame) -> None:
    faulty, _ = inject_faults(bars)
    issues = check_bars(faulty, "SYNTH", FX_HOURS)
    now = utc(2026, 1, 7)
    [a] = quality_alerts(issues, {"SYNTH"}, now)
    assert a.kind == "data_quality" and a.severity == "warning" and a.details["symbol"] == "SYNTH"
    assert quality_alerts(issues, {"EURUSD"}, now) == []  # not traded: report only
    low = [QualityIssue("SYNTH", now, now, IssueType.GAP, Severity.LOW)]
    assert quality_alerts(low, {"SYNTH"}, now) == []


def test_data_version_stable_and_sensitive(bars: pl.DataFrame) -> None:
    v1 = data_version(bars, "SYNTH", "M1", synthetic=True)
    assert v1 == data_version(bars.sample(fraction=1.0, shuffle=True, seed=1), "SYNTH", "M1", synthetic=True)
    assert is_synthetic(v1)
    tweaked = bars.with_columns(
        pl.when(pl.int_range(pl.len()) == 500)
        .then(pl.col("bid_c") + 0.01)
        .otherwise(pl.col("bid_c"))
        .alias("bid_c")
    )
    assert data_version(tweaked, "SYNTH", "M1", synthetic=True) != v1
    assert not is_synthetic(data_version(bars, "SYNTH", "M1"))


def test_resample_h1_matches_manual(bars: pl.DataFrame) -> None:
    h1 = resample(bars, Timeframe.H1)
    first = h1.row(3, named=True)
    m1 = bars.filter(
        (pl.col("open_time") >= first["open_time"]) & (pl.col("open_time") < first["close_time"])
    )
    assert m1.height > 0
    assert first["bid_o"] == m1["bid_o"][0]
    assert first["bid_c"] == m1["bid_c"][-1]
    assert first["bid_h"] == m1["bid_h"].max()
    assert first["volume"] == pytest.approx(m1["volume"].sum())


def test_resample_d1_closes_at_17_new_york(bars: pl.DataFrame) -> None:
    d1 = resample(bars, Timeframe.D1)
    ny_close = d1["close_time"].dt.convert_time_zone("America/New_York")
    assert (ny_close.dt.hour() == 17).all() and (ny_close.dt.minute() == 0).all()
    # no bar is visible before its period ends
    assert (d1["close_time"] <= bars["open_time"][-1] + timedelta(minutes=1)).all()


def test_resample_drops_incomplete_last_bucket(bars: pl.DataFrame) -> None:
    cut = bars.head(bars.height - 7)  # end mid-hour
    h1 = resample(cut, Timeframe.H1)
    assert h1["close_time"][-1] <= cut["open_time"][-1] + timedelta(minutes=1)


def test_file_source_requires_timezone(tmp_path: Path, bars: pl.DataFrame) -> None:
    naive = bars.head(100).with_columns(pl.col("open_time").dt.replace_time_zone(None))
    naive.write_csv(tmp_path / "SYNTH_M1.csv")
    with pytest.raises(ValueError, match="no timezone"):
        FileSource(tmp_path).m1_bars("SYNTH", utc(2000, 1, 1), utc(2100, 1, 1))
    got = make_source("file", root=tmp_path, assume_tz="UTC").m1_bars(
        "SYNTH", utc(2000, 1, 1), utc(2100, 1, 1)
    )
    assert got.height == 100
    np.testing.assert_allclose(got["bid_c"].to_numpy(), bars.head(100)["bid_c"].to_numpy())


def test_instruments_config_loads() -> None:
    inst, h = load_instruments(ROOT / "config" / "instruments.yaml")
    assert set(inst) >= {"EURUSD", "XAUUSD"}
    assert str(inst["XAUUSD"].contract_size) == "100"
    assert str(inst["EURUSD"].pip_size) == "0.0001"
    assert len(h) == 64


FF_SAMPLE = json.dumps(  # the shape of ForexFactory's weekly export
    [
        {"title": "Bank Holiday", "country": "JPY", "date": "2026-09-20T19:00:00-04:00", "impact": "Holiday"},
        {
            "title": "Non-Farm Employment Change",
            "country": "USD",
            "date": "2026-09-25T08:30:00-04:00",
            "impact": "High",
            "forecast": "150K",
            "previous": "142K",
        },
        {"title": "German ifo", "country": "EUR", "date": "2026-09-24T04:00:00-04:00", "impact": "Medium"},
        {
            "title": "All-country thing",
            "country": "All",
            "date": "2026-09-24T04:00:00-04:00",
            "impact": "High",
        },
    ]
).encode()


def test_forexfactory_export_parses_caches_and_rate_limits(tmp_path: Path) -> None:
    calls: list[int] = []

    def fetch() -> bytes:
        calls.append(1)
        return FF_SAMPLE

    now = utc(2026, 9, 24, 12)
    cal = ForexFactoryCalendar(tmp_path / "ff.json", fetch=fetch)
    assert not cal.fresh(now) and cal.events(now - timedelta(days=7), now + timedelta(days=7)) == []
    assert cal.refresh(now) and calls == [1]
    evs = cal.events(now - timedelta(days=7), now + timedelta(days=7))
    assert [(e.currency, e.impact) for e in evs] == [("EUR", "medium"), ("USD", "high")]  # holiday, "All" out
    nfp = evs[1]
    assert nfp.time == utc(2026, 9, 25, 12, 30)  # the feed's -04:00 offset, converted to UTC
    assert [r["impact"] for r in cal.rows] == ["holiday", "medium", "high"]  # rows keep holidays for display
    assert not cal.refresh(now + timedelta(minutes=5)) and calls == [1]  # rate limited
    again = ForexFactoryCalendar(tmp_path / "ff.json", fetch=fetch)  # restart: the cache is enough
    assert again.fresh(now + timedelta(hours=1)) and len(again.events(now, now + timedelta(days=2))) == 1
    assert not again.fresh(now + timedelta(hours=13))  # too old to trade on


def test_forexfactory_failed_fetch_keeps_the_previous_calendar(tmp_path: Path) -> None:
    good = ForexFactoryCalendar(tmp_path / "ff.json", fetch=lambda: FF_SAMPLE)
    assert good.refresh(utc(2026, 9, 24))

    def down() -> bytes:
        raise OSError("network down")

    cal = ForexFactoryCalendar(tmp_path / "ff.json", fetch=down)
    assert not cal.refresh(utc(2026, 9, 24, 3))
    assert cal.fetched_at == utc(2026, 9, 24) and len(cal.rows) == 3
    garbage = ForexFactoryCalendar(tmp_path / "ff.json", fetch=lambda: b"<html>rate limited</html>")
    assert not garbage.refresh(utc(2026, 9, 24, 3)) and len(garbage.rows) == 3


def _bi5(rows: list[tuple[int, int, int, int, int, float]]) -> bytes:
    import lzma  # noqa: PLC0415
    import struct  # noqa: PLC0415

    return lzma.compress(b"".join(struct.pack(">5if", *r) for r in rows), format=lzma.FORMAT_ALONE)


def test_dukascopy_day_decodes_merges_and_drops_closed_minutes() -> None:
    from datetime import date  # noqa: PLC0415

    from autotrader.data.dukascopy import decode_day, merge_sides  # noqa: PLC0415

    day = date(2024, 1, 10)
    bid = _bi5([(0, 2030674, 2030415, 2030335, 2030674, 0.03), (60, 2030415, 2030415, 2030415, 2030415, 0.0)])
    ask = _bi5([(0, 2031025, 2030805, 2030715, 2031025, 0.01), (60, 2030765, 2030765, 2030765, 2030765, 0.0)])
    df = merge_sides(decode_day(bid, day, 1000), decode_day(ask, day, 1000))
    assert df.height == 1  # the flat zero-volume minute is a closed market
    r = df.row(0, named=True)
    assert r["open_time"] == utc(2024, 1, 10)
    assert (r["bid_o"], r["bid_h"], r["bid_l"], r["bid_c"]) == (2030.674, 2030.674, 2030.335, 2030.415)
    assert r["ask_o"] == 2031.025 and r["ask_c"] > r["bid_c"]


async def test_dukascopy_fetch_retries_resumes_and_writes_a_file_source(tmp_path: Path) -> None:
    from datetime import date  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    from autotrader.data.dukascopy import DukascopyFetcher  # noqa: PLC0415

    calls: list[str] = []
    first_503: set[str] = set()

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        calls.append(url)
        if "/2024/00/11/" in url:  # a day the feed does not have
            return httpx.Response(404)
        if url not in first_503:  # every file is throttled once, then served
            first_503.add(url)
            return httpx.Response(503)
        px = 2030000 if "BID" in url else 2030300
        return httpx.Response(200, content=_bi5([(0, px, px + 100, px - 100, px + 200, 1.0)]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    f = DukascopyFetcher(tmp_path, client=client, retries=3, backoff=0.001, pause=0)
    rep = await f.fetch("XAUUSD", date(2024, 1, 10), date(2024, 1, 11))
    assert rep.failed == [] and rep.downloaded == 4 and rep.bars == 1
    n = len(calls)
    again = await DukascopyFetcher(tmp_path, client=client, backoff=0.001, pause=0).fetch(
        "XAUUSD", date(2024, 1, 10), date(2024, 1, 11)
    )
    assert len(calls) == n and again.cached == 4  # resumed from the raw cache: no new requests
    bars = FileSource(tmp_path).m1_bars("XAUUSD", utc(2024, 1, 1), utc(2024, 2, 1))
    assert bars.height == 1 and bars["bid_h"][0] == 2030.2 and bars["ask_o"][0] == 2030.3
