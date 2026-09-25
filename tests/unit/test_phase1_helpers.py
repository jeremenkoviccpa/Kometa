from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autotrader.core import indicators as ind
from autotrader.core.models import CalendarEvent
from autotrader.core.series import from_ns, to_ns
from autotrader.core.timeutil import utc
from autotrader.data.calendar import CsvCalendar, StaticCalendar, make_calendar
from autotrader.data.convert import median_spread_by_hour, spread_stats, to_bars_array
from autotrader.data.synthetic import SyntheticSpec, generate


def _walk(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return c + np.abs(rng.normal(0, 0.5, n)), c - np.abs(rng.normal(0, 0.5, n)), c


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_levels_batch_matches_incremental(seed: int) -> None:
    h, lo, c = _walk(600, seed)
    batch = ind.levels(h, lo, c)
    tr = ind.LevelTracker()
    for t in range(h.size):
        got = tr.update(float(h[t]), float(lo[t]), float(c[t]))
        assert len(got) == len(batch[t])
        for a, b in zip(got, batch[t], strict=True):
            assert a.touches == b.touches
            assert a.price == pytest.approx(b.price, rel=1e-9, abs=1e-9)
            assert (a.broken, a.flipped, a.kind, a.last_touch) == (b.broken, b.flipped, b.kind, b.last_touch)
    assert any(lv.touches > 1 for lv in batch[-1])


def test_levels_no_lookahead() -> None:
    h, lo, c = _walk(400, 5)
    t = 250
    ph, pl_, pc = h.copy(), lo.copy(), c.copy()
    rng = np.random.default_rng(0)
    for a in (ph, pl_, pc):
        a[t + 1 :] = rng.uniform(0, 1000, a.size - t - 1)
    assert ind.levels(h, lo, c)[: t + 1] == ind.levels(ph, pl_, pc)[: t + 1]


def test_level_break_and_flip() -> None:
    # range with resistance near 110, then break up and retest from above
    up = [100, 104, 110, 104, 100, 104, 110.2, 104, 100, 104, 109.9, 104, 100]
    tail = [106, 112, 116, 118, 116, 113, 111, 110.5, 113, 116, 118]
    c = np.array(up + tail, dtype=float)
    snaps = ind.levels(c + 0.3, c - 0.3, c, left=1, right=1, atr_n=3, k=1.0, break_atr=0.2)
    res = [lv for lv in snaps[-1] if abs(lv.price - 110) < 1.5]
    assert res, "resistance near 110 not found"
    lv = res[0]
    assert lv.touches >= 2
    assert lv.broken and lv.kind == "support"
    assert lv.flipped and lv.flipped_at > lv.broken_at


def test_fit_line_convergence_breakout() -> None:
    upper = ind.fit_line([0, 10, 20], [110, 108, 106])
    lower = ind.fit_line([0, 10, 20], [100, 101, 102])
    assert upper.slope == pytest.approx(-0.2)
    assert upper.r2 == pytest.approx(1.0)
    conv = ind.convergence(upper, lower, 0, 20)
    assert conv.converging
    assert conv.apex_x == pytest.approx(10 / 0.3)
    assert ind.breakout(107.0, upper, 20) == 1
    assert ind.breakout(105.0, upper, 20) == -1
    assert ind.breakout(106.05, upper, 20, min_distance=0.1) == 0
    xs, ys = ind.last_points([np.nan, 1.0, np.nan, 2.0, 3.0], 2)
    assert xs.tolist() == [3.0, 4.0]
    assert ys.tolist() == [2.0, 3.0]
    with pytest.raises(ValueError, match="vertical"):
        ind.fit_line([1, 1], [1, 2])


def test_sessions() -> None:
    # 2026-01-07 is a Wednesday; 13:00 UTC = London 13:00, New York 08:00
    assert ind.sessions_of(utc(2026, 1, 7, 13)) == ("london", "new_york")
    assert ind.sessions_of(utc(2026, 1, 7, 2)) == ("asia",)
    assert ind.sessions_of(utc(2026, 1, 10, 13)) == ()  # Saturday


def test_event_index_blackout() -> None:
    ev = [
        CalendarEvent(time=utc(2026, 1, 7, 13, 30), currency="USD", impact="high", name="CPI"),
        CalendarEvent(time=utc(2026, 1, 7, 9, 0), currency="EUR", impact="low", name="minor"),
    ]
    idx = ind.EventIndex(ev)
    assert idx.minutes_to_next(utc(2026, 1, 7, 13, 0), "USD") == 30.0
    assert idx.minutes_to_next(utc(2026, 1, 7, 14, 0), "USD") is None
    assert idx.minutes_to_next(utc(2026, 1, 7, 8, 0), "EUR") is None  # low impact ignored
    assert idx.in_blackout(utc(2026, 1, 7, 13, 20), ["EUR", "USD"], 15)
    assert idx.in_blackout(utc(2026, 1, 7, 13, 44), ["USD"], 15)
    assert not idx.in_blackout(utc(2026, 1, 7, 13, 46), ["USD"], 15)


def test_calendar_sources(tmp_path: Path) -> None:
    p = tmp_path / "cal.csv"
    p.write_text("time,currency,impact,name\n2026-01-07T13:30:00+00:00,USD,high,CPI\n")
    cal = CsvCalendar(p)
    assert [e.name for e in cal.events(utc(2026, 1, 1), utc(2026, 2, 1))] == ["CPI"]
    assert cal.events(utc(2026, 2, 1), utc(2026, 3, 1)) == []
    assert isinstance(make_calendar("static", events=[]), StaticCalendar)


def test_bars_array_and_spread_stats() -> None:
    df = generate(SyntheticSpec(days=14, seed=1))
    ba = to_bars_array(df)
    assert len(ba) == df.height
    assert from_ns(int(ba.open_time[0])) == df["open_time"][0]
    assert int(ba.close_time[0] - ba.open_time[0]) == to_ns(utc(2020, 1, 1, 0, 1)) - to_ns(utc(2020, 1, 1))
    part = ba.slice(0, 10)
    part.bid_c[0] = -1.0
    assert ba.bid_c[0] != -1.0  # slices are copies
    stats = spread_stats(df)
    assert stats.height > 100
    assert (stats["p90_spread"] >= stats["median_spread"]).all()
    med = median_spread_by_hour(df)
    assert med.shape == (168,)
    assert (med > 0).all()
