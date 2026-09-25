"""BacktestProfile from validation output (the reference the lifecycle evaluator uses)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from autotrader.core.timeutil import utc
from autotrader.engine.simbroker import TradeRecord
from autotrader.validation.profile import profile_from_trades, weekly_counts


def tr(r: float, day: int) -> TradeRecord:
    ns = int(utc(2026, 1, 5 + day).timestamp()) * 1_000_000_000
    return TradeRecord(
        "t", "s", "x", "1", "EURUSD", "buy", 0.1, ns, 1.1, 1.09, ns + 3_600_000_000_000, 1.1, "target",
        0.0, 0.0, 0.0, 0.0, 0.0, r * 100, 100.0, r, 0.0, 0.0, 1, (),
    )  # fmt: skip


def test_weekly_counts_include_empty_weeks() -> None:
    start = utc(2026, 1, 5)
    times = [start + timedelta(days=d) for d in (0, 1, 15, 16, 17)]
    assert weekly_counts(times, start, start + timedelta(weeks=4)) == (2, 0, 3, 0)


def test_profile_converts_drawdown_to_r() -> None:
    trades = [tr(1.0, 0), tr(-1.0, 1), tr(2.0, 8)]
    p = profile_from_trades(
        "x",
        "1",
        trades,
        (utc(2026, 1, 5), utc(2026, 1, 19)),
        mc_dd_p95=0.04,
        risk_fraction=0.005,
        model_slippage={"EURUSD": 0.00002},
        source="t",
        synthetic=True,
    )
    assert p.mc_dd_p95_r == pytest.approx(8.0)
    assert p.weekly_entries == (2, 1)
    assert p.avg_r == pytest.approx(2 / 3) and p.win_rate == pytest.approx(2 / 3)
