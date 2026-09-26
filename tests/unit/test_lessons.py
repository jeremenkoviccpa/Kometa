"""L6 failure lessons (spec 14.7): "on every demotion, retirement ... and failed validation, a lesson". The
diagnostician names the market condition where a strategy does worst, only with enough signals on both
sides and a real gap; otherwise it says so. Every "no lesson"/"no finding" case has a control that has one."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from autotrader.core.events import StageChanged
from autotrader.core.models import Stage, Timeframe
from autotrader.learning.features import FeatureSnapshot
from autotrader.learning.journal import JournalEntry, Outcome
from autotrader.learning.lessons import (
    LessonBook,
    LessonService,
    diagnose,
    lesson_from_stage,
    lesson_from_validation,
)

T0 = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)


def feats(slope_d1: float) -> FeatureSnapshot:
    return FeatureSnapshot(
        atr_h1=10.0,
        atr_pct_h1=0.5,
        slope_h1=None,
        slope_h4=None,
        slope_d1=slope_d1,
        level_dist_atr=None,
        realized_vol_d=None,
        spread_ratio=None,
        minutes_to_event=None,
        session=("london",),
        hour=10,
        weekday=1,
    )


def done(i: int, r: float, slope_d1: float, sid: str = "swing") -> JournalEntry:
    return JournalEntry(
        signal_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sid}{i}")),
        strategy_id=sid,
        strategy_version="1.0.0",
        symbol="XAUUSD",
        side="buy",
        timeframe=Timeframe.H4,
        created_at=T0 + timedelta(hours=i),
        entry=2000.0,
        stop=1990.0,
        target=2020.0,
        shadow=False,
        strategy_win_rate_20=None,
        features=feats(slope_d1),
        outcome=Outcome(
            label=1 if r > 0 else -1,
            r=r,
            mae_r=0.5,
            mfe_r=1.0,
            minutes=60,
            resolved_at=T0 + timedelta(hours=i, minutes=60),
        ),
    )


def against_the_trend_loses() -> list[JournalEntry]:
    with_trend = [done(i, 2.0 if i % 2 else -1.0, 0.3) for i in range(30)]  # +0.5R a signal
    against = [done(100 + i, 2.0 if i % 5 == 0 else -1.0, -0.3) for i in range(20)]  # -0.4R
    return with_trend + against


# ---------------------------------------------------------------- the diagnostician


def test_the_worst_condition_is_named_with_its_counts_as_a_hypothesis() -> None:
    d = diagnose(against_the_trend_loses())
    assert d.worst is not None and d.worst.condition == "against the D1 trend"
    assert (
        d.worst.n == 20 and d.worst.avg_r == pytest.approx(-0.4) and d.worst.rest_avg_r == pytest.approx(0.5)
    )
    assert "hypothesis" in d.finding and "20 signals" in d.finding


def test_no_finding_when_nothing_stands_out() -> None:
    even = [done(i, 2.0 if i % 2 else -1.0, 0.3 if i % 3 else -0.3) for i in range(60)]
    d = diagnose(even)
    assert d.worst is None and "no single market condition" in d.finding
    assert diagnose(against_the_trend_loses()).worst is not None  # control


def test_no_finding_from_too_few_signals() -> None:
    d = diagnose(against_the_trend_loses()[:5] + against_the_trend_loses()[-5:])
    assert d.worst is None and "too few" in d.finding


# ---------------------------------------------------------------- lessons on events


def stage(frm: Stage, to: Stage, reason: str = "drawdown 16.0% > 15%") -> StageChanged:
    return StageChanged(
        at=T0, strategy_id="swing", strategy_version="1.0.0", from_stage=frm, to_stage=to, reason=reason
    )


@pytest.mark.parametrize(
    ("frm", "to", "event"),
    [
        (Stage.LIVE, Stage.MICRO, "demotion"),
        (Stage.MICRO, Stage.SHADOW, "demotion"),
        (Stage.DEMO_ONLY, Stage.SHADOW, "demotion"),  # by the system, not the owner's switch
        (Stage.SHADOW, Stage.RETIRED, "retirement"),
        (Stage.MICRO, Stage.LIVE, None),  # promotion: nothing failed
        (Stage.CANDIDATE, Stage.SHADOW, None),
    ],
)
def test_a_lesson_on_every_demotion_and_retirement_and_none_on_promotion(
    frm: Stage, to: Stage, event: str | None
) -> None:
    x = lesson_from_stage(
        stage(frm, to), "swing_trend_pullback", against_the_trend_loses(), {"trend_len": 50}
    )
    assert (x.event if x else None) == event
    if x is not None:
        assert "drawdown 16.0%" in x.what_failed and "against the D1 trend" in x.suspected_cause
        assert '"trend_len": 50' in x.what_tried and x.evidence["journal_signals"] == 50


def test_the_owner_switching_paper_trading_off_is_not_a_failure() -> None:
    off = stage(Stage.DEMO_ONLY, Stage.SHADOW, "owner: paper trading stopped")
    assert lesson_from_stage(off, "f", []) is None
    assert lesson_from_stage(stage(Stage.DEMO_ONLY, Stage.SHADOW), "f", []) is not None  # control


def report(passed: bool) -> dict[str, Any]:
    checks = [
        {"name": "oos_trades", "value": 38.0, "op": ">=", "threshold": 200, "passed": passed},
        {"name": "profit_factor", "value": 1.6, "op": ">=", "threshold": 1.3, "passed": True},
    ]
    return {
        "strategy_id": "smc_sniper_active",
        "version": "1.0.0",
        "family": "smc_top_down",
        "code_hash": "abc",
        "final_params": {"disp_atr": 1.5},
        "checks": checks,
        "oos_trades": [{}] * 38,
    }


def test_a_failed_validation_leaves_a_lesson_naming_the_metric() -> None:
    x = lesson_from_validation(report(passed=False), [], T0)
    assert x is not None and x.event == "failed_validation" and x.metric == "oos_trades"
    assert x.value == 38.0 and x.threshold == ">= 200" and "oos_trades 38" in x.what_failed
    assert lesson_from_validation(report(passed=True), [], T0) is None  # control: passed, nothing to learn


# ---------------------------------------------------------------- storage and the service


def test_the_book_keeps_one_lesson_per_event_and_survives_a_restart(tmp_path: Path) -> None:
    book = LessonBook(tmp_path / "lessons.jsonl")
    x = lesson_from_stage(stage(Stage.LIVE, Stage.MICRO), "swing_trend_pullback", [])
    assert x is not None and book.add(x) and not book.add(x)
    y = lesson_from_validation(report(passed=False), [], T0 + timedelta(days=1))
    assert y is not None and book.add(y)
    again = LessonBook(tmp_path / "lessons.jsonl")
    assert [z.lesson_id for z in again.top()] == [y.lesson_id, x.lesson_id]  # newest first
    assert [z.lesson_id for z in again.top("smc_top_down")] == [y.lesson_id]


async def test_the_service_writes_lessons_from_the_bus() -> None:
    book = LessonBook(None)
    svc = LessonService(book, against_the_trend_loses, {"swing": "swing_trend_pullback"})
    await svc._on_stage(stage(Stage.MICRO, Stage.LIVE))
    assert not book.lessons
    await svc._on_stage(stage(Stage.LIVE, Stage.MICRO))
    [x] = book.lessons.values()
    assert x.family == "swing_trend_pullback" and x.event == "demotion"
