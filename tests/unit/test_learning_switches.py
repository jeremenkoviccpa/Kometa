"""Learning safety switches and the weekly report (spec 14.9): "learning.freeze stops all loops without
affecting live trading", "if total equity drawdown exceeds 8 percent, loops L1 to L4 pause until equity
recovers to within 4 percent of peak", "a weekly learning report: variants tried, passed, promoted, swapped,
rolled back, retired, lessons added, API cost". Each pause has an unpaused control."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from autotrader.cli.main import _frozen
from autotrader.core.events import StageChanged
from autotrader.core.models import Stage
from autotrader.learning.guard import LearningGuard
from autotrader.learning.lessons import LessonBook, LessonService
from autotrader.learning.report import markdown, weekly_report

T0 = datetime(2026, 9, 21, 12, tzinfo=UTC)


def test_the_owners_freeze_stops_every_loop(tmp_path: Path) -> None:
    g = LearningGuard(tmp_path / "learning.freeze")
    assert all(g.paused(x) is None for x in ("L1", "L2", "L3", "L4", "L6", "L8"))  # control
    (tmp_path / "learning.freeze").touch()
    assert all("frozen" in (g.paused(x) or "") for x in ("L1", "L2", "L6", "L8"))


def test_a_drawdown_pauses_l1_to_l4_with_hysteresis_and_a_restart_remembers(tmp_path: Path) -> None:
    state = tmp_path / "guard.json"
    g = LearningGuard(tmp_path / "f", state)
    g.update(0.07, T0)
    assert g.paused("L2") is None
    g.update(0.09, T0)
    assert "drawdown" in (g.paused("L2") or "") and g.paused("L6") is None  # lessons keep learning
    assert state.exists()  # persisted as it took effect
    again = LearningGuard(tmp_path / "f", state)
    assert "9.0%" in (again.paused("L3") or "")  # a restart remembers why
    again.update(0.06, T0)
    assert again.paused("L1") is not None  # between 4% and 8%: still paused
    again.update(0.04, T0)
    assert again.paused("L1") is None and LearningGuard(tmp_path / "f", state).paused("L1") is None


def test_an_unreadable_state_pauses_to_be_safe(tmp_path: Path) -> None:
    (tmp_path / "guard.json").write_text("{not json")
    assert "unreadable" in (LearningGuard(tmp_path / "f", tmp_path / "guard.json").paused("L2") or "")


async def test_lessons_stop_while_frozen(tmp_path: Path) -> None:
    g = LearningGuard(tmp_path / "freeze")
    book = LessonBook(None)
    svc = LessonService(book, list, {"s": "f"}, paused=g.paused)
    demote = StageChanged(
        at=T0,
        strategy_id="s",
        strategy_version="1.0.0",
        from_stage=Stage.LIVE,
        to_stage=Stage.MICRO,
        reason="dd",
    )
    (tmp_path / "freeze").touch()
    await svc._on_stage(demote)
    assert not book.lessons
    (tmp_path / "freeze").unlink()
    await svc._on_stage(demote)  # control
    assert len(book.lessons) == 1


def test_the_offline_commands_refuse_while_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AT_LEARNING_FREEZE_PATH", str(tmp_path / "freeze"))
    assert not _frozen()
    (tmp_path / "freeze").touch()
    assert _frozen() and "learning is frozen" in capsys.readouterr().out


class Rec:
    def __init__(self, at: datetime, frm: str, to: str, reason: str = "gate") -> None:
        self.at, self.from_stage, self.to_stage, self.reason = at, Stage(frm), Stage(to), reason


def test_the_report_counts_the_last_week_from_the_records() -> None:
    now = T0 + timedelta(days=7)
    records = [
        Rec(T0 + timedelta(days=1), "shadow", "micro"),  # promoted
        Rec(T0 + timedelta(days=2), "live", "micro"),  # demoted
        Rec(
            T0 + timedelta(days=3), "shadow", "live", "swap: replaces 1.0.0: +0.3R"
        ),  # swapped (not a promotion)
        Rec(T0 + timedelta(days=3), "live", "shadow", "swap: replaced by 1.0.1; kept in shadow"),
        Rec(T0 + timedelta(days=4), "shadow", "live", "rollback: restored over 1.0.1"),  # rolled back
        Rec(T0 + timedelta(days=5), "shadow", "retired", "owner"),  # retired
        Rec(T0 - timedelta(days=3), "shadow", "micro"),  # last week's: not counted
    ]
    models = [
        {"created_at": (T0 + timedelta(days=1)).isoformat(), "status": "passed_oos"},
        {"created_at": (T0 + timedelta(days=2)).isoformat(), "status": "refused"},
        {"created_at": (T0 - timedelta(days=9)).isoformat(), "status": "passed_oos"},
    ]
    tests = [{"at": (T0 + timedelta(days=3)).isoformat()}, {"at": (T0 - timedelta(days=1)).isoformat()}]
    r = weekly_report(now, stage_records=records, swap_tests=tests, models=models, api_calls=12)
    assert (r["promoted"], r["demoted"], r["swapped"], r["rolled_back"], r["retired"]) == (1, 1, 1, 1, 1)
    assert (r["swap_tests"], r["models_trained"], r["models_passed"], r["api_calls"]) == (1, 2, 1, 12)
    text = markdown(r)
    assert "Promoted 1, demoted 1, swapped 1 (tests 1), rolled back 1, retired 1." in text
    assert "Learning: running." in text
    frozen = markdown({**r, "learning_paused": {"frozen_by_owner": True}})
    assert "frozen by the owner" in frozen
