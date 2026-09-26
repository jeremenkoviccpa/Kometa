"""The weekly learning report (spec 14.9): what the learning loops did in the last seven days, counted from
the records themselves (the registry's stage changes and swap tests, lessons, meta models, trials, the
journal), so it cannot disagree with them. Sent through the alert router (Telegram when configured) and shown
in the hub. API cost is not billed data here: the report gives the number of calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from autotrader.learning.journal import JournalEntry
from autotrader.learning.lessons import Lesson

WEEK = timedelta(days=7)
LADDER = {"candidate": 0, "shadow": 1, "demo_only": 2, "micro": 2, "live": 3, "scaled": 4, "retired": -1}


def _at(x: Any) -> datetime:
    return x if isinstance(x, datetime) else datetime.fromisoformat(str(x))


def weekly_report(
    now: datetime,
    *,
    stage_records: Iterable[Any],  # lifecycle StageRecord (at, from_stage, to_stage, reason, ...)
    swap_tests: Sequence[Mapping[str, Any]] = (),
    lessons: Iterable[Lesson] = (),
    models: Sequence[Mapping[str, Any]] = (),
    trials: Sequence[Any] = (),  # validation Trial (window_end is not when it ran: counted in full)
    journal: Iterable[JournalEntry] = (),
    api_calls: int = 0,
    guard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    since = now - WEEK
    recent = [r for r in stage_records if _at(r.at) >= since and r.from_stage is not None]

    def moved(r: Any, up: bool) -> bool:
        a, b = LADDER.get(str(r.from_stage.value), 0), LADDER.get(str(r.to_stage.value), 0)
        return (b > a) if up else (0 <= b < a)

    swaps = [r for r in recent if r.reason.startswith("swap: replaces")]
    signals = [e for e in journal if e.created_at >= since]
    return {
        "from": since.isoformat(),
        "to": now.isoformat(),
        "variants_tried": len(trials),
        "variants_passed": sum(1 for t in trials if getattr(t, "passed", None)),
        "promoted": sum(
            1 for r in recent if moved(r, True) and not r.reason.startswith(("swap", "rollback"))
        ),
        "demoted": sum(
            1 for r in recent if moved(r, False) and not r.reason.startswith(("swap", "rollback"))
        ),
        "swapped": len(swaps),
        "rolled_back": sum(1 for r in recent if r.reason.startswith("rollback:") and moved(r, True)),
        "retired": sum(1 for r in recent if str(r.to_stage.value) == "retired"),
        "swap_tests": sum(1 for t in swap_tests if _at(t["at"]) >= since),
        "lessons_added": sum(1 for x in lessons if x.at >= since),
        "models_trained": sum(1 for m in models if _at(m["created_at"]) >= since),
        "models_passed": sum(
            1 for m in models if _at(m["created_at"]) >= since and m.get("status") == "passed_oos"
        ),
        "signals_journaled": len(signals),
        "outcomes_known": sum(1 for e in signals if e.outcome is not None),
        "api_calls": api_calls,
        "learning_paused": guard or {},
    }


def markdown(r: Mapping[str, Any]) -> str:
    g = r.get("learning_paused") or {}
    state = (
        "frozen by the owner"
        if g.get("frozen_by_owner")
        else f"paused in a drawdown ({g.get('drawdown_reason')})"
        if g.get("drawdown_paused_since")
        else "running"
    )
    lines = [
        f"Kometa weekly learning report, {r['from'][:10]} to {r['to'][:10]}",
        f"Learning: {state}.",
        f"Variants tried {r['variants_tried']}, passed {r['variants_passed']} "
        "(all recorded trials: a trial carries no run time).",
        f"Promoted {r['promoted']}, demoted {r['demoted']}, swapped {r['swapped']} "
        f"(tests {r['swap_tests']}), rolled back {r['rolled_back']}, retired {r['retired']}.",
        f"Lessons added {r['lessons_added']}. "
        f"Meta models trained {r['models_trained']}, passed {r['models_passed']}.",
        f"Signals journaled {r['signals_journaled']} ({r['outcomes_known']} with an outcome). "
        f"API calls {r['api_calls']}.",
    ]
    return "\n".join(lines)
