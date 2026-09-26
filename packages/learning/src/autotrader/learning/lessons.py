"""L6 failure learning (spec 14.7): a structured lesson on every demotion, retirement and failed validation.

The Diagnostician here is rules, not an LLM: it splits a strategy's journaled outcomes by market condition
(session, with or against the D1/H4 trend, volatility, distance to a level, time of day, spread, news) and
names the condition where results are worst against the rest. Several conditions are compared, so the
finding is written as a hypothesis with its counts, never as a fact; lessons are read by discovery (L1),
they never change a strategy by themselves.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from autotrader.core.bus import STAGES, Handler
from autotrader.core.events import StageChanged
from autotrader.core.models import Stage
from autotrader.learning.journal import JournalEntry
from autotrader.lifecycle.registry import PAPER_OFF, PAPER_ON

LESSON_NS = uuid.UUID("6f1d2c1e-4c7b-5e0a-9a51-4c2f5d7a8b10")
LADDER = {
    Stage.CANDIDATE: 0,
    Stage.SHADOW: 1,
    Stage.DEMO_ONLY: 2,
    Stage.MICRO: 2,
    Stage.LIVE: 3,
    Stage.SCALED: 4,
}
MIN_N = 10  # signals on each side of a split before it can be named
MIN_GAP = 0.3  # R per signal between a condition and the rest
LessonEvent = Literal["demotion", "retirement", "failed_validation", "critic_veto"]


class Split(BaseModel):
    model_config = ConfigDict(frozen=True)

    condition: str  # e.g. "against the D1 trend"
    n: int
    win_rate: float
    avg_r: float
    rest_n: int
    rest_avg_r: float


class Diagnosis(BaseModel):
    model_config = ConfigDict(frozen=True)

    signals: int  # with a known outcome
    avg_r: float | None
    worst: Split | None
    finding: str
    splits: list[Split]


class Lesson(BaseModel):
    model_config = ConfigDict(frozen=True)

    lesson_id: str
    at: datetime
    event: LessonEvent
    strategy_id: str
    strategy_version: str
    family: str
    what_tried: str
    what_failed: str
    metric: str | None = None
    value: float | None = None
    threshold: str | None = None
    suspected_cause: str
    evidence: dict[str, Any]
    author: Literal["rules"] = "rules"


# ---------------------------------------------------------------- the diagnostician


def _conditions(e: JournalEntry) -> list[str]:
    f = e.features
    out = [f"in the {s} session" for s in f.session] or ["outside the main sessions"]
    sign = 1.0 if e.side == "buy" else -1.0
    for tf, v in (("D1", f.slope_d1), ("H4", f.slope_h4)):
        if v is not None and v != 0:
            out.append(f"{'with' if v * sign > 0 else 'against'} the {tf} trend")
    if f.atr_pct_h1 is not None:
        out.append(
            "in low volatility (bottom third)"
            if f.atr_pct_h1 < 1 / 3
            else "in high volatility (top third)"
            if f.atr_pct_h1 > 2 / 3
            else "in normal volatility"
        )
    if f.level_dist_atr is not None:
        out.append("within 1 ATR of an H4 level" if f.level_dist_atr < 1 else "away from H4 levels")
    out.append(
        "00-07 UTC"
        if f.hour < 7
        else "07-12 UTC"
        if f.hour < 12
        else "12-17 UTC"
        if f.hour < 17
        else "17-24 UTC"
    )
    if f.spread_ratio is not None:
        out.append("with the spread over 1.5x its median" if f.spread_ratio > 1.5 else "with a normal spread")
    if f.minutes_to_event is not None:
        out.append("within an hour of high-impact news" if f.minutes_to_event <= 60 else "with no news near")
    return out


def diagnose(entries: Sequence[JournalEntry]) -> Diagnosis:
    done = [e for e in entries if e.outcome is not None]
    rs = [e.outcome.r for e in done if e.outcome is not None]
    if len(done) < 2 * MIN_N:
        return Diagnosis(
            signals=len(done),
            avg_r=sum(rs) / len(rs) if rs else None,
            worst=None,
            finding=f"too few signals with an outcome to point at a cause ({len(done)}; needs {2 * MIN_N})",
            splits=[],
        )
    tagged = [(set(_conditions(e)), e.outcome) for e in done if e.outcome is not None]
    names = sorted({c for cs, _ in tagged for c in cs})
    splits = []
    for c in names:
        inside = [o for cs, o in tagged if c in cs]
        rest = [o for cs, o in tagged if c not in cs]
        if len(inside) < MIN_N or len(rest) < MIN_N:
            continue
        splits.append(
            Split(
                condition=c,
                n=len(inside),
                win_rate=sum(o.label == 1 for o in inside) / len(inside),
                avg_r=sum(o.r for o in inside) / len(inside),
                rest_n=len(rest),
                rest_avg_r=sum(o.r for o in rest) / len(rest),
            )
        )
    splits.sort(key=lambda s: s.avg_r - s.rest_avg_r)
    worst = splits[0] if splits and splits[0].rest_avg_r - splits[0].avg_r >= MIN_GAP else None
    avg = sum(rs) / len(rs)
    if worst is not None:
        finding = (
            f"results are worst {worst.condition}: {worst.avg_r:+.2f}R per signal over {worst.n} signals "
            f"(win rate {worst.win_rate:.0%}) against {worst.rest_avg_r:+.2f}R over the other "
            f"{worst.rest_n}. A hypothesis: {len(splits)} conditions were compared, so one may look bad "
            "by chance alone."
        )
    else:
        finding = (
            f"no single market condition stands out ({len(splits)} compared over {len(done)} signals); "
            "suspect costs, a fading edge or too few signals"
        )
    return Diagnosis(signals=len(done), avg_r=avg, worst=worst, finding=finding, splits=splits)


# ---------------------------------------------------------------- lessons from events


def _id(*parts: str) -> str:
    return str(uuid.uuid5(LESSON_NS, "|".join(parts)))


def lesson_from_stage(
    ev: StageChanged, family: str, entries: Sequence[JournalEntry], params: Mapping[str, Any] | None = None
) -> Lesson | None:
    """A lesson when a version is demoted or retired by the system. The owner switching paper trading off
    is a choice, not a failure: no lesson."""
    if ev.reason in (PAPER_ON, PAPER_OFF):
        return None
    if ev.to_stage == Stage.RETIRED:
        event: LessonEvent = "retirement"
    elif LADDER.get(ev.to_stage, 0) < LADDER.get(ev.from_stage, 0):
        event = "demotion"
    else:
        return None
    mine = [
        e for e in entries if e.strategy_id == ev.strategy_id and e.strategy_version == ev.strategy_version
    ]
    d = diagnose(mine)
    return Lesson(
        lesson_id=_id(event, ev.strategy_id, ev.strategy_version, ev.at.isoformat()),
        at=ev.at,
        event=event,
        strategy_id=ev.strategy_id,
        strategy_version=ev.strategy_version,
        family=family,
        what_tried=f"{ev.strategy_id} {ev.strategy_version} at {ev.from_stage.value}"
        + (f" with {json.dumps(dict(params), sort_keys=True)}" if params else ""),
        what_failed=f"{event} to {ev.to_stage.value}: {ev.reason}",
        suspected_cause=d.finding,
        evidence={"diagnosis": d.model_dump(mode="json"), "journal_signals": len(mine)},
    )


def lesson_from_validation(
    report: Mapping[str, Any], entries: Sequence[JournalEntry], at: datetime
) -> Lesson | None:
    """A lesson from a failed `at validate` report (its JSON form)."""
    checks = report.get("checks") or []
    failed = [c for c in checks if not c.get("passed")]
    if checks and not failed:
        return None
    sid, ver = str(report["strategy_id"]), str(report["version"])
    first = failed[0] if failed else None
    mine = [e for e in entries if e.strategy_id == sid]
    d = diagnose(mine)
    return Lesson(
        lesson_id=_id("failed_validation", sid, ver, str(report.get("code_hash", ""))),
        at=at,
        event="failed_validation",
        strategy_id=sid,
        strategy_version=ver,
        family=str(report.get("family", "")),
        what_tried=f"{sid} {ver} with {json.dumps(report.get('final_params', {}), sort_keys=True)}",
        what_failed="failed validation: "
        + (
            ", ".join(f"{c['name']} {c['value']:.4g} (needs {c['op']} {c['threshold']})" for c in failed)
            or "no checks ran"
        ),
        metric=first["name"] if first else None,
        value=float(first["value"]) if first else None,
        threshold=f"{first['op']} {first['threshold']}" if first else None,
        suspected_cause=d.finding
        if mine
        else "not traded live yet: the report's failed checks are the evidence",
        evidence={
            "failed_checks": failed,
            "oos_trades": len(report.get("oos_trades") or []),
            "diagnosis": d.model_dump(mode="json") if mine else None,
        },
    )


# ---------------------------------------------------------------- storage and the service


class LessonBook:
    """Append-only JSONL, one lesson per id. SPEC-QUESTION: the spec retrieves lessons by pgvector similarity;
    until the Postgres store exists, `top` returns the newest lessons of the family."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.lessons: dict[str, Lesson] = {}
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    x = Lesson.model_validate_json(line)
                    self.lessons[x.lesson_id] = x

    def add(self, lesson: Lesson) -> bool:
        if lesson.lesson_id in self.lessons:
            return False
        self.lessons[lesson.lesson_id] = lesson
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(lesson.model_dump_json() + "\n")
        return True

    def top(self, family: str | None = None, k: int = 5) -> list[Lesson]:
        xs = [x for x in self.lessons.values() if family is None or x.family == family]
        return sorted(xs, key=lambda x: x.at, reverse=True)[:k]


class LessonService:
    """Writes a lesson on every system demotion or retirement it sees on the bus."""

    name = "lessons"

    def __init__(
        self,
        book: LessonBook,
        entries: Callable[[], Sequence[JournalEntry]],
        families: Mapping[str, str],
        params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.book, self.entries, self.families, self.params = (
            book,
            entries,
            dict(families),
            dict(params or {}),
        )

    def handlers(self) -> Mapping[str, Handler]:
        return {STAGES: self._on_stage}

    async def _on_stage(self, msg: Any) -> None:
        if isinstance(msg, StageChanged):
            x = lesson_from_stage(
                msg, self.families.get(msg.strategy_id, ""), self.entries(), self.params.get(msg.strategy_id)
            )
            if x is not None:
                self.book.add(x)
