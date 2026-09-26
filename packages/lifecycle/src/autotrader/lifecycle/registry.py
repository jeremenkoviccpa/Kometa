"""Strategy registry and the promotion ladder state machine (spec section 10).

Everything is an append-only ledger record (`strategy_version`, `profile`, `stage_change`); the
current state is a replay of the ledger, so history cannot be rewritten. New versions enter only
through `submit_candidate` (the one door research and learning may use, spec section 4).

    candidate -> shadow -> micro -> live -> scaled
    scaled -> live            (rolling stats slip, or any demotion rule)
    live -> micro             (any demotion rule)
    micro -> retired          (second demotion within the retire window)
    micro -> shadow           (first demotion while in micro; SPEC-QUESTION, see open_questions.md)
    shadow -> retired         (signals diverge from backtest)
    shadow <-> demo_only      (owner only, demo_only versions only: paper trading at minimum size;
                               the risk gate trades it only if the signed risk config grants the stage)
    candidate -> retired      (failed validation)
    any -> retired            (owner command)

Illegal transitions raise and alert. demo_only versions can never leave shadow except to retire.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, NoReturn

from pydantic import Field

from autotrader.core.alerts import Alert, AlertSink, Severity
from autotrader.core.clock import Clock
from autotrader.core.events import StageChanged
from autotrader.core.ledger import Ledger
from autotrader.core.models import Frozen, Stage, UtcDatetime
from autotrader.core.profile import BacktestProfile

Origin = Literal["trader", "research_agent", "owner", "learning_reopt"]  # same as the strategy manifest
Actor = Literal["evaluator", "validation", "owner", "learning"]
# the owner's paper-trading switch (read by learning: switching off is a choice, not a failure)
PAPER_ON, PAPER_OFF = "owner: paper trading", "owner: back to shadow"
SWAP_IN, SWAP_OUT, ROLLBACK = "swap: replaces", "swap: replaced by", "rollback:"

ALLOWED: frozenset[tuple[Stage, Stage]] = frozenset(
    {
        (Stage.CANDIDATE, Stage.SHADOW),
        (Stage.SHADOW, Stage.MICRO),
        (Stage.MICRO, Stage.LIVE),
        (Stage.LIVE, Stage.SCALED),
        (Stage.SCALED, Stage.LIVE),
        (Stage.LIVE, Stage.MICRO),
        (Stage.MICRO, Stage.RETIRED),
        (Stage.MICRO, Stage.SHADOW),
        (Stage.SHADOW, Stage.RETIRED),
        (Stage.CANDIDATE, Stage.RETIRED),
        (Stage.SHADOW, Stage.DEMO_ONLY),
        (Stage.DEMO_ONLY, Stage.SHADOW),
        (Stage.DEMO_ONLY, Stage.RETIRED),
    }
)
MONEY_STAGES = frozenset({Stage.MICRO, Stage.LIVE, Stage.SCALED})
DEMOTIONS = frozenset(
    {
        (Stage.SCALED, Stage.LIVE),
        (Stage.LIVE, Stage.MICRO),
        (Stage.MICRO, Stage.SHADOW),
        (Stage.MICRO, Stage.RETIRED),
    }
)


class IllegalTransitionError(Exception):
    pass


class VersionInfo(Frozen):
    """What a new version brings. Built from the strategy manifest by the caller."""

    strategy_id: str
    version: str
    family: str
    origin: Origin
    demo_only: bool
    code_hash: str
    params: dict[str, Any] = Field(default_factory=dict)
    parent_version: str | None = None
    created_by: str
    tenant_id: str = "default"


class StageRecord(Frozen):
    strategy_id: str
    version: str
    from_stage: Stage | None
    to_stage: Stage
    reason: str
    actor: Actor
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    at: UtcDatetime


class VersionState(Frozen):
    info: VersionInfo
    stage: Stage
    stage_since: UtcDatetime
    history: tuple[StageRecord, ...]
    profile: BacktestProfile | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.info.strategy_id, self.info.version)

    def demotions_since(self, t: datetime) -> int:
        return sum(
            1 for h in self.history if h.at >= t and h.from_stage and (h.from_stage, h.to_stage) in DEMOTIONS
        )

    def money_since(self) -> datetime | None:
        """Start of the current run of money stages (last entry into micro from shadow)."""
        if self.stage not in MONEY_STAGES:
            return None
        start = None
        for h in self.history:
            if h.to_stage == Stage.MICRO and h.from_stage == Stage.SHADOW:
                start = h.at
        return start


class Registry:
    def __init__(
        self,
        ledger: Ledger,
        clock: Clock,
        alerts: AlertSink,
        on_stage_change: Callable[[StageChanged], None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.clock = clock
        self.alerts = alerts
        self.on_stage_change = on_stage_change
        self._v: dict[tuple[str, str], VersionState] = {}
        self.swap_tests: list[dict[str, Any]] = []
        for rec in ledger.records():
            self._apply(rec["kind"], rec["payload"])

    # ------------------------------------------------------------ replay

    def _apply(self, kind: str, p: dict[str, Any]) -> None:
        if kind == "strategy_version":
            info = VersionInfo.model_validate(p["info"])
            at = datetime.fromisoformat(p["at"])
            first = StageRecord(
                strategy_id=info.strategy_id,
                version=info.version,
                from_stage=None,
                to_stage=Stage.CANDIDATE,
                reason="submitted",
                actor="validation",
                at=at,
            )
            self._v[(info.strategy_id, info.version)] = VersionState(
                info=info, stage=Stage.CANDIDATE, stage_since=at, history=(first,)
            )
        elif kind == "profile":
            prof = BacktestProfile.model_validate(p)
            key = (prof.strategy_id, prof.strategy_version)
            self._v[key] = self._v[key].model_copy(update={"profile": prof})
        elif kind == "swap_test":
            self.swap_tests.append(p)
        elif kind == "stage_change":
            r = StageRecord.model_validate(p)
            key = (r.strategy_id, r.version)
            s = self._v[key]
            self._v[key] = s.model_copy(
                update={"stage": r.to_stage, "stage_since": r.at, "history": (*s.history, r)}
            )

    # ------------------------------------------------------------ reads

    def get(self, strategy_id: str, version: str) -> VersionState:
        try:
            return self._v[(strategy_id, version)]
        except KeyError:
            raise KeyError(f"unknown version {strategy_id} {version}") from None

    def versions(self, stage: Stage | None = None) -> list[VersionState]:
        return [v for v in self._v.values() if stage is None or v.stage == stage]

    def stages(self) -> dict[tuple[str, str], Stage]:
        """(strategy_id, version) -> stage, the map the risk gate's stage check reads."""
        return {k: v.stage for k, v in self._v.items()}

    def promotions_to(self, stage: Stage, since: datetime) -> int:
        return sum(1 for v in self._v.values() for h in v.history if h.to_stage == stage and h.at >= since)

    # ------------------------------------------------------------ writes

    def submit_candidate(self, info: VersionInfo, profile: BacktestProfile | None = None) -> VersionState:
        """The only way a new version enters the system. It always starts as a candidate."""
        key = (info.strategy_id, info.version)
        if key in self._v:
            raise ValueError(f"{info.strategy_id} {info.version} already exists; versions are immutable")
        payload = {"info": info.model_dump(mode="json"), "at": self.clock.now().isoformat()}
        self.ledger.append("strategy_version", payload)
        self._apply("strategy_version", payload)
        if profile is not None:
            self.set_profile(profile)
        return self._v[key]

    def set_profile(self, profile: BacktestProfile) -> None:
        self.get(profile.strategy_id, profile.strategy_version)
        payload = profile.model_dump(mode="json")
        self.ledger.append("profile", payload)
        self._apply("profile", payload)

    def transition(
        self,
        strategy_id: str,
        version: str,
        to: Stage,
        reason: str,
        *,
        actor: Actor,
        metrics: dict[str, float | int | str | bool | None] | None = None,
    ) -> StageChanged:
        v = self.get(strategy_id, version)
        frm = v.stage
        legal = (frm, to) in ALLOWED or (to == Stage.RETIRED and actor == "owner" and frm != Stage.RETIRED)
        if v.info.demo_only and frm == Stage.SHADOW and to not in (Stage.RETIRED, Stage.DEMO_ONLY):
            legal = False
        if to == Stage.DEMO_ONLY and not (v.info.demo_only and actor == "owner"):
            legal = False  # the paper stage is for demo_only versions, entered only by the owner
        if (
            legal
            and (frm, to) == (Stage.CANDIDATE, Stage.SHADOW)
            and v.profile is None
            and not v.info.demo_only
        ):
            legal = False  # nothing to compare shadow against without a validated profile
        if not legal:
            self._illegal(
                f"illegal transition {frm.value} -> {to.value} for {strategy_id} {version} ({reason})"
            )
        return self._record(v, to, reason, actor, metrics)

    def _illegal(self, msg: str) -> NoReturn:
        self.alerts.send(
            Alert(severity=Severity.CRITICAL, kind="illegal_transition", message=msg, at=self.clock.now())
        )
        raise IllegalTransitionError(msg)

    def _record(
        self,
        v: VersionState,
        to: Stage,
        reason: str,
        actor: Actor,
        metrics: dict[str, float | int | str | bool | None] | None = None,
    ) -> StageChanged:
        strategy_id, version = v.key
        frm = v.stage
        now = self.clock.now()
        clean = {
            k: (None if isinstance(x, float) and not math.isfinite(x) else x)
            for k, x in (metrics or {}).items()
        }
        rec = StageRecord(
            strategy_id=strategy_id,
            version=version,
            from_stage=frm,
            to_stage=to,
            reason=reason,
            actor=actor,
            metrics=clean,  # a profit factor with no losses is inf; JSON cannot hold it
            at=now,
        )
        payload = rec.model_dump(mode="json")
        self.ledger.append("stage_change", payload)
        self._apply("stage_change", payload)
        event = StageChanged(
            at=now,
            strategy_id=strategy_id,
            strategy_version=version,
            from_stage=frm,
            to_stage=to,
            reason=reason,
        )
        demotion = (frm, to) in DEMOTIONS or to == Stage.RETIRED
        self.alerts.send(
            Alert(
                severity=Severity.WARNING if demotion else Severity.INFO,
                kind="demotion" if demotion else "promotion",
                message=f"{strategy_id} {version}: {frm.value} -> {to.value} ({reason})",
                at=now,
                details={"strategy_id": strategy_id, "version": version, "to": to.value},
            )
        )
        if self.on_stage_change is not None:
            self.on_stage_change(event)
        return event

    # ------------------------------------------------------------ champion vs challenger (spec 14.8)

    def swap(
        self,
        strategy_id: str,
        champion: str,
        challenger: str,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> tuple[StageChanged, StageChanged]:
        """The challenger takes the champion's stage; the champion goes to shadow (kept 4 weeks for rollback).
        Only a re-optimized child of the champion, in shadow, may replace a champion in a money stage."""
        champ, chal = self.get(strategy_id, champion), self.get(strategy_id, challenger)
        if not (
            chal.info.origin == "learning_reopt"
            and chal.info.parent_version == champion
            and chal.stage == Stage.SHADOW
            and champ.stage in MONEY_STAGES
        ):
            self._illegal(
                f"illegal swap {strategy_id} {champion} ({champ.stage.value}) <- {challenger} "
                f"({chal.stage.value}, parent {chal.info.parent_version}): {reason}"
            )
        stage = champ.stage
        a = self._record(chal, stage, f"{SWAP_IN} {champion}: {reason}", "evaluator", metrics)
        b = self._record(
            champ, Stage.SHADOW, f"{SWAP_OUT} {challenger}; kept in shadow for rollback", "evaluator"
        )
        return a, b

    def last_swap(self, strategy_id: str) -> tuple[datetime, str, str, Stage] | None:
        """(when, new champion, old champion, the stage it took) of the strategy's latest swap."""
        best = None
        for v in self._v.values():
            if v.info.strategy_id != strategy_id:
                continue
            for h in v.history:
                if h.reason.startswith(SWAP_IN) and (best is None or h.at > best[0]):
                    old = h.reason[len(SWAP_IN) :].split(":")[0].strip()
                    best = (h.at, h.version, old, h.to_stage)
        return best

    def rollback(self, strategy_id: str, reason: str) -> tuple[StageChanged, ...]:
        """Undo the latest swap: the old champion gets its stage back, the new one goes to shadow."""
        last = self.last_swap(strategy_id)
        if last is None:
            self._illegal(f"no swap to roll back for {strategy_id}: {reason}")
        _at, new, old, stage = last
        old_v, new_v = self.get(strategy_id, old), self.get(strategy_id, new)
        if old_v.stage != Stage.SHADOW:
            self._illegal(f"cannot roll back {strategy_id}: {old} is {old_v.stage.value}, not shadow")
        out = []
        if new_v.stage != Stage.SHADOW:
            out.append(self._record(new_v, Stage.SHADOW, f"{ROLLBACK} {old}: {reason}", "evaluator"))
        out.append(self._record(old_v, stage, f"{ROLLBACK} restored over {new}: {reason}", "evaluator"))
        return tuple(out)

    def record_swap_test(self, payload: dict[str, Any]) -> None:
        """Every swap test, passed or failed, stays in the ledger (spec 14.8)."""
        self.ledger.append("swap_test", payload)
        self._apply("swap_test", payload)

    def promote_candidate(
        self, strategy_id: str, version: str, *, validation_passed: bool, synthetic: bool
    ) -> StageChanged:
        """Validation's verdict. Real-data passes go to shadow; demo_only versions may enter shadow
        (it risks nothing) to exercise the plumbing, and are capped there."""
        v = self.get(strategy_id, version)
        if v.info.demo_only:
            return self.transition(
                strategy_id, version, Stage.SHADOW, "demo_only: plumbing test", actor="validation"
            )
        if validation_passed and not synthetic:
            return self.transition(
                strategy_id, version, Stage.SHADOW, "passed validation", actor="validation"
            )
        why = "failed validation" if not validation_passed else "validated on synthetic data only"
        return self.transition(strategy_id, version, Stage.RETIRED, why, actor="validation")

    def retire(self, strategy_id: str, version: str, reason: str) -> StageChanged:
        """Owner command: any stage -> retired."""
        return self.transition(strategy_id, version, Stage.RETIRED, reason, actor="owner")

    def paper_trade(self, strategy_id: str, version: str, on: bool = True) -> StageChanged:
        """Owner command: a demo_only version trades at minimum size where risk config allows (paper)."""
        to = Stage.DEMO_ONLY if on else Stage.SHADOW
        return self.transition(strategy_id, version, to, PAPER_ON if on else PAPER_OFF, actor="owner")
