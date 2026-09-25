"""The risk-gate service (spec sections 11 and 18): the gate on the bus.

- account updates (from execution) feed the loss-halt checks; a new halt is published at once
- every order intent gets a signed decision; without fresh account data the answer is a signed reject
- stages come from the lifecycle (snapshot + changes); an unknown version does not trade
- on start, an existing halt is published again so execution finishes acting on it after a restart
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from autotrader.core.alerts import Alert, AlertSink, Severity
from autotrader.core.bus import ACCOUNT, CONTROL, HALTS, HEARTBEATS, INTENTS, STAGES, Bus, Handler
from autotrader.core.bus import DECISIONS as DECISIONS_STREAM
from autotrader.core.clock import Clock
from autotrader.core.events import (
    AccountUpdate,
    HaltCleared,
    HaltEntered,
    Heartbeat,
    OrderIntentCreated,
    ResumeRequested,
    RiskDecided,
    StageChanged,
    StageSnapshot,
)
from autotrader.core.fx import rate_from_quotes
from autotrader.core.indicators.sessions import EventIndex
from autotrader.core.models import HaltCommand, HaltState, Instrument, Stage
from autotrader.core.timeutil import ensure_utc
from autotrader.risk.config import ConfigSignatureError, RiskLimits, load_signed
from autotrader.risk.gate import Exposure, RiskGate, Snapshot


class RiskGateService:
    name = "risk-gate"

    def __init__(
        self,
        gate: RiskGate,
        bus: Bus,
        clock: Clock,
        instruments: Mapping[str, Instrument],
        *,
        max_account_age: timedelta = timedelta(seconds=10),
        news: Callable[[], EventIndex | None] | None = None,
    ) -> None:
        """`news`: the economic calendar for check 8. None means no calendar is configured (simulated
        markets, whose clock is not the real one). A configured calendar that returns None is stale or
        unreachable, and then no entry is approved: trading blind into news is refused (fail closed)."""
        self.news = news
        self.gate = gate
        self.bus = bus
        self.clock = clock
        self.instruments = dict(instruments)
        self.max_account_age = max_account_age
        self.account: AccountUpdate | None = None
        self.stages: dict[tuple[str, str], Stage] = {}

    def handlers(self) -> Mapping[str, Handler]:
        return {
            STAGES: self._on_stage,
            ACCOUNT: self._on_account,
            INTENTS: self._on_intent,
            CONTROL: self._on_control,
        }

    async def start(self) -> None:
        """After a restart a halt survives (persisted); tell execution again so it finishes the job."""
        if self.gate.state.halt != HaltState.NORMAL:
            await self._publish_halt(
                HaltCommand.for_state(self.gate.state.halt, self.gate.state.halt_reason, self.clock.now())
            )

    async def heartbeat(self) -> None:
        await self.bus.publish(HEARTBEATS, Heartbeat(at=self.clock.now(), service=self.name))

    async def roll_day(self, *, new_week: bool) -> None:
        """Scheduler, at the 17:00 New York rollover: new loss references; expired halts clear."""
        if self.account is None:
            return
        before = self.gate.state.halt
        self.gate.roll_day(self.account.equity, self.clock.now(), new_week=new_week)
        if before != HaltState.NORMAL and self.gate.state.halt == HaltState.NORMAL:
            await self.bus.publish(HALTS, HaltCleared(at=self.clock.now(), previous=before, actor="rollover"))

    async def _on_control(self, msg: Any) -> None:
        """Owner resume of a FULL_HALT: only an owner-signed, unexpired, unused token clears it."""
        if not isinstance(msg, ResumeRequested):
            return
        before = self.gate.state.halt
        if self.gate.resume(msg.token, msg.signature, self.clock.now()):
            await self.bus.publish(HALTS, HaltCleared(at=self.clock.now(), previous=before, actor="owner"))

    async def _publish_halt(self, cmd: HaltCommand) -> None:
        await self.bus.publish(HALTS, HaltEntered(at=cmd.at, state=cmd.state, reason=cmd.reason))

    async def _on_stage(self, msg: Any) -> None:
        if isinstance(msg, StageSnapshot):
            self.stages = {(sid, v): st for sid, v, st in msg.stages}
        elif isinstance(msg, StageChanged):
            self.stages[(msg.strategy_id, msg.strategy_version)] = msg.to_stage

    async def _on_account(self, msg: Any) -> None:
        if not isinstance(msg, AccountUpdate):
            return
        self.account = msg
        halt = self.gate.on_account(msg.equity, self.clock.now())
        if halt is not None:
            await self._publish_halt(halt)

    def _snapshot(
        self, acct: AccountUpdate, symbol: str, now: datetime, events: EventIndex | None = None
    ) -> Snapshot:
        quotes = {q.symbol: q for q in acct.quotes}
        q = quotes[symbol]  # KeyError -> rejected by the caller
        return Snapshot(
            now=now,
            equity=acct.equity,
            free_margin=acct.free_margin,
            bid=q.bid,
            ask=q.ask,
            exposures=[
                Exposure(
                    symbol=e.symbol,
                    side=e.side,
                    lots=e.lots,
                    entry=e.entry,
                    stop=e.stop,
                    strategy_id=e.strategy_id,
                    pending=e.pending,
                    external=e.external,
                )
                for e in acct.exposures
            ],
            instruments=self.instruments,
            to_account=lambda ccy: rate_from_quotes(quotes, ccy, acct.currency),
            margin_per_lot=acct.margin_per_lot,
            stages=self.stages,
            events=events,
        )

    async def _on_intent(self, msg: Any) -> None:
        if not isinstance(msg, OrderIntentCreated):
            return
        now = self.clock.now()
        intent = msg.intent
        acct = self.account
        events = self.news() if self.news is not None else None
        if acct is None or now - acct.at > self.max_account_age:
            decision = self.gate.reject(intent, "no fresh account data", now)
        elif self.news is not None and events is None:
            # SPEC-QUESTION: the spec does not say what a missing calendar means; no news data is treated
            # like no market data (docs/open_questions.md 31)
            decision = self.gate.reject(intent, "economic calendar unavailable or stale", now)
        else:
            try:
                decision = self.gate.decide(intent, self._snapshot(acct, intent.signal.symbol, now, events))
            except KeyError as e:  # no quote for the symbol or for a conversion pair
                decision = self.gate.reject(intent, f"missing market data: {e}", now)
        await self.bus.publish(DECISIONS_STREAM, RiskDecided(at=now, decision=decision))


def load_limits_or_alert(
    config: Path, signature: Path, owner_key: Ed25519PublicKey, alerts: AlertSink, now: datetime
) -> tuple[RiskLimits, str]:
    """Start-up: a bad or missing signature refuses to start AND raises a critical alert (spec 11, 16)."""
    try:
        return load_signed(config, signature, owner_key)
    except ConfigSignatureError as e:
        alerts.send(
            Alert(severity=Severity.CRITICAL, kind="bad_config_signature", message=str(e), at=ensure_utc(now))
        )
        raise
