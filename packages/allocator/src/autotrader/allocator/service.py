"""The allocator service: money-stage signals -> order intents with proposed lots.

Shadow signals are ignored (shadow never sizes). Without an account update, a quote for the symbol or a
conversion rate, a signal is dropped: the allocator never guesses a size.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from autotrader.allocator.allocator import Allocation, Allocator
from autotrader.allocator.weights import VersionPerformance
from autotrader.core.bus import ACCOUNT, INTENTS, SIGNALS, STAGES, Bus, Handler
from autotrader.core.clock import Clock
from autotrader.core.events import (
    AccountUpdate,
    OrderIntentCreated,
    SignalEmitted,
    StageChanged,
    StageSnapshot,
)
from autotrader.core.fx import rate_from_quotes
from autotrader.core.models import Instrument, Stage

log = logging.getLogger(__name__)


class AllocatorService:
    name = "allocator"

    def __init__(
        self, allocator: Allocator, bus: Bus, clock: Clock, instruments: Mapping[str, Instrument]
    ) -> None:
        self.allocator = allocator
        self.bus = bus
        self.clock = clock
        self.instruments = dict(instruments)
        self.account: AccountUpdate | None = None
        self.stages: dict[tuple[str, str], Stage] = {}
        self.dropped: list[str] = []

    def handlers(self) -> Mapping[str, Handler]:
        return {STAGES: self._on_stage, ACCOUNT: self._on_account, SIGNALS: self._on_signal}

    def rebalance(self, perf: Sequence[VersionPerformance], now: datetime | None = None) -> Allocation:
        """Weekly job (scheduler). Not due: the current allocation stays."""
        return self.allocator.rebalance(perf, now or self.clock.now())

    async def _on_stage(self, msg: Any) -> None:
        if isinstance(msg, StageSnapshot):
            self.stages = {(sid, v): st for sid, v, st in msg.stages}
        elif isinstance(msg, StageChanged):
            self.stages[(msg.strategy_id, msg.strategy_version)] = msg.to_stage

    async def _on_account(self, msg: Any) -> None:
        if isinstance(msg, AccountUpdate):
            self.account = msg

    def _drop(self, why: str) -> None:
        self.dropped.append(why)
        log.info("signal dropped: %s", why)

    async def _on_signal(self, msg: Any) -> None:
        if not isinstance(msg, SignalEmitted) or msg.shadow:
            return
        s = msg.signal
        stage = self.stages.get((s.strategy_id, s.strategy_version), Stage.CANDIDATE)
        acct, inst = self.account, self.instruments.get(s.symbol)
        if acct is None or inst is None:
            return self._drop(f"{s.signal_id}: no account data or unknown instrument")
        quotes = {q.symbol: q for q in acct.quotes}
        q = quotes.get(s.symbol)
        if q is None:
            return self._drop(f"{s.signal_id}: no quote for {s.symbol}")
        entry = (
            (q.ask if s.side == "buy" else q.bid) if s.entry_type == "market" else Decimal(str(s.entry_price))
        )
        try:
            fx = rate_from_quotes(quotes, inst.quote, acct.currency)
        except KeyError as e:
            return self._drop(f"{s.signal_id}: {e}")
        intent = self.allocator.propose(
            s,
            stage,
            equity=acct.equity,
            entry=entry,
            instrument=inst,
            to_account=fx,
            account_id=acct.account_id,
        )
        if intent is None:
            return self._drop(
                f"{s.signal_id}: no budget for {s.strategy_id} {s.strategy_version} ({stage.value})"
            )
        await self.bus.publish(
            INTENTS, OrderIntentCreated(at=self.clock.now(), intent=intent, timeframe=msg.timeframe)
        )
        return None
