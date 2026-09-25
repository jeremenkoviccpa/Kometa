"""The lifecycle service: closed trades and shadow signals in, stage changes and demotion orders out.

The evaluator runs after every closed trade and on the hourly job; a stage snapshot goes out at start
and hourly so services that restarted catch up.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autotrader.core.bus import CONTROL, SIGNALS, STAGES, TRADES, Bus, Handler
from autotrader.core.clock import Clock
from autotrader.core.events import DemotionOrder, PositionClosed, SignalEmitted, StageChanged, StageSnapshot
from autotrader.lifecycle.evaluator import Evaluator, MemoryStageData
from autotrader.lifecycle.registry import Registry


class BusDemotionActions:
    """DemotionActions for the evaluator: orders go to execution over the bus (queued, then flushed)."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.queue: list[DemotionOrder] = []

    async def cancel_pending_of(self, strategy_id: str, version: str, reason: str) -> None:
        self.queue.append(
            DemotionOrder(
                at=self.clock.now(),
                strategy_id=strategy_id,
                strategy_version=version,
                close_positions=False,
                reason=reason,
            )
        )

    async def close_positions_of(self, strategy_id: str, version: str, reason: str) -> None:
        self.queue.append(
            DemotionOrder(
                at=self.clock.now(),
                strategy_id=strategy_id,
                strategy_version=version,
                close_positions=True,
                reason=reason,
            )
        )


class LifecycleService:
    name = "lifecycle"

    def __init__(
        self, registry: Registry, evaluator: Evaluator, data: MemoryStageData, bus: Bus, clock: Clock
    ) -> None:
        self.reg = registry
        self.ev = evaluator
        self.data = data
        self.bus = bus
        self.clock = clock
        self.actions = BusDemotionActions(clock)
        evaluator.actions = self.actions
        self._changes: list[StageChanged] = []
        registry.on_stage_change = self._changes.append

    def handlers(self) -> Mapping[str, Handler]:
        return {SIGNALS: self._on_signal, TRADES: self._on_trade}

    async def flush(self) -> None:
        # stage changes first: the risk gate and allocator must know before execution acts
        while self._changes:
            await self.bus.publish(STAGES, self._changes.pop(0))
        while self.actions.queue:
            await self.bus.publish(CONTROL, self.actions.queue.pop(0))

    async def publish_snapshot(self) -> None:
        stages = tuple((sid, v, st) for (sid, v), st in sorted(self.reg.stages().items()))
        await self.bus.publish(STAGES, StageSnapshot(at=self.clock.now(), stages=stages))

    async def hourly(self) -> None:
        await self.ev.evaluate_all()
        await self.flush()
        await self.publish_snapshot()

    async def _on_signal(self, msg: Any) -> None:
        if isinstance(msg, SignalEmitted) and msg.shadow:
            s = msg.signal
            self.data.add_signal(s.strategy_id, s.strategy_version, s.created_at)

    async def _on_trade(self, msg: Any) -> None:
        if isinstance(msg, PositionClosed):
            self.data.add_trade(msg.trade)
            await self.ev.on_trade_closed(msg.trade)
            await self.flush()
