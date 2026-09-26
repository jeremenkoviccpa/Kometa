"""engine-live (spec section 18): runs shadow sessions and money-stage strategies on live quotes.

- Shadow versions run in ShadowSessions; their signals and trades go to the lifecycle marked shadow.
- Money versions (micro, live, scaled) run in LiveRunners on one shared bar builder; their signals go
  to the allocator, and close/tighten/cancel requests go to execution. A version publishes only while
  the lifecycle says it is in a money stage: a demotion silences it at once.
- Positions a money strategy sees are the broker's, mapped to it by execution (AccountUpdate).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from autotrader.core.broker import Quote
from autotrader.core.bus import ACCOUNT, HEARTBEATS, QUOTES, REQUESTS, SIGNALS, STAGES, TRADES, Bus, Handler
from autotrader.core.clock import Clock
from autotrader.core.events import (
    AccountUpdate,
    Heartbeat,
    PositionClosed,
    QuoteUpdate,
    SignalEmitted,
    StageChanged,
    StageSnapshot,
    StrategyRequestEmitted,
)
from autotrader.core.models import Signal, Stage, Timeframe, Trade
from autotrader.core.series import BarsArray
from autotrader.engine.live import LiveRunner
from autotrader.engine.live_bars import LiveBarBuilder
from autotrader.engine.market import NewsFn
from autotrader.engine.shadow import ShadowSession
from autotrader.strategies_api.base import PendingView, PositionView, Request, Strategy
from autotrader.strategies_api.manifest import ParamValue

MONEY = frozenset({Stage.MICRO, Stage.LIVE, Stage.SCALED, Stage.DEMO_ONLY})  # demo_only: paper, min size


class EngineLiveService:
    name = "engine"

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        *,
        shadow: Sequence[ShadowSession] = (),
        money: Sequence[tuple[type[Strategy], Mapping[str, ParamValue]]] = (),
        history: Mapping[tuple[str, Timeframe], BarsArray] | None = None,
        news_fn: NewsFn | None = None,
        shadow_signals: bool = False,
    ) -> None:
        """`history`: closed bars before the first live quote, so strategies start warmed up. They must
        end exactly where live bars begin (a trading-day boundary), or the first live bar would be partial."""
        self.bus = bus
        self.clock = clock
        # demo: a runner whose version is in shadow still publishes its signals, marked shadow (never sized),
        # so the owner sees what every strategy would trade; production runs shadow versions in sessions
        self.shadow_signals = shadow_signals
        self.stages: dict[tuple[str, str], Stage] = {}
        self.account: AccountUpdate | None = None
        self._out: list[tuple[str, Any]] = []
        self.shadow = list(shadow)
        for s in self.shadow:
            tf = s.tf
            s.on_signal = lambda sig, tf=tf: self._out.append(
                (SIGNALS, SignalEmitted(at=sig.created_at, signal=sig, timeframe=tf, shadow=True))
            )
            s.on_trade = self._shadow_trade
        self.runners = [
            LiveRunner(
                cls,
                spread_fn=self._spread,
                positions_fn=self._positions,
                pending_fn=self._pending,
                params=p,
                history=history,
                news_fn=news_fn,
            )
            for cls, p in money
        ]
        self.builder = LiveBarBuilder([k for r in self.runners for k in r.subs])
        self._quotes: dict[str, Quote] = {}

    def handlers(self) -> Mapping[str, Handler]:
        return {STAGES: self._on_stage, ACCOUNT: self._on_account, QUOTES: self._on_quote}

    # ------------------------------------------------------------ views for money strategies

    def _spread(self, symbol: str, _ns: int) -> float:
        q = self._quotes.get(symbol)
        return float(q.ask - q.bid) if q is not None else 0.0

    def _positions(self, strategy_id: str, symbol: str | None) -> list[PositionView]:
        if self.account is None:
            return []
        return [
            PositionView(
                e.position_id or "",
                "",
                e.symbol,
                e.side,
                float(e.lots),
                float(e.entry),
                float(e.stop) if e.stop is not None else 0.0,
                None,
                self.account.at,
            )
            for e in self.account.exposures
            if not e.pending and e.strategy_id == strategy_id and (symbol is None or e.symbol == symbol)
        ]

    def _pending(self, strategy_id: str, symbol: str | None) -> list[PendingView]:
        return []  # SPEC-QUESTION: pending views for live strategies need signal ids from execution

    # ------------------------------------------------------------ inbound

    async def _on_stage(self, msg: Any) -> None:
        if isinstance(msg, StageSnapshot):
            self.stages = {(sid, v): st for sid, v, st in msg.stages}
        elif isinstance(msg, StageChanged):
            self.stages[(msg.strategy_id, msg.strategy_version)] = msg.to_stage

    async def _on_account(self, msg: Any) -> None:
        if isinstance(msg, AccountUpdate):
            self.account = msg

    async def _on_quote(self, msg: Any) -> None:
        if not isinstance(msg, QuoteUpdate):
            return
        q = Quote(symbol=msg.symbol, bid=Decimal(str(msg.bid)), ask=Decimal(str(msg.ask)), time=msg.at)
        self._quotes[q.symbol] = q
        for s in self.shadow:
            s.on_quote(q)
        for r in self.runners:
            self._route(r, r.on_bars(self.builder.on_quote(q)))
        await self.flush()

    async def on_time(self, now: datetime) -> None:
        """Ticker: close bars on time when a market goes quiet."""
        for s in self.shadow:
            s.on_time(now)
        events = self.builder.on_time(now)
        for r in self.runners:
            self._route(r, r.on_bars(events))
        await self.flush()

    # ------------------------------------------------------------ outbound

    def _shadow_trade(self, t: Trade) -> None:
        self._out.append((TRADES, PositionClosed(at=t.exit_time, trade=t)))

    def _route(self, r: LiveRunner, reqs: list[Request]) -> None:
        key = (r.manifest.id, r.manifest.version)
        tf = min(r.manifest.timeframes, key=lambda t: t.minutes)
        if self.stages.get(key) not in MONEY:
            if self.shadow_signals and self.stages.get(key) == Stage.SHADOW:
                for req in reqs:
                    if isinstance(req, Signal):
                        self._out.append(
                            (SIGNALS, SignalEmitted(at=req.created_at, signal=req, timeframe=tf, shadow=True))
                        )
            return  # demoted or not promoted yet: never sized, never an order
        for req in reqs:
            if isinstance(req, Signal):
                self._out.append((SIGNALS, SignalEmitted(at=req.created_at, signal=req, timeframe=tf)))
            else:
                self._out.append((REQUESTS, StrategyRequestEmitted(at=self.clock.now(), request=req)))

    async def flush(self) -> None:
        while self._out:
            stream, msg = self._out.pop(0)
            await self.bus.publish(stream, msg)

    async def heartbeat(self) -> None:
        await self.bus.publish(HEARTBEATS, Heartbeat(at=self.clock.now(), service=self.name))
