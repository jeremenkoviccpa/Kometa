"""Execution on the bus: signed decisions -> orders; halts, demotions and strategy requests -> actions;
the broker's account, quotes and closed trades -> everyone else.

Execution never acts on an intent alone: it waits for the matching RiskDecision, which the order
manager verifies (signature, intent, expiry, sequence). Risk-reducing requests (close, tighten, cancel)
from a strategy apply only to that strategy's own orders and positions.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from autotrader.core.alerts import Severity
from autotrader.core.broker import Quote, client_order_id, intent_id_for, is_system_comment
from autotrader.core.bus import (
    ACCOUNT,
    CONTROL,
    DECISIONS,
    HALTS,
    HEARTBEATS,
    INTENTS,
    ORDERS,
    QUOTES,
    REQUESTS,
    TRADES,
    Bus,
    Handler,
)
from autotrader.core.events import (
    AccountUpdate,
    DemotionOrder,
    Event,
    Exposure,
    HaltEntered,
    Heartbeat,
    OrderFilled,
    OrderIntentCreated,
    PositionClosed,
    QuoteUpdate,
    RiskDecided,
    StrategyRequestEmitted,
)
from autotrader.core.models import CancelRequest, CloseRequest, Fill, HaltCommand, ModifyStopRequest, Trade
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.watchdog import Watchdog

log = logging.getLogger(__name__)
INTENT_CACHE = 10_000


class ExecutionBusService:
    name = "execution"

    def __init__(self, om: OrderManager, watchdog: Watchdog, bus: Bus) -> None:
        self.om = om
        self.watchdog = watchdog
        self.bus = bus
        self.intents: OrderedDict[str, OrderIntentCreated] = OrderedDict()
        self._outbox: list[tuple[str, Event]] = []  # in the order things happened
        om.on_trade = self._trade
        om.on_order = self._order
        om.on_fill = self._fill

    def handlers(self) -> Mapping[str, Handler]:
        return {
            stream: self._then_flush(h)
            for stream, h in {
                HEARTBEATS: self._on_heartbeat,
                HALTS: self._on_halt,
                CONTROL: self._on_control,
                INTENTS: self._on_intent,
                DECISIONS: self._on_decision,
                REQUESTS: self._on_request,
            }.items()
        }

    def _then_flush(self, h: Handler) -> Handler:
        async def run(msg: Any) -> None:
            await h(msg)
            await self.flush()

        return run

    # ------------------------------------------------------------ outbound

    def _trade(self, t: Trade) -> None:
        self._outbox.append((TRADES, PositionClosed(at=t.exit_time, trade=t)))

    def _order(self, e: Event) -> None:
        self._outbox.append((ORDERS, e))

    def _fill(self, f: Fill) -> None:
        self._outbox.append((ORDERS, OrderFilled(at=f.filled_at, fill=f)))

    async def flush(self) -> None:
        while self._outbox:
            stream, msg = self._outbox.pop(0)
            await self.bus.publish(stream, msg)

    async def heartbeat(self) -> None:
        await self.bus.publish(HEARTBEATS, Heartbeat(at=self.om.clock.now(), service=self.name))

    async def on_quote(self, q: Quote) -> None:
        self.om.quotes.update(q)
        await self.bus.publish(
            QUOTES, QuoteUpdate(at=q.time, symbol=q.symbol, bid=float(q.bid), ask=float(q.ask))
        )

    async def publish_account(self) -> AccountUpdate:
        """The account as the broker sees it, with each system position mapped to its strategy version."""
        om = self.om
        acct = await om.adapter.account()
        positions = await om.adapter.open_positions()
        orders = await om.adapter.pending_orders()
        exposures = []
        for p in positions:
            t = om.state.orders.get(p.comment) if is_system_comment(p.comment) else None
            exposures.append(
                Exposure(
                    symbol=p.symbol,
                    side=p.side,
                    lots=p.lots,
                    entry=p.price_open,
                    stop=p.sl,
                    strategy_id=t.strategy_id if t else None,
                    strategy_version=t.strategy_version if t else None,
                    position_id=p.position_id,
                    external=not is_system_comment(p.comment),
                )
            )
        for o in orders:
            t = om.state.orders.get(o.comment) if is_system_comment(o.comment) else None
            exposures.append(
                Exposure(
                    symbol=o.symbol,
                    side=o.side,
                    lots=o.lots,
                    entry=o.price,
                    stop=o.sl,
                    strategy_id=t.strategy_id if t else None,
                    strategy_version=t.strategy_version if t else None,
                    pending=True,
                    external=not is_system_comment(o.comment),
                )
            )
        quotes = tuple(q for s in om.symbols if (q := om.quotes.get(s)) is not None)
        margin = {s: i.margin_per_lot for s, i in om.symbols.items() if i.margin_per_lot is not None}
        upd = AccountUpdate(
            at=om.clock.now(),
            account_id=acct.account_id,
            currency=acct.currency,
            balance=acct.balance,
            equity=acct.equity,
            free_margin=acct.free_margin,
            exposures=tuple(exposures),
            quotes=quotes,
            margin_per_lot=margin,
        )
        await self.bus.publish(ACCOUNT, upd)
        return upd

    async def cycle(self) -> None:
        """Periodic: deals, stops, expiry (order manager), then the account for the other services."""
        await self.om.sync_deals()
        await self.om.expire_pending()
        await self.flush()
        await self.publish_account()

    # ------------------------------------------------------------ inbound

    async def _on_heartbeat(self, msg: Any) -> None:
        if isinstance(msg, Heartbeat):
            self.watchdog.beat(msg.service, msg.at)

    async def _on_halt(self, msg: Any) -> None:
        if isinstance(msg, HaltEntered):
            await self.om.apply_halt(HaltCommand.for_state(msg.state, msg.reason, msg.at))

    async def _on_control(self, msg: Any) -> None:
        if isinstance(msg, DemotionOrder):
            await self.om.cancel_pending_of(msg.strategy_id, msg.strategy_version, msg.reason)
            if msg.close_positions:
                await self.om.close_positions_of(msg.strategy_id, msg.strategy_version, msg.reason)

    async def _on_intent(self, msg: Any) -> None:
        if isinstance(msg, OrderIntentCreated):
            self.intents[str(msg.intent.intent_id)] = msg
            while len(self.intents) > INTENT_CACHE:
                self.intents.popitem(last=False)

    async def _on_decision(self, msg: Any) -> None:
        if not isinstance(msg, RiskDecided) or msg.decision.verdict == "reject":
            return
        created = self.intents.get(str(msg.decision.intent_id))
        if created is None:
            self.om.alert(
                Severity.WARNING, "decision_without_intent", "approved decision for an unknown intent"
            )
            return
        r = await self.om.execute(created.intent, msg.decision, created.timeframe)
        if not r.placed:
            log.info("not placed %s: %s", r.client_order_id, r.reason)

    async def _on_request(self, msg: Any) -> None:
        if not isinstance(msg, StrategyRequestEmitted):
            return
        req = msg.request
        om = self.om
        if isinstance(req, CancelRequest):
            t = om.state.orders.get(client_order_id(intent_id_for(req.signal_id)))
            if (
                t is not None
                and t.strategy_id == req.strategy_id
                and t.state == "pending"
                and t.broker_order_id
            ):
                ack = await om.adapter.cancel(t.broker_order_id)
                if ack.ok:
                    t.state, t.note = "cancelled", f"strategy: {req.reason}"
                    om.save()
            return
        t = om.state.by_position(req.position_id)
        if t is None or t.strategy_id != req.strategy_id or t.strategy_version != req.strategy_version:
            om.alert(
                Severity.WARNING,
                "foreign_request",
                f"{req.strategy_id} asked about a position it does not own",
            )
            return
        if isinstance(req, CloseRequest):
            await om.close(req.position_id, f"strategy: {req.reason}")
        elif isinstance(req, ModifyStopRequest):
            await om.tighten_stop(req.position_id, req.new_stop)
