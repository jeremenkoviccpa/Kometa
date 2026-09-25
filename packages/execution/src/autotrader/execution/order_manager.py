"""Order manager (spec section 12): the only code that sends orders to a broker.

Rules it enforces:
- An entry order is sent only with a RiskDecision whose Ed25519 signature, intent, expiry and sequence
  verify (core.signing.DecisionVerifier), and never for more lots than the decision approved.
- `client_order_id` is derived from the intent; it is written to the journal before sending, and the
  broker is searched for it before every send, so retries and restarts never place twice.
- The stop is attached at placement and confirmed after every fill. A missing stop is set again;
  if that fails `stop_confirm_attempts` times, the position is closed and a critical alert goes out.
- Closing a position and tightening a stop only reduce risk and need no decision. Loosening is refused.
- Every change of a tracked order (placed, modified, filled, cancelled or expired) is reported through
  `on_order` when the journal is written, so the audit log sees each one (spec section 16).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

from autotrader.core.alerts import Alert, AlertSink, Severity
from autotrader.core.broker import (
    BrokerAck,
    BrokerPosition,
    ExecutionQuality,
    ModifyRequest,
    PlaceRequest,
    SymbolInfo,
    client_order_id,
    is_system_comment,
    magic_number,
)
from autotrader.core.clock import Clock
from autotrader.core.events import Event, OrderCancelled, OrderModified, OrderPlaced
from autotrader.core.models import (
    Fill,
    HaltCommand,
    Order,
    OrderIntent,
    OrderStatus,
    RiskDecision,
    Side,
    Timeframe,
    Trade,
)
from autotrader.core.signing import DecisionRejectedError, DecisionVerifier
from autotrader.execution.adapter import BrokerAdapter, BrokerUnavailableError
from autotrader.execution.config import ExecutionConfig
from autotrader.execution.journal import ExecState, Journal, OrderState, TrackedOrder
from autotrader.execution.quality import QualitySink
from autotrader.execution.quotes import QuoteBook

log = logging.getLogger(__name__)
ZERO = Decimal(0)
STATUS: dict[OrderState, OrderStatus] = {
    "sending": OrderStatus.NEW,
    "pending": OrderStatus.PLACED,
    "open": OrderStatus.FILLED,
    "closed": OrderStatus.FILLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "failed": OrderStatus.REJECTED,
}
_Seen = tuple[OrderState, Decimal, Decimal, str | None]  # what an order event reports


def _seen(t: TrackedOrder) -> _Seen:
    return (t.state, t.sl, t.lots, t.broker_order_id)


def order_view(t: TrackedOrder, now: datetime) -> Order:
    return Order(
        client_order_id=t.client_order_id,
        broker_order_id=t.broker_order_id,
        account_id=t.account_id,
        tenant_id=t.tenant_id,
        strategy_id=t.strategy_id,
        strategy_version=t.strategy_version,
        symbol=t.symbol,
        side=t.side,
        entry_type=t.order_type,
        status=STATUS[t.state],
        lots=t.lots,
        price=t.requested_price,
        sl=t.sl,
        tp=t.tp,
        created_at=t.created_at,
        updated_at=now,
    )


@dataclass(frozen=True)
class ExecResult:
    client_order_id: str
    state: OrderState | None
    placed: bool
    reason: str = ""


def round_price(x: float | Decimal, digits: int, mode: str = ROUND_HALF_EVEN) -> Decimal:
    return Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=mode)


def stop_rounding(side: Side) -> str:
    """Round a stop toward the entry, so rounding can only shrink the money at risk."""
    return ROUND_CEILING if side == "buy" else ROUND_FLOOR


def stop_protects(side: Side, broker_sl: Decimal | None, wanted: Decimal) -> bool:
    """The broker stop exists and is at least as tight as ours."""
    if broker_sl is None or broker_sl <= 0:
        return False
    return broker_sl >= wanted if side == "buy" else broker_sl <= wanted


class OrderManager:
    def __init__(
        self,
        *,
        adapter: BrokerAdapter,
        verifier: DecisionVerifier,
        journal: Journal,
        alerts: AlertSink,
        clock: Clock,
        config: ExecutionConfig,
        quotes: QuoteBook,
        quality: QualitySink,
        account_id: str,
        symbols: Mapping[str, SymbolInfo],
        on_fill: Callable[[Fill], None] | None = None,
        on_trade: Callable[[Trade], None] | None = None,
        on_order: Callable[[Event], None] | None = None,
        to_account: Callable[[str], Decimal] | None = None,
    ) -> None:
        self.adapter = adapter
        self.verifier = verifier
        self.journal = journal
        self.alerts = alerts
        self.clock = clock
        self.cfg = config
        self.quotes = quotes
        self.quality = quality
        self.account_id = account_id
        self.symbols = dict(symbols)
        self.on_fill = on_fill
        self.on_trade = on_trade
        self.on_order = on_order
        self.to_account = to_account or (lambda _symbol: Decimal(1))  # quote -> account currency
        self.state: ExecState = journal.load()  # raises JournalCorruptError: caller fails closed
        self.verifier.last_sequence = max(self.verifier.last_sequence, self.state.last_decision_sequence)
        self._reported: dict[str, _Seen] = {k: _seen(t) for k, t in self.state.orders.items()}

    # ------------------------------------------------------------ helpers

    def save(self) -> None:
        self.state.last_decision_sequence = self.verifier.last_sequence
        self.journal.save(self.state, self.clock.now())
        self._report_orders()

    def _report_orders(self) -> None:
        """One event per order whose state, stop, lots or broker id changed since it was last reported.
        Runs after the journal write, so an event never describes a state that could be lost."""
        now = self.clock.now()
        for coid, t in self.state.orders.items():
            seen = _seen(t)
            before = self._reported.get(coid)
            if before == seen:
                continue
            self._reported[coid] = seen
            if self.on_order is None:
                continue
            if t.state in ("cancelled", "expired"):
                reason = f"{t.state}: {t.note}" if t.note else t.state
                self.on_order(OrderCancelled(at=now, client_order_id=coid, reason=reason))
            elif before is None:
                self.on_order(OrderPlaced(at=now, order=order_view(t, now)))
            else:
                self.on_order(OrderModified(at=now, order=order_view(t, now)))

    def alert(self, severity: Severity, kind: str, message: str, **details: str) -> None:
        self.alerts.send(
            Alert(severity=severity, kind=kind, message=message, at=self.clock.now(), details=details)
        )

    async def initialize(self) -> None:
        """First start: deals before now are history, the current balance is the reference."""
        if self.state.cash_since is None:
            acct = await self.adapter.account()
            now = self.clock.now()
            self.state.cash_since = now
            self.state.deals_cursor = now
            self.state.expected_balance = acct.balance
            self.save()

    async def find_at_broker(self, coid: str) -> tuple[BrokerPosition | None, str | None]:
        """(open position, pending broker order id) carrying this client order id."""
        pos = next((p for p in await self.adapter.open_positions() if p.comment == coid), None)
        if pos is not None:
            return pos, None
        order = next((o for o in await self.adapter.pending_orders() if o.comment == coid), None)
        return None, order.broker_order_id if order else None

    # ------------------------------------------------------------ entries

    async def execute(self, intent: OrderIntent, decision: RiskDecision, timeframe: Timeframe) -> ExecResult:
        """Send one approved entry. Idempotent per intent."""
        now = self.clock.now()
        coid = client_order_id(intent.intent_id)
        if coid in self.state.orders:
            return ExecResult(coid, self.state.orders[coid].state, False, "duplicate intent")
        try:
            self.verifier.verify(decision, str(intent.intent_id), now)
        except DecisionRejectedError as e:
            if "signature" in str(e):
                self.alert(Severity.CRITICAL, "forged_decision", str(e), intent_id=str(intent.intent_id))
            return ExecResult(coid, None, False, str(e))
        self.save()  # persist the consumed sequence number before anything can reach the broker
        sig = intent.signal
        info = self.symbols.get(sig.symbol)
        lots = decision.approved_lots
        if info is None:
            return ExecResult(coid, None, False, f"unknown symbol {sig.symbol}")
        if not ZERO < lots <= intent.proposed_lots or (lots / info.lot_step) % 1 != 0 or lots < info.min_lot:
            return ExecResult(coid, None, False, f"approved lots {lots} not valid for this intent")
        requested: Decimal | None
        if sig.entry_type == "market":
            q = self.quotes.fresh(sig.symbol, now, self.cfg.max_quote_age_seconds)
            if q is None:
                self.alert(
                    Severity.WARNING, "stale_quotes", f"no fresh quote for {sig.symbol}; entry skipped"
                )
                return ExecResult(coid, None, False, "no fresh quote")
            requested = q.ask if sig.side == "buy" else q.bid
            price = None
        else:
            requested = price = round_price(sig.entry_price or 0.0, info.digits)
        expires_at = None
        if sig.entry_type != "market" and sig.expiry_bars:
            expires_at = sig.created_at + timedelta(minutes=sig.expiry_bars * timeframe.minutes)
        req = PlaceRequest(
            client_order_id=coid,
            symbol=sig.symbol,
            side=sig.side,
            order_type=sig.entry_type,
            lots=lots,
            price=price,
            sl=round_price(sig.stop_price, info.digits, stop_rounding(sig.side)),
            tp=round_price(sig.target_price, info.digits) if sig.target_price is not None else None,
            magic=magic_number(sig.strategy_id, sig.strategy_version),
            expires_at=expires_at,
        )
        t = TrackedOrder(
            client_order_id=coid,
            intent_id=intent.intent_id,
            account_id=intent.account_id,
            tenant_id=intent.tenant_id,
            strategy_id=sig.strategy_id,
            strategy_version=sig.strategy_version,
            symbol=sig.symbol,
            side=sig.side,
            order_type=sig.entry_type,
            lots=lots,
            requested_price=requested,
            sl=req.sl,
            tp=req.tp,
            magic=req.magic,
            state="sending",
            created_at=now,
            expires_at=expires_at,
        )
        self.state.orders[coid] = t
        self.save()  # write-ahead: a crash after this point is resolved by reconciliation
        return await self._send(t, req)

    async def _send(self, t: TrackedOrder, req: PlaceRequest) -> ExecResult:
        """Place with at most one retry; search the broker for the id before every attempt."""
        sent_at = self.clock.now()
        ack: BrokerAck | None = None
        for _attempt in range(2):
            try:
                pos, order_id = await self.find_at_broker(t.client_order_id)
                if pos is not None or order_id is not None:
                    await self._adopt(t, pos, order_id)
                    return ExecResult(t.client_order_id, t.state, True, "found at broker")
                ack = await self.adapter.place(req)
                break
            except BrokerUnavailableError as e:
                log.warning("place %s: %s", t.client_order_id, e)
        if ack is None:
            self.alert(
                Severity.WARNING,
                "order_unconfirmed",
                "broker did not answer; reconciliation will resolve the order",
                client_order_id=t.client_order_id,
            )
            return ExecResult(t.client_order_id, "sending", False, "broker unavailable")
        latency = (self.clock.now() - sent_at).total_seconds() * 1000 or ack.latency_ms
        if not ack.ok:
            t.state, t.note = "rejected", ack.error or "rejected by broker"
            self.save()
            return ExecResult(t.client_order_id, t.state, False, t.note)
        if t.order_type == "market":
            t.state, t.position_id = "open", ack.position_id
            if ack.filled_lots is not None:
                t.lots = ack.filled_lots
            self._opened(t, ack.filled_price or t.requested_price or ZERO, t.lots)
            self.save()
            self._record_fill(t, ack.filled_price or t.requested_price or ZERO, t.lots, latency, ZERO)
            await self.confirm_stop(t)
        else:
            t.state, t.broker_order_id = "pending", ack.broker_order_id
            self.save()
        return ExecResult(t.client_order_id, t.state, True)

    async def _adopt(self, t: TrackedOrder, pos: BrokerPosition | None, order_id: str | None) -> None:
        if pos is not None:
            t.state, t.position_id, t.lots = "open", pos.position_id, pos.lots
            self._opened(t, pos.price_open, pos.lots)
            self.save()
            self._record_fill(t, pos.price_open, pos.lots, 0.0, ZERO)
            await self.confirm_stop(t)
        elif order_id is not None:
            t.state, t.broker_order_id = "pending", order_id
            self.save()

    @staticmethod
    def _opened(t: TrackedOrder, price: Decimal, lots: Decimal) -> None:
        if t.entry_price is None:
            t.entry_price, t.opened_lots, t.initial_sl = price, lots, t.sl
            t.filled_at = t.filled_at or t.created_at

    def _closed_trade(self, t: TrackedOrder, exit_time: datetime) -> None:
        """A fully closed position becomes a Trade with its R multiple (spec section 5)."""
        if self.on_trade is None or t.entry_price is None or not t.out_lots:
            return
        info = self.symbols[t.symbol]
        stop = t.initial_sl if t.initial_sl is not None else t.sl
        lots = t.opened_lots or t.out_lots
        mar = abs(t.entry_price - stop) * info.contract_size * lots * self.to_account(t.symbol)
        if mar <= 0:
            self.alert(
                Severity.WARNING,
                "trade_without_risk",
                "closed trade with no money at risk",
                order=t.client_order_id,
            )
            return
        net = t.realized - t.costs
        self.on_trade(
            Trade(
                trade_id=t.client_order_id,
                account_id=t.account_id,
                tenant_id=t.tenant_id,
                strategy_id=t.strategy_id,
                strategy_version=t.strategy_version,
                symbol=t.symbol,
                side=t.side,
                lots=lots,
                entry_time=t.filled_at or t.created_at,
                entry_price=t.entry_price,
                stop_price=stop,
                exit_time=exit_time,
                exit_price=t.out_value / t.out_lots,
                pnl_gross=t.realized,
                costs=t.costs,
                pnl_net=net,
                money_at_risk=mar,
                r_multiple=float(net / mar),
                mae=0.0,  # not tracked on the live path yet
                mfe=0.0,
            )
        )

    def _record_fill(
        self, t: TrackedOrder, price: Decimal, lots: Decimal, latency_ms: float, commission: Decimal
    ) -> None:
        q = self.quotes.get(t.symbol)
        spread = q.ask - q.bid if q is not None else None
        slippage = None
        if t.requested_price is not None:
            slippage = price - t.requested_price if t.side == "buy" else t.requested_price - price
        now = self.clock.now()
        self.quality.record(
            ExecutionQuality(
                client_order_id=t.client_order_id,
                account_id=t.account_id,
                tenant_id=t.tenant_id,
                strategy_id=t.strategy_id,
                strategy_version=t.strategy_version,
                symbol=t.symbol,
                side=t.side,
                order_type=t.order_type,
                lots=lots,
                requested_price=t.requested_price,
                filled_price=price,
                spread_at_fill=spread,
                slippage=slippage,
                latency_ms=latency_ms,
                filled_at=now,
            )
        )
        if self.on_fill is not None:
            self.on_fill(
                Fill(
                    client_order_id=t.client_order_id,
                    account_id=t.account_id,
                    tenant_id=t.tenant_id,
                    symbol=t.symbol,
                    side=t.side,
                    price=price,
                    lots=lots,
                    commission=commission,
                    spread_at_fill=spread if spread is not None else ZERO,
                    requested_price=t.requested_price,
                    latency_ms=latency_ms,
                    filled_at=now,
                )
            )

    # ------------------------------------------------------------ stops

    async def confirm_stop(self, t: TrackedOrder) -> bool:
        """Make sure the broker position carries our stop. Returns True if protected (or already gone)."""
        if t.state != "open" or t.position_id is None:
            return True
        try:
            pos = await self._position(t.position_id)
            if pos is None:
                return True  # closed already; deals will record it
            if stop_protects(t.side, pos.sl, t.sl):
                t.stop_confirmed = True
                self.save()
                return True
            self.alert(
                Severity.CRITICAL,
                "missing_stop",
                "position without its stop after fill",
                position=t.position_id,
            )
            for _ in range(self.cfg.stop_confirm_attempts):
                try:
                    await self.adapter.modify(ModifyRequest(position_id=t.position_id, sl=t.sl, tp=t.tp))
                except BrokerUnavailableError as e:
                    log.warning("set stop %s: %s", t.position_id, e)
                pos = await self._position(t.position_id)
                if pos is None or stop_protects(t.side, pos.sl, t.sl):
                    t.stop_confirmed = True
                    self.save()
                    self.alert(Severity.INFO, "stop_repaired", "stop set after fill", position=t.position_id)
                    return True
            ack = await self.adapter.close_position(t.position_id)
        except BrokerUnavailableError as e:
            self.alert(
                Severity.CRITICAL, "missing_stop", f"cannot confirm stop: {e}", position=t.position_id or ""
            )
            return False
        if ack.ok:
            t.note = "closed: stop could not be set"
            self.alert(
                Severity.CRITICAL,
                "missing_stop",
                "stop could not be set; position closed",
                position=t.position_id,
            )
        else:
            self.alert(
                Severity.CRITICAL,
                "missing_stop",
                f"stop could not be set and close failed: {ack.error}",
                position=t.position_id,
            )
        self.save()
        return False

    async def _position(self, position_id: str) -> BrokerPosition | None:
        return next((p for p in await self.adapter.open_positions() if p.position_id == position_id), None)

    async def tighten_stop(self, position_id: str, new_stop: float) -> bool:
        """Move a stop toward price. Loosening is refused: it would add risk without a decision."""
        t = self.state.by_position(position_id)
        if t is None or t.state != "open":
            return False
        info = self.symbols[t.symbol]
        sl = round_price(new_stop, info.digits, stop_rounding(t.side))
        if not (sl > t.sl if t.side == "buy" else sl < t.sl):
            return False
        ack = await self.adapter.modify(ModifyRequest(position_id=position_id, sl=sl, tp=t.tp))
        if ack.ok:
            t.sl = sl
            self.save()
        return ack.ok

    async def close(self, position_id: str, reason: str) -> bool:
        ack = await self.adapter.close_position(position_id)
        t = self.state.by_position(position_id)
        if ack.ok and t is not None:
            t.note = reason
            self.save()
        return ack.ok

    # ------------------------------------------------------------ deals, expiry, halts

    async def sync_deals(self) -> int:
        """Apply new broker deals once each (duplicates and restarts are harmless). Returns new deals."""
        now = self.clock.now()
        cursor = self.state.deals_cursor or now
        deals = await self.adapter.deals(cursor - timedelta(seconds=self.cfg.deal_lookback_seconds))
        new = 0
        for d in sorted(deals, key=lambda d: (d.time, d.deal_id)):
            if d.deal_id in self.state.seen_deals:
                continue
            self.state.seen_deals[d.deal_id] = d.time
            new += 1
            if self.state.cash_since is not None and d.time >= self.state.cash_since:
                self.state.expected_balance = (self.state.expected_balance or ZERO) + (
                    d.profit + d.commission + d.swap
                )
            cursor = max(cursor, d.time)
            if d.kind == "balance":
                self.alert(
                    Severity.WARNING, "balance_operation", f"balance changed by {d.profit}", deal=d.deal_id
                )
                continue
            t = self.state.orders.get(d.comment) if is_system_comment(d.comment) else None
            t = t or self.state.by_position(d.position_id)
            if t is None:
                continue
            if d.entry == "in":
                t.costs -= d.commission  # broker commissions are negative
                if t.state in ("pending", "sending"):
                    t.state, t.position_id, t.lots = "open", d.position_id, d.lots
                    self._opened(t, d.price, d.lots)
                    self._record_fill(t, d.price, d.lots, 0.0, d.commission)
                    self.save()
                    await self.confirm_stop(t)
            elif d.entry == "out" and t.state == "open" and t.position_id == d.position_id:
                t.realized += d.profit
                t.costs -= d.commission + d.swap
                t.out_value += d.price * d.lots
                t.out_lots += d.lots
                t.lots -= d.lots
                if t.lots <= 0:
                    t.state, t.lots = "closed", ZERO
                    self._closed_trade(t, d.time)
        if new or cursor != self.state.deals_cursor:
            self.state.deals_cursor = cursor
            self.save()
        return new

    async def expire_pending(self) -> int:
        now = self.clock.now()
        n = 0
        for t in self.state.active():
            if (
                t.state == "pending"
                and t.expires_at is not None
                and t.expires_at <= now
                and t.broker_order_id
            ):
                ack = await self.adapter.cancel(t.broker_order_id)
                if ack.ok or not any(
                    o.broker_order_id == t.broker_order_id for o in await self.adapter.pending_orders()
                ):
                    t.state = "expired"
                    n += 1
        if n:
            self.save()
        return n

    async def cancel_pending_entries(self, reason: str) -> list[str]:
        """Cancel every pending entry order the system placed (as the broker sees them)."""
        cancelled = []
        for o in await self.adapter.pending_orders():
            if not is_system_comment(o.comment):
                continue
            ack = await self.adapter.cancel(o.broker_order_id)
            if ack.ok:
                cancelled.append(o.broker_order_id)
                t = self.state.orders.get(o.comment)
                if t is not None and t.state == "pending":
                    t.state, t.note = "cancelled", reason
        self.save()
        return cancelled

    async def apply_halt(self, cmd: HaltCommand) -> None:
        """Act on a risk gate halt: cancel pending entries; for daily/weekly/full halts close positions."""
        failed: list[str] = []
        if cmd.cancel_pending_entries:
            await self.cancel_pending_entries(f"halt {cmd.state.value}")
        if cmd.close_positions:
            for p in await self.adapter.open_positions():
                if not is_system_comment(p.comment):
                    # SPEC-QUESTION: external positions are not closed on a halt; the owner is alerted
                    self.alert(
                        Severity.CRITICAL,
                        "external_position_open_in_halt",
                        "external position left open during halt",
                        position=p.position_id,
                    )
                    continue
                try:
                    ack = await self.adapter.close_position(p.position_id)
                except BrokerUnavailableError as e:
                    ack = BrokerAck(ok=False, error=str(e))
                if not ack.ok:
                    failed.append(p.position_id)
        if failed:
            self.alert(
                Severity.CRITICAL,
                "halt_close_failed",
                "positions not closed on halt",
                positions=",".join(failed),
            )
        self.save()

    # ------------------------------------------------------------ lifecycle demotions

    def _of(self, strategy_id: str, version: str, state: OrderState) -> list[TrackedOrder]:
        return [
            t
            for t in self.state.active()
            if t.state == state and t.strategy_id == strategy_id and t.strategy_version == version
        ]

    async def cancel_pending_of(self, strategy_id: str, version: str, reason: str) -> None:
        """Demotion: cancel every pending entry of one version. Raises if any cancel fails."""
        failed = []
        for t in self._of(strategy_id, version, "pending"):
            ack = await self.adapter.cancel(t.broker_order_id or "")
            if ack.ok:
                t.state, t.note = "cancelled", f"demotion: {reason}"
            else:
                failed.append(t.client_order_id)
        self.save()
        if failed:
            raise RuntimeError(f"cancel failed for {failed}")

    async def close_positions_of(self, strategy_id: str, version: str, reason: str) -> None:
        """Demotion to shadow or retired: close the version's positions at market."""
        failed = []
        for t in self._of(strategy_id, version, "open"):
            if not await self.close(t.position_id or "", f"demotion: {reason}"):
                failed.append(t.client_order_id)
        if failed:
            raise RuntimeError(f"close failed for {failed}")

    def open_positions_of(self, strategy_id: str) -> list[TrackedOrder]:
        return [t for t in self.state.active() if t.state == "open" and t.strategy_id == strategy_id]
