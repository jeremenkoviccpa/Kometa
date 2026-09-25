"""In-memory hedging broker for tests, chaos tests and simulated paper runs.

Implements BrokerAdapter. Prices move only when the test calls `set_quote`; time comes from the
injected clock. Faults are injected explicitly, never at random, so every chaos test is reproducible:

- `fail(method, when, skip)`: a later call (after `skip` good ones) raises BrokerUnavailableError,
  either before the request is applied ("before") or after it was applied at the broker but the
  answer was lost ("after").
- `down = True`: every call fails (bridge killed).
- `ignore_sl_on_fill`: market fills come back without the stop that was attached.
- `modify_failures`: the next N modify calls are refused.
- `duplicate_deals`: `deals()` returns every deal twice.

Realism for simulated paper runs (off by default): `slippage(symbol, side, spread)` returns an adverse
price offset applied to market fills and stop-outs, `latency_ms()` the latency each answer reports.
Pass seeded functions to keep a run reproducible.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from autotrader.core.broker import (
    AccountInfo,
    BrokerAck,
    BrokerDeal,
    BrokerOrder,
    BrokerPosition,
    MarginMode,
    ModifyRequest,
    PlaceRequest,
    Quote,
    SymbolInfo,
    TradeMode,
)
from autotrader.core.clock import Clock
from autotrader.core.models import Bar, Side, Timeframe
from autotrader.execution.adapter import BrokerUnavailableError

When = Literal["before", "after"]
ZERO = Decimal(0)


class FakeBroker:
    def __init__(
        self,
        *,
        symbols: list[SymbolInfo],
        clock: Clock,
        balance: Decimal = Decimal(10000),
        currency: str = "USD",
        account_id: str = "fake-1",
        trade_mode: TradeMode = "demo",
        margin_mode: MarginMode = "hedging",
        commission_per_lot_side: Decimal = ZERO,
        to_account: Callable[[str], Decimal] | None = None,
        server_time_offset_s: float = 0.0,
        slippage: Callable[[str, Side, Decimal], Decimal] | None = None,
        latency_ms: Callable[[], float] | None = None,
    ) -> None:
        self.slippage = slippage
        self.latency_ms = latency_ms
        self.symbol_info = {s.symbol: s for s in symbols}
        self.clock = clock
        self.balance = balance
        self.currency = currency
        self.account_id = account_id
        self.trade_mode: TradeMode = trade_mode
        self.margin_mode: MarginMode = margin_mode
        self.commission = commission_per_lot_side
        self._fx = to_account or (lambda _symbol: Decimal(1))
        self.server_time_offset_s = server_time_offset_s
        self.quotes: dict[str, Quote] = {}
        self.positions: dict[str, BrokerPosition] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self.deal_log: list[BrokerDeal] = []
        self.calls: list[tuple[str, object]] = []
        # faults
        self.down = False
        self.ignore_sl_on_fill = False
        self.modify_failures = 0
        self.duplicate_deals = False
        self._faults: dict[str, deque[list[int | When]]] = defaultdict(deque)
        self._ticket = 1000
        self._subscribers: list[asyncio.Queue[Quote]] = []

    # ------------------------------------------------------------ test controls

    def fail(self, method: str, when: When = "before", skip: int = 0) -> None:
        self._faults[method].append([skip, when])

    def clear_faults(self) -> None:
        self._faults.clear()

    def set_quote(self, symbol: str, bid: Decimal | str, ask: Decimal | str) -> None:
        q = Quote(symbol=symbol, bid=Decimal(bid), ask=Decimal(ask), time=self.clock.now())
        self.quotes[symbol] = q
        for sub in self._subscribers:
            sub.put_nowait(q)
        self._expire_orders()
        self._trigger_orders(symbol)
        self._trigger_stops(symbol)

    def open_external(self, symbol: str, side: Side, lots: Decimal, sl: Decimal | None = None) -> str:
        """A manual trade from the phone app: magic 0, no system comment."""
        return self._open(symbol, side, lots, sl, None, magic=0, comment="")

    def deposit(self, amount: Decimal) -> None:
        self.balance += amount
        self.deal_log.append(
            BrokerDeal(
                deal_id=self._next(),
                kind="balance",
                position_id="",
                symbol="",
                side="buy",
                entry="in",
                lots=ZERO,
                price=ZERO,
                commission=ZERO,
                swap=ZERO,
                profit=amount,
                time=self.clock.now(),
                magic=0,
                comment="deposit",
            )
        )

    # ------------------------------------------------------------ fault plumbing

    def _enter(self, method: str, arg: object = None) -> When | None:
        self.calls.append((method, arg))
        if self.down:
            raise BrokerUnavailableError(f"{method}: bridge down")
        q = self._faults.get(method)
        when: When | None = None
        if q:
            head = q[0]
            if head[0]:
                head[0] = int(head[0]) - 1
            else:
                when = q.popleft()[1]  # type: ignore[assignment]
        if when == "before":
            raise BrokerUnavailableError(f"{method}: connection lost before the request")
        return when

    @staticmethod
    def _exit[T](when: When | None, method: str, result: T) -> T:
        if when == "after":
            raise BrokerUnavailableError(f"{method}: connection lost after the request was applied")
        return result

    # ------------------------------------------------------------ BrokerAdapter

    async def connect(self) -> None:
        self._enter("connect")

    async def account(self) -> AccountInfo:
        w = self._enter("account")
        floating = sum((p.profit for p in self._marked_positions()), ZERO)
        margin = sum(
            (p.lots * (self.symbol_info[p.symbol].margin_per_lot or ZERO) for p in self.positions.values()),
            ZERO,
        )
        equity = self.balance + floating
        info = AccountInfo(
            account_id=self.account_id,
            currency=self.currency,
            balance=self.balance,
            equity=equity,
            margin=margin,
            free_margin=equity - margin,
            server_time=self.clock.now() + timedelta(seconds=self.server_time_offset_s),
            trade_mode=self.trade_mode,
            margin_mode=self.margin_mode,
        )
        return self._exit(w, "account", info)

    async def symbols(self) -> list[SymbolInfo]:
        w = self._enter("symbols")
        return self._exit(w, "symbols", list(self.symbol_info.values()))

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        self._enter("stream_quotes", symbols)
        q: asyncio.Queue[Quote] = asyncio.Queue()
        self._subscribers.append(q)
        try:
            while True:
                quote = await q.get()
                if quote.symbol in symbols:
                    yield quote
        finally:
            self._subscribers.remove(q)

    async def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]:
        w = self._enter("history_bars", (symbol, tf))
        return self._exit(w, "history_bars", [])

    async def place(self, req: PlaceRequest) -> BrokerAck:
        w = self._enter("place", req)
        return self._exit(w, "place", self._place(req))

    async def modify(self, req: ModifyRequest) -> BrokerAck:
        w = self._enter("modify", req)
        if self.modify_failures > 0:
            self.modify_failures -= 1
            return self._exit(w, "modify", BrokerAck(ok=False, error="modify refused"))
        if req.position_id is not None:
            p = self.positions.get(req.position_id)
            if p is None:
                return self._exit(w, "modify", BrokerAck(ok=False, error="no such position"))
            if req.sl is not None and not self._stop_ok(p.side, self._exit_price(p), req.sl):
                return self._exit(w, "modify", BrokerAck(ok=False, error="invalid stops"))
            self.positions[p.position_id] = p.model_copy(
                update={"sl": req.sl if req.sl is not None else p.sl, "tp": req.tp if req.tp else p.tp}
            )
            return self._exit(w, "modify", BrokerAck(ok=True, position_id=p.position_id))
        o = self.orders.get(req.broker_order_id or "")
        if o is None:
            return self._exit(w, "modify", BrokerAck(ok=False, error="no such order"))
        self.orders[o.broker_order_id] = o.model_copy(
            update={"sl": req.sl if req.sl is not None else o.sl, "tp": req.tp if req.tp else o.tp}
        )
        return self._exit(w, "modify", BrokerAck(ok=True, broker_order_id=o.broker_order_id))

    async def cancel(self, broker_order_id: str) -> BrokerAck:
        w = self._enter("cancel", broker_order_id)
        if self.orders.pop(broker_order_id, None) is None:
            return self._exit(w, "cancel", BrokerAck(ok=False, error="no such order"))
        return self._exit(w, "cancel", BrokerAck(ok=True, broker_order_id=broker_order_id))

    async def close_position(self, position_id: str, lots: Decimal | None = None) -> BrokerAck:
        w = self._enter("close_position", position_id)
        p = self.positions.get(position_id)
        if p is None:
            return self._exit(w, "close_position", BrokerAck(ok=False, error="no such position"))
        price = self._exit_price(p)
        filled = self._close(p, price, lots)
        return self._exit(
            w,
            "close_position",
            BrokerAck(ok=True, position_id=position_id, filled_price=price, filled_lots=filled),
        )

    async def open_positions(self) -> list[BrokerPosition]:
        w = self._enter("open_positions")
        return self._exit(w, "open_positions", self._marked_positions())

    async def pending_orders(self) -> list[BrokerOrder]:
        w = self._enter("pending_orders")
        self._expire_orders()
        return self._exit(w, "pending_orders", list(self.orders.values()))

    async def deals(self, since: datetime) -> list[BrokerDeal]:
        w = self._enter("deals", since)
        out = [d for d in self.deal_log if d.time >= since]
        if self.duplicate_deals:
            out = [d for d in out for _ in range(2)]
        return self._exit(w, "deals", out)

    # ------------------------------------------------------------ internals

    def _next(self) -> str:
        self._ticket += 1
        return str(self._ticket)

    def _exit_price(self, p: BrokerPosition) -> Decimal:
        q = self.quotes[p.symbol]
        return q.bid if p.side == "buy" else q.ask

    def _slipped(self, symbol: str, side: Side, px: Decimal) -> Decimal:
        """A market execution `side` (the side of the trade being done) at px, made worse by slippage."""
        if self.slippage is None:
            return px
        q = self.quotes[symbol]
        s = max(self.slippage(symbol, side, q.ask - q.bid), Decimal(0))
        return px + s if side == "buy" else px - s

    @staticmethod
    def _stop_ok(side: Side, ref: Decimal, sl: Decimal) -> bool:
        return sl < ref if side == "buy" else sl > ref

    def _place(self, req: PlaceRequest) -> BrokerAck:
        info = self.symbol_info.get(req.symbol)
        q = self.quotes.get(req.symbol)
        if info is None or q is None:
            return BrokerAck(ok=False, error="unknown symbol or no price")
        if req.lots < info.min_lot or req.lots > info.max_lot or (req.lots / info.lot_step) % 1 != 0:
            return BrokerAck(ok=False, error="invalid volume")
        entry = q.ask if req.side == "buy" else q.bid
        if req.order_type == "market":
            if not self._stop_ok(req.side, entry, req.sl):
                return BrokerAck(ok=False, error="invalid stops")
            sl = None if self.ignore_sl_on_fill else req.sl
            entry = self._slipped(req.symbol, req.side, entry)
            pid = self._open(
                req.symbol, req.side, req.lots, sl, req.tp, req.magic, req.client_order_id, price=entry
            )
            return BrokerAck(
                ok=True,
                position_id=pid,
                filled_price=entry,
                filled_lots=req.lots,
                latency_ms=self.latency_ms() if self.latency_ms else 0.0,
            )
        if req.price is None:
            return BrokerAck(ok=False, error="pending order without price")
        below = req.price < entry
        wants_below = (req.order_type == "limit") == (req.side == "buy")
        if below != wants_below or not self._stop_ok(req.side, req.price, req.sl):
            return BrokerAck(ok=False, error="invalid price or stops")
        oid = self._next()
        self.orders[oid] = BrokerOrder(
            broker_order_id=oid,
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            lots=req.lots,
            price=req.price,
            sl=req.sl,
            tp=req.tp,
            magic=req.magic,
            comment=req.client_order_id,
            created_at=self.clock.now(),
            expires_at=req.expires_at,
        )
        return BrokerAck(ok=True, broker_order_id=oid)

    def _open(
        self,
        symbol: str,
        side: Side,
        lots: Decimal,
        sl: Decimal | None,
        tp: Decimal | None,
        magic: int,
        comment: str,
        price: Decimal | None = None,
    ) -> str:
        q = self.quotes[symbol]
        price = price if price is not None else (q.ask if side == "buy" else q.bid)
        pid = self._next()
        now = self.clock.now()
        self.positions[pid] = BrokerPosition(
            position_id=pid,
            symbol=symbol,
            side=side,
            lots=lots,
            price_open=price,
            sl=sl,
            tp=tp,
            magic=magic,
            comment=comment,
            opened_at=now,
        )
        self._deal(pid, symbol, side, "in", lots, price, ZERO, magic, comment)
        return pid

    def _close(self, p: BrokerPosition, price: Decimal, lots: Decimal | None) -> Decimal:
        lots = p.lots if lots is None or lots >= p.lots else lots
        info = self.symbol_info[p.symbol]
        sign = 1 if p.side == "buy" else -1
        profit = sign * (price - p.price_open) * info.contract_size * lots * self._fx(p.symbol)
        exit_side: Side = "sell" if p.side == "buy" else "buy"
        self._deal(p.position_id, p.symbol, exit_side, "out", lots, price, profit, p.magic, p.comment)
        if lots == p.lots:
            del self.positions[p.position_id]
        else:
            self.positions[p.position_id] = p.model_copy(update={"lots": p.lots - lots})
        return lots

    def _deal(
        self,
        pid: str,
        symbol: str,
        side: Side,
        entry: Literal["in", "out"],
        lots: Decimal,
        price: Decimal,
        profit: Decimal,
        magic: int,
        comment: str,
    ) -> None:
        commission = -self.commission * lots
        self.balance += profit + commission
        self.deal_log.append(
            BrokerDeal(
                deal_id=self._next(),
                position_id=pid,
                symbol=symbol,
                side=side,
                entry=entry,
                lots=lots,
                price=price,
                commission=commission,
                swap=ZERO,
                profit=profit,
                time=self.clock.now(),
                magic=magic,
                comment=comment,
            )
        )

    def _marked_positions(self) -> list[BrokerPosition]:
        out = []
        for p in self.positions.values():
            info = self.symbol_info[p.symbol]
            sign = 1 if p.side == "buy" else -1
            px = self._exit_price(p)
            pnl = sign * (px - p.price_open) * info.contract_size * p.lots * self._fx(p.symbol)
            out.append(p.model_copy(update={"profit": pnl}))
        return out

    def _expire_orders(self) -> None:
        now = self.clock.now()
        for oid in [o.broker_order_id for o in self.orders.values() if o.expires_at and o.expires_at <= now]:
            del self.orders[oid]

    def _trigger_orders(self, symbol: str) -> None:
        q = self.quotes[symbol]
        for o in [o for o in self.orders.values() if o.symbol == symbol]:
            px = q.ask if o.side == "buy" else q.bid
            if o.order_type == "limit":
                hit = px <= o.price if o.side == "buy" else px >= o.price
                fill = o.price
            else:
                hit = px >= o.price if o.side == "buy" else px <= o.price
                fill = px  # stop entries fill at market, gaps included
            if hit:
                del self.orders[o.broker_order_id]
                sl = None if self.ignore_sl_on_fill else o.sl
                self._open(o.symbol, o.side, o.lots, sl, o.tp, o.magic, o.comment, price=fill)

    def _trigger_stops(self, symbol: str) -> None:
        for p in [p for p in self.positions.values() if p.symbol == symbol]:
            px = self._exit_price(p)
            if p.side == "buy":
                hit = (p.sl is not None and px <= p.sl) or (p.tp is not None and px >= p.tp)
            else:
                hit = (p.sl is not None and px >= p.sl) or (p.tp is not None and px <= p.tp)
            if hit:
                closing: Side = "sell" if p.side == "buy" else "buy"
                stopped = p.sl is not None and (px <= p.sl if p.side == "buy" else px >= p.sl)
                self._close(p, self._slipped(symbol, closing, px) if stopped else px, None)
