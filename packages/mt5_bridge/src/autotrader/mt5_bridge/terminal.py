"""The MT5 terminal behind the bridge: maps the official `MetaTrader5` package to core.broker models.

Holds no strategy logic and no risk logic. The `mt5` module is injected, so the mapping is tested
against a fake module on any OS; on the Windows VPS it is the real `MetaTrader5` package.
The package is not thread safe: every call runs under one lock.

Constants are the documented MetaTrader5 values (checked against the package on the VPS, open
question 9).
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from autotrader.core.broker import (
    AccountInfo,
    BrokerAck,
    BrokerDeal,
    BrokerOrder,
    BrokerPosition,
    MarginMode,
    ModifyRequest,
    OrderType,
    PlaceRequest,
    Quote,
    SymbolInfo,
    TradeMode,
)
from autotrader.core.models import Bar, Side, Timeframe
from autotrader.mt5_bridge.servertime import ServerTimeZone

# MetaTrader5 constants
ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT, ORDER_TYPE_BUY_STOP, ORDER_TYPE_SELL_STOP = 2, 3, 4, 5
TRADE_ACTION_DEAL, TRADE_ACTION_PENDING, TRADE_ACTION_SLTP = 1, 5, 6
TRADE_ACTION_MODIFY, TRADE_ACTION_REMOVE = 7, 8
ORDER_TIME_GTC, ORDER_TIME_SPECIFIED = 0, 2
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2
TRADE_RETCODE_PLACED, TRADE_RETCODE_DONE, TRADE_RETCODE_DONE_PARTIAL = 10008, 10009, 10010
DEAL_TYPE_BUY, DEAL_TYPE_SELL = 0, 1
DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY = 0, 1, 2, 3
POSITION_TYPE_BUY = 0
SYMBOL_TRADE_MODE_DISABLED = 0
TIMEFRAMES = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 16385,
    Timeframe.H4: 16388,
    Timeframe.D1: 16408,
}
TRADE_MODES: dict[int, TradeMode] = {0: "demo", 1: "contest", 2: "real"}
MARGIN_MODES: dict[int, MarginMode] = {0: "netting", 1: "exchange", 2: "hedging"}
PENDING_TYPES: dict[tuple[OrderType, Side], int] = {
    ("limit", "buy"): ORDER_TYPE_BUY_LIMIT,
    ("limit", "sell"): ORDER_TYPE_SELL_LIMIT,
    ("stop", "buy"): ORDER_TYPE_BUY_STOP,
    ("stop", "sell"): ORDER_TYPE_SELL_STOP,
}
PENDING_BY_CODE = {v: k for k, v in PENDING_TYPES.items()}
OK_RETCODES = {TRADE_RETCODE_PLACED, TRADE_RETCODE_DONE, TRADE_RETCODE_DONE_PARTIAL}


class TerminalError(Exception):
    """The terminal is not connected or did not answer. Mapped to HTTP 503 by the app."""


def dec(x: float, places: int | None = None) -> Decimal:
    return Decimal(str(round(x, places) if places is not None else x))


class MT5Terminal:
    def __init__(
        self,
        mt5: Any,
        server_tz: ServerTimeZone,
        symbols: list[str],
        *,
        expect_trade_mode: TradeMode,
        deviation_points: int = 20,
    ) -> None:
        self.mt5 = mt5
        self.tz = server_tz
        self.symbol_names = list(symbols)
        self.expect_trade_mode: TradeMode = expect_trade_mode
        self.deviation = deviation_points
        self._lock = threading.Lock()

    # ------------------------------------------------------------ plumbing

    def _call(self, name: str, *args: Any, **kw: Any) -> Any:
        with self._lock:
            out = getattr(self.mt5, name)(*args, **kw)
            if out is None:
                raise TerminalError(f"{name} failed: {self.mt5.last_error()}")
            return out

    def connect(
        self, login: int | None = None, password: str | None = None, server: str | None = None
    ) -> None:
        kw: dict[str, Any] = {}
        if login is not None:
            kw = {"login": login, "password": password, "server": server}
        with self._lock:
            if not self.mt5.initialize(**kw):
                raise TerminalError(f"initialize failed: {self.mt5.last_error()}")
        for s in self.symbol_names:
            self._call("symbol_select", s, True)
        acct = self.account()
        if acct.trade_mode != self.expect_trade_mode:
            raise TerminalError(
                f"account is {acct.trade_mode}, bridge configured for {self.expect_trade_mode}"
            )

    def _digits(self, symbol: str) -> int:
        return int(self._call("symbol_info", symbol).digits)

    # ------------------------------------------------------------ reads

    def account(self) -> AccountInfo:
        a = self._call("account_info")
        return AccountInfo(
            account_id=str(a.login),
            currency=a.currency,
            balance=dec(a.balance, 2),
            equity=dec(a.equity, 2),
            margin=dec(a.margin, 2),
            free_margin=dec(a.margin_free, 2),
            server_time=self.server_now(),
            trade_mode=TRADE_MODES[a.trade_mode],
            margin_mode=MARGIN_MODES[a.margin_mode],
        )

    def server_now(self) -> datetime:
        """Broker server time = the newest tick time over the bridge's symbols (market must be open)."""
        times = [self._call("symbol_info_tick", s).time_msc / 1000 for s in self.symbol_names]
        return self.tz.to_utc(max(times))

    def symbols(self) -> list[SymbolInfo]:
        out = []
        for name in self.symbol_names:
            s = self._call("symbol_info", name)
            margin = self.mt5.order_calc_margin(ORDER_TYPE_BUY, name, 1.0, s.ask)
            out.append(
                SymbolInfo(
                    symbol=name,
                    digits=s.digits,
                    point=dec(s.point),
                    contract_size=dec(s.trade_contract_size),
                    min_lot=dec(s.volume_min),
                    lot_step=dec(s.volume_step),
                    max_lot=dec(s.volume_max),
                    margin_per_lot=dec(margin, 2) if margin is not None else None,
                    trade_allowed=s.trade_mode != SYMBOL_TRADE_MODE_DISABLED,
                )
            )
        return out

    def quotes(self, symbols: list[str]) -> list[Quote]:
        out = []
        for s in symbols:
            t = self._call("symbol_info_tick", s)
            d = self._digits(s)
            out.append(
                Quote(symbol=s, bid=dec(t.bid, d), ask=dec(t.ask, d), time=self.tz.to_utc(t.time_msc / 1000))
            )
        return out

    def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]:
        """Bid bars from MT5; ask = bid + the bar's spread in points (MT5 stores bid-based rates)."""
        info = self._call("symbol_info", symbol)
        rates = self._call(
            "copy_rates_range", symbol, TIMEFRAMES[tf], self.tz.to_server(start), self.tz.to_server(end)
        )
        point = float(info.point)
        span = timedelta(minutes=tf.minutes)
        bars = []
        for r in rates:
            opened = self.tz.to_utc(float(r["time"]))
            sp = float(r["spread"]) * point
            o, h, lo, c = (float(r[k]) for k in ("open", "high", "low", "close"))
            bars.append(
                Bar(
                    symbol=symbol,
                    timeframe=tf,
                    open_time=opened,
                    close_time=opened + span,
                    bid_o=o,
                    bid_h=h,
                    bid_l=lo,
                    bid_c=c,
                    ask_o=o + sp,
                    ask_h=h + sp,
                    ask_l=lo + sp,
                    ask_c=c + sp,
                    volume=float(r["tick_volume"]),
                )
            )
        return bars

    def open_positions(self) -> list[BrokerPosition]:
        out = []
        for p in self._call("positions_get"):
            d = self._digits(p.symbol)
            out.append(
                BrokerPosition(
                    position_id=str(p.ticket),
                    symbol=p.symbol,
                    side="buy" if p.type == POSITION_TYPE_BUY else "sell",
                    lots=dec(p.volume),
                    price_open=dec(p.price_open, d),
                    sl=dec(p.sl, d) if p.sl else None,
                    tp=dec(p.tp, d) if p.tp else None,
                    magic=p.magic,
                    comment=p.comment,
                    opened_at=self.tz.to_utc(p.time_msc / 1000),
                    profit=dec(p.profit, 2),
                )
            )
        return out

    def pending_orders(self) -> list[BrokerOrder]:
        out = []
        for o in self._call("orders_get"):
            if o.type not in PENDING_BY_CODE:
                continue
            kind, side = PENDING_BY_CODE[o.type]
            d = self._digits(o.symbol)
            out.append(
                BrokerOrder(
                    broker_order_id=str(o.ticket),
                    symbol=o.symbol,
                    side=side,
                    order_type=kind,
                    lots=dec(o.volume_current),
                    price=dec(o.price_open, d),
                    sl=dec(o.sl, d) if o.sl else None,
                    tp=dec(o.tp, d) if o.tp else None,
                    magic=o.magic,
                    comment=o.comment,
                    created_at=self.tz.to_utc(o.time_setup_msc / 1000),
                    expires_at=self.tz.to_utc(o.time_expiration) if o.time_expiration else None,
                )
            )
        return out

    def deals(self, since: datetime) -> list[BrokerDeal]:
        until = self.tz.to_server(datetime.now(since.tzinfo) + timedelta(days=1))
        out = []
        for d in self._call("history_deals_get", self.tz.to_server(since), until):
            trade = d.type in (DEAL_TYPE_BUY, DEAL_TYPE_SELL)
            entry: Literal["in", "out"] = "in" if d.entry == DEAL_ENTRY_IN else "out"
            out.append(
                BrokerDeal(
                    deal_id=str(d.ticket),
                    kind="trade" if trade else "balance",
                    position_id=str(d.position_id) if trade else "",
                    symbol=d.symbol if trade else "",
                    side="buy" if d.type == DEAL_TYPE_BUY else "sell",
                    entry=entry,
                    lots=dec(d.volume) if trade else Decimal(0),
                    price=dec(d.price),
                    commission=dec(d.commission, 2),
                    swap=dec(d.swap, 2),
                    profit=dec(d.profit, 2),
                    time=self.tz.to_utc(d.time_msc / 1000),
                    magic=d.magic,
                    comment=d.comment,
                )
            )
        return out

    # ------------------------------------------------------------ trading

    def _filling(self, symbol: str) -> int:
        mode = int(self._call("symbol_info", symbol).filling_mode)
        if mode & SYMBOL_FILLING_FOK:
            return ORDER_FILLING_FOK
        if mode & SYMBOL_FILLING_IOC:
            return ORDER_FILLING_IOC
        return ORDER_FILLING_RETURN

    def _send(self, request: dict[str, Any]) -> Any:
        return self._call("order_send", request)

    def place(self, req: PlaceRequest) -> BrokerAck:
        base: dict[str, Any] = {
            "symbol": req.symbol,
            "volume": float(req.lots),
            "sl": float(req.sl),
            "tp": float(req.tp) if req.tp is not None else 0.0,
            "magic": req.magic,
            "comment": req.client_order_id,
        }
        if req.order_type == "market":
            tick = self._call("symbol_info_tick", req.symbol)
            request = base | {
                "action": TRADE_ACTION_DEAL,
                "type": ORDER_TYPE_BUY if req.side == "buy" else ORDER_TYPE_SELL,
                "price": tick.ask if req.side == "buy" else tick.bid,
                "deviation": self.deviation,
                "type_time": ORDER_TIME_GTC,
                "type_filling": self._filling(req.symbol),
            }
        else:
            if req.price is None:
                return BrokerAck(ok=False, error="pending order without price")
            request = base | {
                "action": TRADE_ACTION_PENDING,
                "type": PENDING_TYPES[(req.order_type, req.side)],
                "price": float(req.price),
                "type_time": ORDER_TIME_SPECIFIED if req.expires_at else ORDER_TIME_GTC,
                "type_filling": ORDER_FILLING_RETURN,
            }
            if req.expires_at is not None:
                request["expiration"] = self.tz.to_server_seconds(req.expires_at)
        res = self._send(request)
        if res.retcode not in OK_RETCODES:
            return BrokerAck(ok=False, error=f"{res.retcode} {res.comment}")
        if req.order_type == "market":
            # hedging accounts: the position ticket is the ticket of the order that opened it
            return BrokerAck(
                ok=True,
                broker_order_id=str(res.order),
                position_id=str(res.order),
                filled_price=dec(res.price, self._digits(req.symbol)),
                filled_lots=dec(res.volume),
            )
        return BrokerAck(ok=True, broker_order_id=str(res.order))

    def modify(self, req: ModifyRequest) -> BrokerAck:
        if req.position_id is not None:
            [p] = self._call("positions_get", ticket=int(req.position_id)) or [None]
            if p is None:
                return BrokerAck(ok=False, error="no such position")
            request = {
                "action": TRADE_ACTION_SLTP,
                "position": p.ticket,
                "symbol": p.symbol,
                "sl": float(req.sl) if req.sl is not None else p.sl,
                "tp": float(req.tp) if req.tp is not None else p.tp,
            }
        else:
            [o] = self._call("orders_get", ticket=int(req.broker_order_id or 0)) or [None]
            if o is None:
                return BrokerAck(ok=False, error="no such order")
            request = {
                "action": TRADE_ACTION_MODIFY,
                "order": o.ticket,
                "price": o.price_open,
                "sl": float(req.sl) if req.sl is not None else o.sl,
                "tp": float(req.tp) if req.tp is not None else o.tp,
            }
        res = self._send(request)
        ok = res.retcode in OK_RETCODES
        return BrokerAck(
            ok=ok, position_id=req.position_id, error=None if ok else f"{res.retcode} {res.comment}"
        )

    def cancel(self, broker_order_id: str) -> BrokerAck:
        res = self._send({"action": TRADE_ACTION_REMOVE, "order": int(broker_order_id)})
        ok = res.retcode in OK_RETCODES
        return BrokerAck(
            ok=ok, broker_order_id=broker_order_id, error=None if ok else f"{res.retcode} {res.comment}"
        )

    def close_position(self, position_id: str, lots: Decimal | None = None) -> BrokerAck:
        [p] = self._call("positions_get", ticket=int(position_id)) or [None]
        if p is None:
            return BrokerAck(ok=False, error="no such position")
        tick = self._call("symbol_info_tick", p.symbol)
        buy = p.type == POSITION_TYPE_BUY
        volume = float(lots) if lots is not None and float(lots) < p.volume else p.volume
        res = self._send(
            {
                "action": TRADE_ACTION_DEAL,
                "position": p.ticket,
                "symbol": p.symbol,
                "volume": volume,
                "type": ORDER_TYPE_SELL if buy else ORDER_TYPE_BUY,
                "price": tick.bid if buy else tick.ask,
                "deviation": self.deviation,
                "magic": p.magic,
                "comment": p.comment,
                "type_time": ORDER_TIME_GTC,
                "type_filling": self._filling(p.symbol),
            }
        )
        if res.retcode not in OK_RETCODES:
            return BrokerAck(ok=False, error=f"{res.retcode} {res.comment}")
        return BrokerAck(
            ok=True, position_id=position_id, filled_price=dec(res.price), filled_lots=dec(res.volume)
        )
