"""CTraderAdapter: BrokerAdapter over the cTrader Open API (JSON over WebSocket).

For any cTrader broker's account (e.g. an IC Markets cTrader demo), from any machine: no MT5, no Windows.

Protocol (spotware/openapi-proto-messages, help.ctrader.com/open-api): messages are
{"clientMsgId", "payloadType", "payload"}; the JSON port is 5036 on demo.ctraderapi.com / live.ctraderapi.com.
Authentication: application (client id and secret) then account (ctidTraderAccountId and an access token).
A heartbeat goes out every 10 s. Replies and the execution events a request causes carry its clientMsgId.

Mapping to Kometa's broker model (which follows MT5):
- Volumes are in hundredths of a unit: volume = lots x lotSize (lotSize from the broker's symbol, also in
  hundredths, so contract size = lotSize / 100). Spot prices are integers in 1/100,000.
- Our client order id goes into the order's comment and clientOrderId; the resulting position carries the
  comment, so the order manager finds its orders and positions after a crash, exactly as with MT5. The
  strategy's magic number rides in the label ("m<magic>").
- Market orders take a relative stop (a distance), not a price; the order manager confirms every stop after
  the fill and moves it to the exact level if slippage left it looser.
- One position per order: the account must be HEDGED, else it reports "netting" and execution refuses it.
- Deals: a deal with closePositionDetail is an exit (gross profit, swap, commission); others are entries,
  mapped to our client order id through the order they filled.

Transport failures and timeouts become BrokerUnavailableError; an invalid or expired token raises
PermissionError. History bars are bid-only (cTrader trendbars): ask = bid + the current spread, fine for
warm-up; tuning uses bid/ask data (Dukascopy).
SPEC-QUESTION: token refresh (access tokens last about 30 days) is manual for now (docs/open_questions.md 33).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, Protocol

from autotrader.core.broker import (
    AccountInfo,
    BrokerAck,
    BrokerDeal,
    BrokerOrder,
    BrokerPosition,
    ModifyRequest,
    PlaceRequest,
    Quote,
    SymbolInfo,
)
from autotrader.core.models import Bar, Timeframe
from autotrader.execution.adapter import BrokerUnavailableError, register_adapter

HOSTS = {
    "demo": ("wss://demo.ctraderapi.com:5036", "demo"),
    "live": ("wss://live.ctraderapi.com:5036", "real"),
}
PRICE = Decimal(100_000)  # spot and relative prices are integers in 1/100,000

# payload types (ProtoOAPayloadType, ProtoPayloadType)
ERROR, HEARTBEAT = 50, 51
APP_AUTH, APP_AUTH_RES, ACC_AUTH, ACC_AUTH_RES = 2100, 2101, 2102, 2103
NEW_ORDER, CANCEL_ORDER, AMEND_ORDER, AMEND_SLTP, CLOSE_POSITION = 2106, 2108, 2109, 2110, 2111
SYMBOLS_LIST, SYMBOLS_LIST_RES, SYMBOL_BY_ID, SYMBOL_BY_ID_RES = 2114, 2115, 2116, 2117
ASSET_LIST, ASSET_LIST_RES = 2112, 2113
ACCOUNTS_BY_TOKEN, ACCOUNTS_BY_TOKEN_RES = 2149, 2150
TRADER, TRADER_RES, RECONCILE, RECONCILE_RES, EXECUTION = 2121, 2122, 2124, 2125, 2126
SUB_SPOTS, SUB_SPOTS_RES, SPOT = 2127, 2128, 2131
ORDER_ERROR, DEAL_LIST, DEAL_LIST_RES, TRENDBARS, TRENDBARS_RES, OA_ERROR = 2132, 2133, 2134, 2137, 2138, 2142
TOKEN_INVALIDATED, CLIENT_DISCONNECT, ORDER_DETAILS, ORDER_DETAILS_RES = 2147, 2148, 2181, 2182
PNL, PNL_RES, ACCOUNT_DISCONNECT = 2187, 2188, 2164

MARKET, LIMIT, STOP = 1, 2, 3
BUY, SELL = 1, 2
ACCEPTED, FILLED, REPLACED, CANCELLED, EXPIRED, REJECTED, CANCEL_REJECTED, PARTIAL = 2, 3, 4, 5, 6, 7, 8, 11
GOOD_TILL_DATE, GOOD_TILL_CANCEL = 1, 2
HEDGED = 0
M1_PERIOD = 1

AUTH_ERRORS = ("CH_ACCESS_TOKEN_INVALID", "CH_CLIENT_AUTH_FAILURE", "OA_AUTH_TOKEN_EXPIRED", "INVALID_TOKEN")
ENUMS = {  # JSON may carry enums as numbers or names
    "BUY": BUY, "SELL": SELL, "MARKET": MARKET, "LIMIT": LIMIT, "STOP": STOP, "HEDGED": HEDGED,
    "NETTED": 1, "ORDER_ACCEPTED": ACCEPTED, "ORDER_FILLED": FILLED, "ORDER_REPLACED": REPLACED,
    "ORDER_CANCELLED": CANCELLED, "ORDER_EXPIRED": EXPIRED, "ORDER_REJECTED": REJECTED,
    "ORDER_CANCEL_REJECTED": CANCEL_REJECTED, "ORDER_PARTIAL_FILL": PARTIAL,
}  # fmt: skip


def _enum(v: Any) -> int:
    return ENUMS.get(v, 0) if isinstance(v, str) else int(v)


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def _from_ms(ms: Any) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC)


def _magic(label: Any) -> int:
    s = str(label or "")
    return int(s[1:]) if s.startswith("m") and s[1:].isdigit() else 0


Msg = dict[str, Any]
_Waiter = tuple[Callable[[Msg], bool], "asyncio.Future[list[Msg]]", list[Msg]]


class Transport(Protocol):
    async def send(self, text: str) -> None: ...

    async def recv(self) -> str: ...

    async def close(self) -> None: ...


class _WebSocket:
    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def send(self, text: str) -> None:
        await self._ws.send(text)

    async def recv(self) -> str:
        msg = await self._ws.recv()
        return msg if isinstance(msg, str) else msg.decode()

    async def close(self) -> None:
        await self._ws.close()


async def _websocket(url: str) -> Transport:
    import websockets  # noqa: PLC0415

    return _WebSocket(await websockets.connect(url, open_timeout=15, ping_interval=None))


class _Sym:
    def __init__(self, name: str, symbol_id: int, details: Mapping[str, Any]) -> None:
        self.name, self.id = name, symbol_id
        self.digits = int(details["digits"])
        self.lot_size = int(details.get("lotSize") or 10_000_000)  # hundredths of a unit
        self.min_volume = int(details.get("minVolume") or 1)
        self.step_volume = int(details.get("stepVolume") or 1)
        self.max_volume = int(details.get("maxVolume") or 10**12)

    def lots(self, volume: Any) -> Decimal:
        return Decimal(int(volume)) / self.lot_size

    def volume(self, lots: Decimal) -> int:
        return int((lots * self.lot_size).to_integral_value())

    def price(self, raw: Any) -> Decimal:
        return (Decimal(int(raw)) / PRICE).quantize(Decimal(1).scaleb(-self.digits))

    def px(self, x: Any) -> Decimal:
        return Decimal(str(x)).quantize(Decimal(1).scaleb(-self.digits))


class CTraderAdapter:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: int,
        symbols: list[str],
        *,
        environment: str = "demo",
        timeout_s: float = 10.0,
        heartbeat_s: float = 10.0,
        connect: Callable[[str], Awaitable[Transport]] | None = None,
    ) -> None:
        if environment not in HOSTS:
            raise ValueError(f"environment must be one of {sorted(HOSTS)}")
        url, mode = HOSTS[environment]
        self.url = url
        self.trade_mode: Literal["demo", "real"] = "demo" if mode == "demo" else "real"
        self._creds = (client_id, client_secret, access_token)
        self.account_id = int(account_id)
        self.wanted = list(symbols)
        self.timeout_s, self.heartbeat_s = timeout_s, heartbeat_s
        self._connect_fn = connect or _websocket
        self._t: Transport | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._waiters: dict[str, _Waiter] = {}
        self._syms: dict[str, _Sym] = {}
        self._by_id: dict[int, _Sym] = {}
        self._quotes: dict[str, Quote] = {}
        self._quote_q: asyncio.Queue[Quote] = asyncio.Queue(maxsize=10_000)
        self._order_comment: dict[int, str] = {}  # orderId -> our client order id
        self._lock = asyncio.Lock()
        self._fatal: Exception | None = None

    # ------------------------------------------------------------ session

    async def connect(self) -> None:
        async with self._lock:
            if self._t is not None and self._fatal is None:
                return
            await self._close()
            self._fatal = None
            try:
                self._t = await self._connect_fn(self.url)
            except OSError as e:
                raise BrokerUnavailableError(f"connect {self.url}: {e}") from e
            self._tasks = [asyncio.create_task(self._reader()), asyncio.create_task(self._heartbeat())]
            cid, secret, token = self._creds
            await self._call(APP_AUTH, {"clientId": cid, "clientSecret": secret}, {APP_AUTH_RES})
            await self._call(
                ACC_AUTH, {"ctidTraderAccountId": self.account_id, "accessToken": token}, {ACC_AUTH_RES}
            )
            await self._load_symbols()
            ids = [self._syms[s].id for s in self.wanted if s in self._syms]
            if ids:
                await self._call(
                    SUB_SPOTS,
                    {
                        "ctidTraderAccountId": self.account_id,
                        "symbolId": ids,
                        "subscribeToSpotTimestamp": True,
                    },
                    {SUB_SPOTS_RES},
                )

    async def aclose(self) -> None:
        await self._close()

    async def _close(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._tasks = []
        if self._t is not None:
            with contextlib.suppress(Exception):
                await self._t.close()
        self._t = None

    async def _ensure(self) -> None:
        if isinstance(self._fatal, PermissionError):
            raise self._fatal
        if self._t is None or self._fatal is not None:
            await self.connect()

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_s)
            if self._t is not None:
                with contextlib.suppress(Exception):
                    await self._t.send(json.dumps({"payloadType": HEARTBEAT, "payload": {}}))

    async def _reader(self) -> None:
        try:
            while self._t is not None:
                msg = json.loads(await self._t.recv())
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # the connection dropped: every waiter fails, the next call reconnects
            self._fatal = self._fatal or BrokerUnavailableError(
                f"cTrader connection lost: {type(e).__name__}"
            )
            for _pred, fut, _got in list(self._waiters.values()):
                if not fut.done():
                    fut.set_exception(self._fatal)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        pt, payload = int(msg.get("payloadType", 0)), msg.get("payload") or {}
        if pt == SPOT:
            self._on_spot(payload)
            return
        if pt in (TOKEN_INVALIDATED, CLIENT_DISCONNECT, ACCOUNT_DISCONNECT):
            err: Exception = (
                PermissionError("cTrader access token invalidated")
                if pt == TOKEN_INVALIDATED
                else BrokerUnavailableError(f"cTrader disconnected ({pt})")
            )
            self._fatal = err
            for _pred, fut, _got in list(self._waiters.values()):
                if not fut.done():
                    fut.set_exception(err)
            return
        waiter = self._waiters.get(str(msg.get("clientMsgId", "")))
        if waiter is None:
            return  # unsolicited events (e.g. server-side fills): state comes from reconcile and deals
        pred, fut, got = waiter
        got.append(msg)
        if fut.done():
            return
        if pt in (ERROR, OA_ERROR):
            code = str(payload.get("errorCode", ""))
            desc = payload.get("description", "")
            exc: Exception = (
                PermissionError(f"cTrader refused the credentials: {code}")
                if code in AUTH_ERRORS
                else _BrokerError(f"{code}: {desc}".strip(": "))
            )
            fut.set_exception(exc)
        elif pt == ORDER_ERROR or pred(msg):
            fut.set_result(got)

    async def _call(
        self,
        payload_type: int,
        payload: dict[str, Any],
        done: set[int] | Callable[[dict[str, Any]], bool],
    ) -> list[dict[str, Any]]:
        """Send one request; collect replies carrying its clientMsgId until `done` says it is complete."""
        if self._t is None:
            raise BrokerUnavailableError("not connected")
        pred = done if callable(done) else (lambda m, s=done: int(m.get("payloadType", 0)) in s)
        cid = uuid.uuid4().hex[:16]
        fut: asyncio.Future[list[dict[str, Any]]] = asyncio.get_running_loop().create_future()
        self._waiters[cid] = (pred, fut, [])
        try:
            await self._t.send(
                json.dumps({"clientMsgId": cid, "payloadType": payload_type, "payload": payload})
            )
            return await asyncio.wait_for(fut, self.timeout_s)
        except TimeoutError as e:
            raise BrokerUnavailableError(
                f"cTrader did not answer {payload_type} in {self.timeout_s:.0f} s"
            ) from e
        except (PermissionError, BrokerUnavailableError):
            raise  # both are OSErrors: a rejected token must never turn into "try again later"
        except OSError as e:
            self._fatal = BrokerUnavailableError(f"cTrader send failed: {e}")
            raise self._fatal from e
        finally:
            self._waiters.pop(cid, None)

    async def _req(self, payload_type: int, payload: dict[str, Any], done: Any) -> list[dict[str, Any]]:
        await self._ensure()
        return await self._call(payload_type, {"ctidTraderAccountId": self.account_id, **payload}, done)

    async def _load_symbols(self) -> None:
        light = await self._call(SYMBOLS_LIST, {"ctidTraderAccountId": self.account_id}, {SYMBOLS_LIST_RES})
        names = {
            s["symbolName"]: int(s["symbolId"])
            for s in light[-1]["payload"].get("symbol", [])
            if s.get("symbolName")
        }
        missing = [s for s in self.wanted if s not in names]
        if missing:
            raise BrokerUnavailableError(f"the account has no symbol {missing}")
        ids = [names[s] for s in self.wanted]
        full = await self._call(
            SYMBOL_BY_ID, {"ctidTraderAccountId": self.account_id, "symbolId": ids}, {SYMBOL_BY_ID_RES}
        )
        details = {int(d["symbolId"]): d for d in full[-1]["payload"].get("symbol", [])}
        for name in self.wanted:
            sym = _Sym(name, names[name], details[names[name]])
            self._syms[name], self._by_id[sym.id] = sym, sym

    def _on_spot(self, p: Mapping[str, Any]) -> None:
        sym = self._by_id.get(int(p.get("symbolId", 0)))
        if sym is None:
            return
        prev = self._quotes.get(sym.name)
        bid = sym.price(p["bid"]) if p.get("bid") else (prev.bid if prev else None)
        ask = sym.price(p["ask"]) if p.get("ask") else (prev.ask if prev else None)
        if bid is None or ask is None:
            return
        t = _from_ms(p["timestamp"]) if p.get("timestamp") else datetime.now(UTC)
        q = Quote(symbol=sym.name, bid=bid, ask=ask, time=t)
        self._quotes[sym.name] = q
        with contextlib.suppress(asyncio.QueueFull):
            self._quote_q.put_nowait(q)

    # ------------------------------------------------------------ account and market

    async def account(self) -> AccountInfo:
        tr = (await self._req(TRADER, {}, {TRADER_RES}))[-1]["payload"]["trader"]
        md = int(tr.get("moneyDigits", 2))
        balance = Decimal(int(tr["balance"])).scaleb(-md)
        rec = (await self._req(RECONCILE, {}, {RECONCILE_RES}))[-1]["payload"]
        margin = sum(
            (
                Decimal(int(p.get("usedMargin", 0))).scaleb(-int(p.get("moneyDigits", md)))
                for p in rec.get("position", [])
            ),
            Decimal(0),
        )
        unrealized = Decimal(0)
        if rec.get("position"):
            pnl = (await self._req(PNL, {}, {PNL_RES}))[-1]["payload"]
            pmd = int(pnl.get("moneyDigits", md))
            unrealized = sum(
                (
                    Decimal(int(x["netUnrealizedPnL"])).scaleb(-pmd)
                    for x in pnl.get("positionUnrealizedPnL", [])
                ),
                Decimal(0),
            )
        equity = balance + unrealized
        return AccountInfo(
            account_id=str(self.account_id),
            currency=await self._currency(int(tr["depositAssetId"])),
            balance=balance,
            equity=equity,
            margin=margin,
            free_margin=equity - margin,
            server_time=datetime.now(UTC),
            trade_mode=self.trade_mode,
            margin_mode="hedging" if _enum(tr.get("accountType", HEDGED)) == HEDGED else "netting",
        )

    async def _currency(self, asset_id: int) -> str:
        if not hasattr(self, "_assets"):
            res = await self._req(ASSET_LIST, {}, {ASSET_LIST_RES})
            self._assets = {
                int(a["assetId"]): str(a.get("displayName") or a["name"])
                for a in res[-1]["payload"].get("asset", [])
            }
        return self._assets.get(asset_id, "USD")

    async def symbols(self) -> list[SymbolInfo]:
        await self._ensure()
        out = []
        for s in self._syms.values():
            out.append(
                SymbolInfo(
                    symbol=s.name,
                    digits=s.digits,
                    point=Decimal(1).scaleb(-s.digits),
                    contract_size=Decimal(s.lot_size) / 100,
                    min_lot=s.lots(s.min_volume),
                    lot_step=s.lots(s.step_volume),
                    max_lot=s.lots(s.max_volume),
                )
            )
        return out

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        await self._ensure()
        for s in symbols:
            if s in self._quotes:
                yield self._quotes[s]
        while True:
            q = await self._quote_q.get()
            if q.symbol in symbols:
                yield q

    async def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]:
        """M1 bid bars (cTrader trendbars); ask = bid + the latest spread. Paged in 3-day windows."""
        await self._ensure()
        if tf != Timeframe.M1:
            raise ValueError("history is fetched as M1 and aggregated locally")
        sym = self._syms[symbol]
        q = self._quotes.get(symbol)
        spread = float(q.ask - q.bid) if q else 0.0
        out: list[Bar] = []
        cursor = start
        while cursor < end:
            upto = min(cursor + timedelta(days=3), end)
            res = await self._req(
                TRENDBARS,
                {
                    "fromTimestamp": _ms(cursor),
                    "toTimestamp": _ms(upto),
                    "period": M1_PERIOD,
                    "symbolId": sym.id,
                },
                {TRENDBARS_RES},
            )
            for tb in res[-1]["payload"].get("trendbar", []):
                t = datetime.fromtimestamp(int(tb["utcTimestampInMinutes"]) * 60, tz=UTC)
                if not (cursor <= t < upto):
                    continue
                low = int(tb["low"])
                o, h, c = (low + int(tb.get(k, 0)) for k in ("deltaOpen", "deltaHigh", "deltaClose"))
                bo, bh, bl, bc = (float(Decimal(v) / PRICE) for v in (o, h, low, c))
                out.append(
                    Bar(
                        symbol=symbol,
                        timeframe=tf,
                        open_time=t,
                        close_time=t + timedelta(minutes=1),
                        bid_o=bo,
                        bid_h=bh,
                        bid_l=bl,
                        bid_c=bc,
                        ask_o=bo + spread,
                        ask_h=bh + spread,
                        ask_l=bl + spread,
                        ask_c=bc + spread,
                        volume=float(tb.get("volume", 0)),
                    )
                )
            cursor = upto
        out.sort(key=lambda b: b.open_time)
        return out

    # ------------------------------------------------------------ orders and positions

    async def place(self, req: PlaceRequest) -> BrokerAck:
        await self._ensure()
        sym = self._syms[req.symbol]
        body: dict[str, Any] = {
            "symbolId": sym.id,
            "tradeSide": BUY if req.side == "buy" else SELL,
            "volume": sym.volume(req.lots),
            "comment": req.client_order_id,
            "clientOrderId": req.client_order_id,
            "label": f"m{req.magic}",
        }
        if req.order_type == "market":
            q = self._quotes.get(req.symbol)
            if q is None:
                return BrokerAck(ok=False, error="no price yet")
            ref = q.ask if req.side == "buy" else q.bid
            body |= {"orderType": MARKET, "relativeStopLoss": int(abs(ref - req.sl) * PRICE)}
            if req.tp is not None:
                body["relativeTakeProfit"] = int(abs(req.tp - ref) * PRICE)
            finished = {FILLED, REJECTED, CANCELLED, EXPIRED}
        else:
            if req.price is None:
                return BrokerAck(ok=False, error="pending order without price")
            key = "limitPrice" if req.order_type == "limit" else "stopPrice"
            body |= {"orderType": LIMIT if req.order_type == "limit" else STOP, key: float(req.price)}
            body |= {"stopLoss": float(req.sl)}
            if req.tp is not None:
                body["takeProfit"] = float(req.tp)
            if req.expires_at is not None:
                body |= {"timeInForce": GOOD_TILL_DATE, "expirationTimestamp": _ms(req.expires_at)}
            else:
                body["timeInForce"] = GOOD_TILL_CANCEL
            finished = {ACCEPTED, REJECTED, CANCELLED}
        t0 = time.perf_counter()
        try:
            msgs = await self._req(
                NEW_ORDER, body, lambda m: m.get("payloadType") == EXECUTION and _exec_type(m) in finished
            )
        except _BrokerError as e:
            return BrokerAck(ok=False, error=str(e))
        latency = (time.perf_counter() - t0) * 1000
        last = msgs[-1]
        if last.get("payloadType") == ORDER_ERROR:
            return BrokerAck(ok=False, error=str(last["payload"].get("errorCode")), latency_ms=latency)
        p = last["payload"]
        et = _exec_type(last)
        order = p.get("order") or {}
        if order.get("orderId"):
            self._order_comment[int(order["orderId"])] = req.client_order_id
        if et in (REJECTED, CANCELLED, EXPIRED):
            return BrokerAck(ok=False, error=str(p.get("errorCode") or "rejected"), latency_ms=latency)
        if et == FILLED:
            pos, deal = p.get("position") or {}, p.get("deal") or {}
            return BrokerAck(
                ok=True,
                position_id=str(pos.get("positionId") or deal.get("positionId")),
                filled_price=sym.px(deal.get("executionPrice") or pos.get("price")),
                filled_lots=sym.lots(deal.get("filledVolume") or pos.get("tradeData", {}).get("volume", 0)),
                latency_ms=latency,
            )
        return BrokerAck(ok=True, broker_order_id=str(order.get("orderId")), latency_ms=latency)

    async def modify(self, req: ModifyRequest) -> BrokerAck:
        await self._ensure()
        done = lambda m: m.get("payloadType") == EXECUTION  # noqa: E731
        try:
            if req.position_id is not None:
                body: dict[str, Any] = {"positionId": int(req.position_id)}
                if req.sl is not None:
                    body["stopLoss"] = float(req.sl)
                if req.tp is not None:
                    body["takeProfit"] = float(req.tp)
                msgs = await self._req(AMEND_SLTP, body, done)
                ok = msgs[-1].get("payloadType") == EXECUTION
                return BrokerAck(ok=ok, position_id=req.position_id, error=None if ok else _err(msgs[-1]))
            if req.broker_order_id is None:
                return BrokerAck(ok=False, error="nothing to modify")
            body = {"orderId": int(req.broker_order_id)}
            if req.sl is not None:
                body["stopLoss"] = float(req.sl)
            if req.tp is not None:
                body["takeProfit"] = float(req.tp)
            msgs = await self._req(AMEND_ORDER, body, done)
        except _BrokerError as e:
            return BrokerAck(ok=False, error=str(e))
        ok = msgs[-1].get("payloadType") == EXECUTION
        return BrokerAck(ok=ok, broker_order_id=req.broker_order_id, error=None if ok else _err(msgs[-1]))

    async def cancel(self, broker_order_id: str) -> BrokerAck:
        try:
            msgs = await self._req(
                CANCEL_ORDER,
                {"orderId": int(broker_order_id)},
                lambda m: m.get("payloadType") == EXECUTION and _exec_type(m) in (CANCELLED, CANCEL_REJECTED),
            )
        except _BrokerError as e:
            return BrokerAck(ok=False, error=str(e))
        ok = msgs[-1].get("payloadType") == EXECUTION and _exec_type(msgs[-1]) == CANCELLED
        return BrokerAck(ok=ok, broker_order_id=broker_order_id, error=None if ok else _err(msgs[-1]))

    async def close_position(self, position_id: str, lots: Decimal | None = None) -> BrokerAck:
        pos = next(
            (p for p in await self._reconcile_positions() if str(p["positionId"]) == position_id), None
        )
        if pos is None:
            return BrokerAck(ok=False, error="no such position")
        sym = self._by_id[int(pos["tradeData"]["symbolId"])]
        volume = int(pos["tradeData"]["volume"]) if lots is None else sym.volume(lots)
        try:
            msgs = await self._req(
                CLOSE_POSITION,
                {"positionId": int(position_id), "volume": volume},
                lambda m: m.get("payloadType") == EXECUTION and _exec_type(m) in (FILLED, REJECTED),
            )
        except _BrokerError as e:
            return BrokerAck(ok=False, error=str(e))
        last = msgs[-1]
        if last.get("payloadType") != EXECUTION or _exec_type(last) != FILLED:
            return BrokerAck(ok=False, error=_err(last))
        deal = last["payload"].get("deal") or {}
        return BrokerAck(
            ok=True,
            position_id=position_id,
            filled_price=sym.px(deal["executionPrice"]) if deal.get("executionPrice") else None,
            filled_lots=sym.lots(deal.get("filledVolume", volume)),
        )

    async def _reconcile_positions(self) -> list[dict[str, Any]]:
        rec = (await self._req(RECONCILE, {}, {RECONCILE_RES}))[-1]["payload"]
        for o in rec.get("order", []):
            c = (o.get("tradeData") or {}).get("comment") or o.get("clientOrderId")
            if c:
                self._order_comment[int(o["orderId"])] = str(c)
        self._last_orders = rec.get("order", [])
        positions: list[dict[str, Any]] = rec.get("position", [])
        return positions

    async def open_positions(self) -> list[BrokerPosition]:
        out = []
        for p in await self._reconcile_positions():
            td = p["tradeData"]
            sym = self._by_id.get(int(td["symbolId"]))
            if sym is None:
                continue  # an instrument Kometa does not trade
            out.append(
                BrokerPosition(
                    position_id=str(p["positionId"]),
                    symbol=sym.name,
                    side="buy" if _enum(td["tradeSide"]) == BUY else "sell",
                    lots=sym.lots(td["volume"]),
                    price_open=sym.px(p.get("price", 0)),
                    sl=sym.px(p["stopLoss"]) if p.get("stopLoss") else None,
                    tp=sym.px(p["takeProfit"]) if p.get("takeProfit") else None,
                    magic=_magic(td.get("label")),
                    comment=str(td.get("comment", "")),
                    opened_at=_from_ms(td.get("openTimestamp", 0)),
                )
            )
        return out

    async def pending_orders(self) -> list[BrokerOrder]:
        await self._reconcile_positions()
        out = []
        for o in getattr(self, "_last_orders", []):
            ot = _enum(o.get("orderType", 0))
            if ot not in (LIMIT, STOP) or o.get("closingOrder"):
                continue  # stop-loss/take-profit orders belong to positions
            td = o["tradeData"]
            sym = self._by_id.get(int(td["symbolId"]))
            if sym is None:
                continue
            price = o.get("limitPrice") if ot == LIMIT else o.get("stopPrice")
            out.append(
                BrokerOrder(
                    broker_order_id=str(o["orderId"]),
                    symbol=sym.name,
                    side="buy" if _enum(td["tradeSide"]) == BUY else "sell",
                    order_type="limit" if ot == LIMIT else "stop",
                    lots=sym.lots(td["volume"]),
                    price=sym.px(price or 0),
                    sl=sym.px(o["stopLoss"]) if o.get("stopLoss") else None,
                    tp=sym.px(o["takeProfit"]) if o.get("takeProfit") else None,
                    magic=_magic(td.get("label")),
                    comment=str(td.get("comment", "")),
                    created_at=_from_ms(td.get("openTimestamp", 0)),
                    expires_at=_from_ms(o["expirationTimestamp"]) if o.get("expirationTimestamp") else None,
                )
            )
        return out

    async def deals(self, since: datetime) -> list[BrokerDeal]:
        res = await self._req(
            DEAL_LIST,
            {"fromTimestamp": _ms(since), "toTimestamp": _ms(datetime.now(UTC)), "maxRows": 1000},
            {DEAL_LIST_RES},
        )
        out = []
        for d in res[-1]["payload"].get("deal", []):
            if _enum(d.get("dealStatus", 2)) not in (2, 3):  # FILLED, PARTIALLY_FILLED
                continue
            sym = self._by_id.get(int(d["symbolId"]))
            if sym is None:
                continue
            md = int(d.get("moneyDigits", 2))
            cpd = d.get("closePositionDetail")
            side = "buy" if _enum(d["tradeSide"]) == BUY else "sell"
            common = {
                "position_id": str(d["positionId"]),
                "symbol": sym.name,
                "side": side,
                "lots": sym.lots(d.get("filledVolume", d["volume"])),
                "price": sym.px(d.get("executionPrice", 0)),
                "time": _from_ms(d["executionTimestamp"]),
                "magic": 0,
            }
            if cpd:
                cmd = int(cpd.get("moneyDigits", md))
                out.append(
                    BrokerDeal(
                        deal_id=str(d["dealId"]),
                        entry="out",
                        commission=-abs(Decimal(int(cpd.get("commission", 0))).scaleb(-cmd)),
                        swap=Decimal(int(cpd.get("swap", 0))).scaleb(-cmd),
                        profit=Decimal(int(cpd.get("grossProfit", 0))).scaleb(-cmd),
                        comment="",
                        **common,
                    )
                )
            else:
                comment = self._order_comment.get(int(d["orderId"])) or await self._order_comment_of(
                    int(d["orderId"])
                )
                out.append(
                    BrokerDeal(
                        deal_id=str(d["dealId"]),
                        entry="in",
                        commission=-abs(Decimal(int(d.get("commission", 0))).scaleb(-md)),
                        swap=Decimal(0),
                        profit=Decimal(0),
                        comment=comment,
                        **common,
                    )
                )
        return out

    async def _order_comment_of(self, order_id: int) -> str:
        """Our client order id for a filled order the adapter has not seen (after a restart)."""
        try:
            res = await self._req(ORDER_DETAILS, {"orderId": order_id}, {ORDER_DETAILS_RES})
        except _BrokerError:
            return ""
        o = res[-1]["payload"].get("order") or {}
        c = str((o.get("tradeData") or {}).get("comment") or o.get("clientOrderId") or "")
        if c:
            self._order_comment[order_id] = c
        return c


async def list_accounts(
    client_id: str,
    client_secret: str,
    access_token: str,
    *,
    environment: str = "demo",
    connect: Callable[[str], Awaitable[Transport]] | None = None,
) -> list[dict[str, Any]]:
    """The trading accounts an access token can use (ctidTraderAccountId, login, live or demo, broker)."""
    a = CTraderAdapter(
        client_id, client_secret, access_token, 0, [], environment=environment, connect=connect
    )
    a._t = await a._connect_fn(a.url)
    a._tasks = [asyncio.create_task(a._reader())]
    try:
        await a._call(APP_AUTH, {"clientId": client_id, "clientSecret": client_secret}, {APP_AUTH_RES})
        res = await a._call(ACCOUNTS_BY_TOKEN, {"accessToken": access_token}, {ACCOUNTS_BY_TOKEN_RES})
        accounts: list[dict[str, Any]] = res[-1]["payload"].get("ctidTraderAccount", [])
        return accounts
    finally:
        await a.aclose()


class _BrokerError(Exception):
    """A request the broker answered with an error (not a transport failure)."""


def _exec_type(m: Mapping[str, Any]) -> int:
    return _enum((m.get("payload") or {}).get("executionType", 0))


def _err(m: Mapping[str, Any]) -> str:
    p = m.get("payload") or {}
    return str(p.get("errorCode") or p.get("description") or "refused")


def _factory(**kw: Any) -> CTraderAdapter:
    return CTraderAdapter(**kw)


register_adapter("ctrader", _factory)
