"""OandaAdapter: BrokerAdapter over OANDA's v20 REST API (practice or live), no MT5 and no Windows.

Mapping to Kometa's broker model (which follows MT5):
- Symbols: "XAUUSD" <-> "XAU_USD". Lots <-> units: units = lots x contract size (config/instruments.yaml:
  1 gold lot = 100 units = 100 oz, 1 FX lot = 100,000 units).
- Every order carries our 31-char client order id as its clientExtensions id AND as the resulting trade's
  (tradeClientExtensions), so the order manager finds its orders and positions by comment, exactly as with
  MT5; the strategy's magic number rides in the tag ("m<magic>").
- A position is an OANDA trade (one per order; the account must have hedging enabled, else it reports
  "netting" and execution refuses to start).
- Deals come from the transaction history: ORDER_FILL opens are "in", closed/reduced trades are "out"
  (realized P&L and financing), transfers are balance deals.
- Account mode: the practice host is a demo account, the fxtrade host a real one (startup checks refuse a
  real account outside env=live).

Transport failures, timeouts and 5xx become BrokerUnavailableError (the order manager then searches for
its client order id before any retry); 401/403 raise PermissionError and are never retried silently.
SPEC-QUESTION: OANDA reports daily financing as separate transactions; the first practice runs must confirm
the cash reconciliation matches (docs/open_questions.md 32).
"""

from __future__ import annotations

import asyncio
import email.utils
import time
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx

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
    "practice": ("https://api-fxpractice.oanda.com", "demo"),
    "live": ("https://api-fxtrade.oanda.com", "real"),
}
GRANULARITY = {
    Timeframe.M1: "M1",
    Timeframe.M5: "M5",
    Timeframe.M15: "M15",
    Timeframe.H1: "H1",
    Timeframe.H4: "H4",
    Timeframe.D1: "D",
}
ENTRY_TYPES = {"LIMIT": "limit", "STOP": "stop", "MARKET_IF_TOUCHED": "limit"}
MAX_CANDLES = 5000


def to_oanda(symbol: str) -> str:
    return symbol if "_" in symbol else f"{symbol[:3]}_{symbol[3:]}"


def from_oanda(instrument: str) -> str:
    return instrument.replace("_", "")


def parse_time(s: str) -> datetime:
    """OANDA RFC3339 with nanoseconds ("...T12:00:00.123456789Z") -> aware UTC datetime (microseconds)."""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def rfc3339(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _dec(x: Any, default: str = "0") -> Decimal:
    return Decimal(str(x)) if x not in (None, "") else Decimal(default)


def _magic(ext: Mapping[str, Any] | None) -> int:
    tag = str((ext or {}).get("tag", ""))
    return int(tag[1:]) if tag.startswith("m") and tag[1:].isdigit() else 0


class OandaAdapter:
    def __init__(
        self,
        account_id: str,
        token: str,
        contract_size: Mapping[str, Decimal],
        *,
        environment: str = "practice",
        timeout_s: float = 10.0,
        quote_poll_s: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if environment not in HOSTS:
            raise ValueError(f"environment must be one of {sorted(HOSTS)}")
        base, self.trade_mode = HOSTS[environment]
        self.account_id = account_id
        self.contract = {k: Decimal(v) for k, v in contract_size.items()}
        self.quote_poll_s = quote_poll_s
        self._client = httpx.AsyncClient(
            base_url=f"{base}/v3",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            },
            timeout=timeout_s,
            transport=transport,
        )
        self._acct = f"/accounts/{account_id}"

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------ transport

    async def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = await self._client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise BrokerUnavailableError(f"{method} {path}: {type(e).__name__}") from e
        if r.status_code in (401, 403):
            raise PermissionError(f"OANDA refused the token or account ({r.status_code})")
        if r.status_code >= 500 or r.status_code == 429:
            raise BrokerUnavailableError(f"{method} {path}: {r.status_code} {r.text[:200]}")
        return r

    async def _json(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        r = await self._call(method, path, **kw)
        if r.status_code >= 400:
            raise BrokerUnavailableError(f"{method} {path}: {r.status_code} {r.text[:200]}")
        out: dict[str, Any] = r.json()
        return out

    def _lots(self, symbol: str, units: Any) -> Decimal:
        return abs(_dec(units)) / self.contract[symbol]

    def _units(self, symbol: str, lots: Decimal, side: str) -> str:
        u = lots * self.contract[symbol]
        return str(u.normalize() if side == "buy" else (-u).normalize())

    # ------------------------------------------------------------ account and market

    async def connect(self) -> None:
        await self.account()

    async def account(self) -> AccountInfo:
        r = await self._call("GET", f"{self._acct}/summary")
        if r.status_code >= 400:
            raise BrokerUnavailableError(f"account summary: {r.status_code} {r.text[:200]}")
        a = r.json()["account"]
        date = r.headers.get("Date")
        server = email.utils.parsedate_to_datetime(date) if date else datetime.now(UTC)
        return AccountInfo(
            account_id=str(a["id"]),
            currency=str(a["currency"]),
            balance=_dec(a["balance"]),
            equity=_dec(a.get("NAV", a["balance"])),
            margin=_dec(a.get("marginUsed")),
            free_margin=_dec(a.get("marginAvailable")),
            server_time=server.astimezone(UTC),
            trade_mode=self.trade_mode,
            margin_mode="hedging" if a.get("hedgingEnabled") else "netting",
        )

    async def _prices(self, symbols: list[str]) -> list[dict[str, Any]]:
        d = await self._json(
            "GET", f"{self._acct}/pricing", params={"instruments": ",".join(to_oanda(s) for s in symbols)}
        )
        prices: list[dict[str, Any]] = d.get("prices", [])
        return prices

    async def symbols(self) -> list[SymbolInfo]:
        syms = sorted(self.contract)
        d = await self._json(
            "GET", f"{self._acct}/instruments", params={"instruments": ",".join(to_oanda(s) for s in syms)}
        )
        mids = {}
        for p in await self._prices(syms):
            if p.get("bids") and p.get("asks"):
                mids[from_oanda(p["instrument"])] = (
                    _dec(p["bids"][0]["price"]) + _dec(p["asks"][0]["price"])
                ) / 2
        out = []
        for ins in d.get("instruments", []):
            sym = from_oanda(ins["name"])
            if sym not in self.contract:
                continue
            cs = self.contract[sym]
            digits = int(ins["displayPrecision"])
            unit_step = Decimal(1).scaleb(-int(ins.get("tradeUnitsPrecision", 0)))
            margin = mids[sym] * cs * _dec(ins.get("marginRate")) if sym in mids else None
            out.append(
                SymbolInfo(
                    symbol=sym,
                    digits=digits,
                    point=Decimal(1).scaleb(-digits),
                    contract_size=cs,
                    min_lot=_dec(ins.get("minimumTradeSize"), "1") / cs,
                    lot_step=unit_step / cs,
                    max_lot=_dec(ins.get("maximumOrderUnits"), "100000000") / cs,
                    margin_per_lot=margin,
                )
            )
        return out

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """Polls OANDA pricing and yields each quote whose time moved."""
        last: dict[str, datetime] = {}
        while True:
            for p in await self._prices(symbols):
                if not (p.get("bids") and p.get("asks")):
                    continue
                q = Quote(
                    symbol=from_oanda(p["instrument"]),
                    bid=_dec(p["bids"][0]["price"]),
                    ask=_dec(p["asks"][0]["price"]),
                    time=parse_time(p["time"]),
                )
                if last.get(q.symbol) != q.time:
                    last[q.symbol] = q.time
                    yield q
            await asyncio.sleep(self.quote_poll_s)

    async def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]:
        """Complete bid/ask candles in [start, end), paged 5,000 at a time."""
        out: list[Bar] = []
        step = timedelta(minutes=tf.minutes)
        cursor = start
        while cursor < end:
            d = await self._json(
                "GET",
                f"/instruments/{to_oanda(symbol)}/candles",
                params={
                    "granularity": GRANULARITY[tf],
                    "price": "BA",
                    "from": rfc3339(cursor),
                    "count": MAX_CANDLES,
                },
            )
            candles = d.get("candles", [])
            if not candles:
                break
            for c in candles:
                t = parse_time(c["time"])
                if t >= end:
                    return out
                if c.get("complete") and "bid" in c and "ask" in c:
                    b, a = c["bid"], c["ask"]
                    out.append(
                        Bar(
                            symbol=symbol,
                            timeframe=tf,
                            open_time=t,
                            close_time=t + step,
                            bid_o=float(b["o"]),
                            bid_h=float(b["h"]),
                            bid_l=float(b["l"]),
                            bid_c=float(b["c"]),
                            ask_o=float(a["o"]),
                            ask_h=float(a["h"]),
                            ask_l=float(a["l"]),
                            ask_c=float(a["c"]),
                            volume=float(c.get("volume", 0)),
                        )
                    )
            nxt = parse_time(candles[-1]["time"]) + step
            if nxt <= cursor:
                break
            cursor = nxt
        return out

    # ------------------------------------------------------------ orders and positions

    async def place(self, req: PlaceRequest) -> BrokerAck:
        ext = {"id": req.client_order_id, "tag": f"m{req.magic}", "comment": req.client_order_id}
        order: dict[str, Any] = {
            "instrument": to_oanda(req.symbol),
            "units": self._units(req.symbol, req.lots, req.side),
            "positionFill": "OPEN_ONLY",  # a new trade, never netting against another
            "stopLossOnFill": {"price": str(req.sl), "timeInForce": "GTC"},
            "clientExtensions": ext,
            "tradeClientExtensions": ext,
        }
        if req.tp is not None:
            order["takeProfitOnFill"] = {"price": str(req.tp), "timeInForce": "GTC"}
        if req.order_type == "market":
            order |= {"type": "MARKET", "timeInForce": "FOK"}
        else:
            if req.price is None:
                return BrokerAck(ok=False, error="pending order without price")
            order |= {"type": "LIMIT" if req.order_type == "limit" else "STOP", "price": str(req.price)}
            order |= (
                {"timeInForce": "GTD", "gtdTime": rfc3339(req.expires_at)}
                if req.expires_at
                else {"timeInForce": "GTC"}
            )
        t0 = time.perf_counter()
        r = await self._call("POST", f"{self._acct}/orders", json={"order": order})
        latency = (time.perf_counter() - t0) * 1000
        d = r.json() if r.content else {}
        if r.status_code >= 400:
            reason = d.get("errorMessage") or (d.get("orderRejectTransaction") or {}).get("rejectReason")
            return BrokerAck(ok=False, error=str(reason or r.status_code), latency_ms=latency)
        fill = d.get("orderFillTransaction")
        if fill and fill.get("tradeOpened"):
            opened = fill["tradeOpened"]
            return BrokerAck(
                ok=True,
                position_id=str(opened["tradeID"]),
                filled_price=_dec(opened.get("price", fill.get("price"))),
                filled_lots=self._lots(req.symbol, opened["units"]),
                latency_ms=latency,
            )
        cancel = d.get("orderCancelTransaction")
        if cancel:
            return BrokerAck(ok=False, error=str(cancel.get("reason", "cancelled")), latency_ms=latency)
        created = d.get("orderCreateTransaction") or {}
        return BrokerAck(ok=True, broker_order_id=str(created.get("id")), latency_ms=latency)

    async def modify(self, req: ModifyRequest) -> BrokerAck:
        if req.position_id is not None:
            body: dict[str, Any] = {}
            if req.sl is not None:
                body["stopLoss"] = {"price": str(req.sl), "timeInForce": "GTC"}
            if req.tp is not None:
                body["takeProfit"] = {"price": str(req.tp), "timeInForce": "GTC"}
            r = await self._call("PUT", f"{self._acct}/trades/{req.position_id}/orders", json=body)
            if r.status_code >= 400:
                return BrokerAck(ok=False, error=r.text[:200])
            return BrokerAck(ok=True, position_id=req.position_id)
        if req.broker_order_id is None:
            return BrokerAck(ok=False, error="nothing to modify")
        # a pending order is changed by replacing it (OANDA gives the replacement a new id)
        cur = await self._call("GET", f"{self._acct}/orders/{req.broker_order_id}")
        if cur.status_code >= 400:
            return BrokerAck(ok=False, error="no such order")
        o = cur.json()["order"]
        new = {
            k: o[k] for k in ("type", "instrument", "units", "price", "timeInForce", "positionFill") if k in o
        }
        new |= {k: o[k] for k in ("gtdTime", "clientExtensions", "tradeClientExtensions") if k in o}
        new["stopLossOnFill"] = {"price": str(req.sl if req.sl is not None else o["stopLossOnFill"]["price"])}
        if req.tp is not None or "takeProfitOnFill" in o:
            new["takeProfitOnFill"] = {
                "price": str(req.tp if req.tp is not None else o["takeProfitOnFill"]["price"])
            }
        r = await self._call("PUT", f"{self._acct}/orders/{req.broker_order_id}", json={"order": new})
        if r.status_code >= 400:
            return BrokerAck(ok=False, error=r.text[:200])
        created = r.json().get("orderCreateTransaction") or {}
        return BrokerAck(ok=True, broker_order_id=str(created.get("id", req.broker_order_id)))

    async def cancel(self, broker_order_id: str) -> BrokerAck:
        r = await self._call("PUT", f"{self._acct}/orders/{broker_order_id}/cancel")
        if r.status_code >= 400:
            return BrokerAck(ok=False, error="no such order" if r.status_code == 404 else r.text[:200])
        return BrokerAck(ok=True, broker_order_id=broker_order_id)

    async def close_position(self, position_id: str, lots: Decimal | None = None) -> BrokerAck:
        units = "ALL"
        if lots is not None:
            trades = {p.position_id: p for p in await self.open_positions()}
            p = trades.get(position_id)
            if p is None:
                return BrokerAck(ok=False, error="no such position")
            units = str((lots * self.contract[p.symbol]).normalize())
        r = await self._call("PUT", f"{self._acct}/trades/{position_id}/close", json={"units": units})
        if r.status_code >= 400:
            return BrokerAck(ok=False, error="no such position" if r.status_code == 404 else r.text[:200])
        fill = r.json().get("orderFillTransaction") or {}
        sym = from_oanda(fill.get("instrument", ""))
        return BrokerAck(
            ok=True,
            position_id=position_id,
            filled_price=_dec(fill.get("price")) if fill.get("price") else None,
            filled_lots=self._lots(sym, fill.get("units", 0)) if sym in self.contract else None,
        )

    async def open_positions(self) -> list[BrokerPosition]:
        d = await self._json("GET", f"{self._acct}/openTrades")
        out = []
        for t in d.get("trades", []):
            sym = from_oanda(t["instrument"])
            if sym not in self.contract:
                continue  # an instrument Kometa does not know: reconciliation sees it as external elsewhere
            units = _dec(t["currentUnits"])
            ext = t.get("clientExtensions") or {}
            out.append(
                BrokerPosition(
                    position_id=str(t["id"]),
                    symbol=sym,
                    side="buy" if units > 0 else "sell",
                    lots=self._lots(sym, units),
                    price_open=_dec(t["price"]),
                    sl=_dec((t.get("stopLossOrder") or {}).get("price")) if t.get("stopLossOrder") else None,
                    tp=_dec((t.get("takeProfitOrder") or {}).get("price"))
                    if t.get("takeProfitOrder")
                    else None,
                    magic=_magic(ext),
                    comment=str(ext.get("id", "")),
                    opened_at=parse_time(t["openTime"]),
                    profit=_dec(t.get("unrealizedPL")),
                )
            )
        return out

    async def pending_orders(self) -> list[BrokerOrder]:
        d = await self._json("GET", f"{self._acct}/pendingOrders")
        out = []
        for o in d.get("orders", []):
            if o.get("type") not in ENTRY_TYPES:
                continue  # stop-loss and take-profit orders belong to trades
            sym = from_oanda(o["instrument"])
            if sym not in self.contract:
                continue
            units = _dec(o["units"])
            ext = o.get("clientExtensions") or {}
            out.append(
                BrokerOrder(
                    broker_order_id=str(o["id"]),
                    symbol=sym,
                    side="buy" if units > 0 else "sell",
                    order_type=ENTRY_TYPES[o["type"]],
                    lots=self._lots(sym, units),
                    price=_dec(o["price"]),
                    sl=_dec(o["stopLossOnFill"]["price"]) if o.get("stopLossOnFill") else None,
                    tp=_dec(o["takeProfitOnFill"]["price"]) if o.get("takeProfitOnFill") else None,
                    magic=_magic(ext),
                    comment=str(ext.get("id", "")),
                    created_at=parse_time(o["createTime"]),
                    expires_at=parse_time(o["gtdTime"]) if o.get("gtdTime") else None,
                )
            )
        return out

    async def deals(self, since: datetime) -> list[BrokerDeal]:
        d = await self._json(
            "GET",
            f"{self._acct}/transactions",
            params={"from": rfc3339(since), "to": rfc3339(datetime.now(UTC)), "pageSize": 1000},
        )
        txs: list[dict[str, Any]] = []
        for page in d.get("pages", []):
            path = httpx.URL(page).raw_path.decode()
            txs += (await self._json("GET", path.removeprefix("/v3"))).get("transactions", [])
        return [deal for tx in txs for deal in self._deals_of(tx)]

    def _deals_of(self, tx: dict[str, Any]) -> list[BrokerDeal]:
        t = parse_time(tx["time"])
        kind = tx.get("type")
        if kind == "DAILY_FINANCING":
            return self._financing_deals(tx, t)
        if kind == "TRANSFER_FUNDS":
            return [
                BrokerDeal(
                    deal_id=str(tx["id"]),
                    kind="balance",
                    position_id="",
                    symbol="",
                    side="buy",
                    entry="in",
                    lots=Decimal(0),
                    price=Decimal(0),
                    commission=Decimal(0),
                    swap=Decimal(0),
                    profit=_dec(tx.get("amount")),
                    time=t,
                    magic=0,
                    comment="",
                )
            ]
        if kind != "ORDER_FILL":
            return []
        sym = from_oanda(tx.get("instrument", ""))
        if sym not in self.contract:
            return []
        out = []
        price = _dec(tx.get("price"))
        commission = -abs(_dec(tx.get("commission")))
        opened = tx.get("tradeOpened")
        if opened:
            ext = opened.get("clientExtensions") or tx.get("clientExtensions") or {}
            units = _dec(opened["units"])
            out.append(
                BrokerDeal(
                    deal_id=f"{tx['id']}:{opened['tradeID']}:in",
                    position_id=str(opened["tradeID"]),
                    symbol=sym,
                    side="buy" if units > 0 else "sell",
                    entry="in",
                    lots=self._lots(sym, units),
                    price=_dec(opened.get("price", price)),
                    commission=commission,
                    swap=Decimal(0),
                    profit=Decimal(0),
                    time=t,
                    magic=_magic(ext),
                    comment=str(ext.get("id", "")),
                )
            )
            commission = Decimal(0)  # charged once per fill
        closes = list(tx.get("tradesClosed") or []) + ([tx["tradeReduced"]] if tx.get("tradeReduced") else [])
        for c in closes:
            units = _dec(c["units"])  # negative for a closed buy
            out.append(
                BrokerDeal(
                    deal_id=f"{tx['id']}:{c['tradeID']}:out",
                    position_id=str(c["tradeID"]),
                    symbol=sym,
                    side="sell" if units < 0 else "buy",
                    entry="out",
                    lots=self._lots(sym, units),
                    price=_dec(c.get("price", price)),
                    commission=commission,
                    swap=_dec(c.get("financing")),
                    profit=_dec(c.get("realizedPL")),
                    time=t,
                    magic=0,
                    comment="",
                )
            )
            commission = Decimal(0)
        return out

    def _financing_deals(self, tx: dict[str, Any], t: datetime) -> list[BrokerDeal]:
        """Daily financing per open trade as zero-lot deals: counted in the cash check, trade stays open."""
        out = []
        for pos in tx.get("positionFinancings", []):
            sym = from_oanda(pos.get("instrument", ""))
            if sym not in self.contract:
                continue
            for f in pos.get("openTradeFinancings", []):
                out.append(
                    BrokerDeal(
                        deal_id=f"{tx['id']}:{f['tradeID']}:fin",
                        position_id=str(f["tradeID"]),
                        symbol=sym,
                        side="buy",
                        entry="out",
                        lots=Decimal(0),
                        price=Decimal(0),
                        commission=Decimal(0),
                        swap=_dec(f.get("financing")),
                        profit=Decimal(0),
                        time=t,
                        magic=0,
                        comment="",
                    )
                )
        return out


def _factory(**kw: Any) -> OandaAdapter:
    return OandaAdapter(**kw)


register_adapter("oanda", _factory)
