"""OandaAdapter against a small stateful fake of OANDA's v20 API (the documented request/response shapes).

What a first practice run must still confirm lives in docs/open_questions.md 32 (financing and cash).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from autotrader.core.alerts import MemoryAlertSink
from autotrader.core.broker import ModifyRequest, PlaceRequest, client_order_id
from autotrader.core.clock import SimClock
from autotrader.core.models import Bar, Fill, OrderIntent, RiskDecision, Signal, Timeframe
from autotrader.core.signing import (
    DecisionVerifier,
    generate_keypair,
    load_private,
    load_public,
    sign_decision,
)
from autotrader.data.sources import FileSource
from autotrader.execution.adapter import BrokerUnavailableError
from autotrader.execution.config import ExecutionConfig
from autotrader.execution.history import fetch_history
from autotrader.execution.journal import Journal
from autotrader.execution.oanda import OandaAdapter, parse_time, to_oanda
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.quality import MemoryQualityLog
from autotrader.execution.quotes import QuoteBook

T = "2026-01-07T12:00:00.123456789Z"
COID = "at" + "a" * 29


class FakeOanda:
    def __init__(self, hedging: bool = True) -> None:
        self.hedging = hedging
        self.trades: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.txs: list[dict[str, Any]] = []
        self.next = 100
        self.fail_next: int | None = None
        self.requests: list[httpx.Request] = []

    def _id(self) -> str:
        self.next += 1
        return str(self.next)

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if self.fail_next is not None:
            code, self.fail_next = self.fail_next, None
            return httpx.Response(code, json={"errorMessage": "boom"})
        p, m = req.url.path, req.method
        body = json.loads(req.content) if req.content else {}
        if p.endswith("/summary"):
            return httpx.Response(
                200,
                headers={"Date": "Wed, 07 Jan 2026 12:00:00 GMT"},
                json={
                    "account": {
                        "id": "101-004-1-001",
                        "currency": "USD",
                        "balance": "100000.0000",
                        "NAV": "100010.5000",
                        "marginUsed": "215.0",
                        "marginAvailable": "99795.5",
                        "hedgingEnabled": self.hedging,
                    }
                },
            )
        if p.endswith("/instruments"):
            return httpx.Response(
                200,
                json={
                    "instruments": [
                        {
                            "name": "XAU_USD",
                            "displayPrecision": 3,
                            "tradeUnitsPrecision": 0,
                            "minimumTradeSize": "1",
                            "maximumOrderUnits": "5000",
                            "marginRate": "0.05",
                        }
                    ]
                },
            )
        if p.endswith("/pricing"):
            return httpx.Response(
                200,
                json={
                    "prices": [
                        {
                            "instrument": "XAU_USD",
                            "time": T,
                            "bids": [{"price": "4341.80"}],
                            "asks": [{"price": "4342.20"}],
                        }
                    ]
                },
            )
        if p.endswith("/orders") and m == "POST":
            o = body["order"]
            if o["type"] == "MARKET":
                tid, units = self._id(), o["units"]
                price = "4342.20" if not units.startswith("-") else "4341.80"
                self.trades[tid] = {
                    "id": tid,
                    "instrument": o["instrument"],
                    "price": price,
                    "openTime": T,
                    "currentUnits": units,
                    "unrealizedPL": "0",
                    "clientExtensions": o["tradeClientExtensions"],
                    "stopLossOrder": {"price": o["stopLossOnFill"]["price"]},
                }
                fill = {
                    "id": tid,
                    "type": "ORDER_FILL",
                    "time": T,
                    "instrument": o["instrument"],
                    "price": price,
                    "units": units,
                    "commission": "0.35",
                    "tradeOpened": {
                        "tradeID": tid,
                        "units": units,
                        "price": price,
                        "clientExtensions": o["tradeClientExtensions"],
                    },
                }
                self.txs.append(fill)
                return httpx.Response(
                    201, json={"orderCreateTransaction": {"id": tid}, "orderFillTransaction": fill}
                )
            oid = self._id()
            self.orders[oid] = {**o, "id": oid, "createTime": T, "state": "PENDING"}
            return httpx.Response(201, json={"orderCreateTransaction": {"id": oid}})
        if p.endswith("/openTrades"):
            return httpx.Response(200, json={"trades": list(self.trades.values())})
        if p.endswith("/pendingOrders"):
            sl = {
                "id": "999",
                "type": "STOP_LOSS",
                "tradeID": "1",
                "price": "1",
            }  # belongs to a trade: ignored
            return httpx.Response(200, json={"orders": [*self.orders.values(), sl]})
        if p.endswith("/cancel"):
            oid = p.split("/")[-2]
            if self.orders.pop(oid, None) is None:
                return httpx.Response(404, json={"errorMessage": "no such order"})
            return httpx.Response(200, json={"orderCancelTransaction": {"id": self._id()}})
        if p.endswith("/close"):
            tid = p.split("/")[-2]
            t = self.trades.pop(tid, None)
            if t is None:
                return httpx.Response(404, json={"errorMessage": "no such trade"})
            close_units = str(-Decimal(t["currentUnits"]))
            fill = {
                "id": self._id(),
                "type": "ORDER_FILL",
                "time": T,
                "instrument": t["instrument"],
                "price": "4350.00",
                "units": close_units,
                "commission": "0.35",
                "tradesClosed": [
                    {
                        "tradeID": tid,
                        "units": close_units,
                        "realizedPL": "78.00",
                        "financing": "-0.40",
                        "price": "4350.00",
                    }
                ],
            }
            self.txs.append(fill)
            return httpx.Response(200, json={"orderFillTransaction": fill})
        if p.endswith("/orders") and m == "PUT":  # trade dependent orders
            tid = p.split("/")[-2]
            self.trades[tid]["stopLossOrder"] = {"price": body["stopLoss"]["price"]}
            return httpx.Response(200, json={"stopLossOrderTransaction": {"id": self._id()}})
        if p.endswith("/transactions"):
            return httpx.Response(
                200,
                json={
                    "pages": [
                        "https://api-fxpractice.oanda.com/v3/accounts/101-004-1-001/transactions/idrange?from=1&to=999"
                    ]
                },
            )
        if p.endswith("/idrange"):
            extra = [
                {"id": "7", "type": "TRANSFER_FUNDS", "time": T, "amount": "500.00"},
                {
                    "id": "8",
                    "type": "DAILY_FINANCING",
                    "time": T,
                    "positionFinancings": [
                        {
                            "instrument": "XAU_USD",
                            "openTradeFinancings": [{"tradeID": "101", "financing": "-0.12"}],
                        }
                    ],
                },
            ]
            return httpx.Response(200, json={"transactions": [*extra, *self.txs]})
        if "/candles" in p:
            start = parse_time(req.url.params["from"])
            candles = []
            for i in range(3):
                px = {"o": "4340.0", "h": "4341.0", "l": "4339.0", "c": "4340.5"}
                ct = (start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
                ask = {k: str(float(v) + 0.3) for k, v in px.items()}
                candles.append({"complete": i < 2, "volume": 5, "time": ct, "bid": px, "ask": ask})
            return httpx.Response(200, json={"candles": candles})
        return httpx.Response(404, json={"errorMessage": f"unhandled {m} {p}"})


def adapter(fake: FakeOanda) -> OandaAdapter:
    return OandaAdapter(
        "101-004-1-001", "tok", {"XAUUSD": Decimal(100)}, transport=httpx.MockTransport(fake.handler)
    )


async def test_account_symbols_and_quotes() -> None:
    fake = FakeOanda()
    a = adapter(fake)
    acct = await a.account()
    assert (
        acct.trade_mode == "demo" and acct.margin_mode == "hedging" and acct.equity == Decimal("100010.5000")
    )
    assert acct.server_time == datetime(2026, 1, 7, 12, tzinfo=UTC)
    [xau] = await a.symbols()
    assert (xau.symbol, xau.digits, xau.min_lot, xau.lot_step, xau.max_lot) == (
        "XAUUSD",
        3,
        Decimal("0.01"),
        Decimal("0.01"),
        Decimal(50),
    )
    assert xau.margin_per_lot == Decimal("21710.000")  # mid 4342.0 x 100 oz x 5%
    q = await anext(aiter(a.stream_quotes(["XAUUSD"])))
    assert (q.bid, q.ask) == (Decimal("4341.80"), Decimal("4342.20")) and q.time.microsecond == 123456
    assert (
        await adapter(FakeOanda(hedging=False)).account()
    ).margin_mode == "netting"  # execution refuses it
    assert fake.requests[0].headers["Authorization"] == "Bearer tok"
    assert to_oanda("XAUUSD") == "XAU_USD"


async def test_market_order_carries_stop_and_client_id_and_is_found_again() -> None:
    fake = FakeOanda()
    a = adapter(fake)
    ack = await a.place(
        PlaceRequest(client_order_id=COID, symbol="XAUUSD", side="sell", order_type="market",
                     lots=Decimal("0.05"), sl=Decimal("4352.00"), magic=77)
    )  # fmt: skip
    sent = json.loads(fake.requests[-1].content)["order"]
    assert sent["units"] == "-5" and sent["stopLossOnFill"]["price"] == "4352.00"
    assert sent["positionFill"] == "OPEN_ONLY" and sent["timeInForce"] == "FOK"
    assert sent["tradeClientExtensions"] == {"id": COID, "tag": "m77", "comment": COID}
    assert ack.ok and ack.filled_lots == Decimal("0.05") and ack.filled_price == Decimal("4341.80")
    [pos] = await a.open_positions()  # after a crash the order manager finds its order by comment
    assert (pos.comment, pos.magic, pos.side, pos.lots, pos.sl) == (
        COID,
        77,
        "sell",
        Decimal("0.05"),
        Decimal("4352.00"),
    )
    assert (await a.modify(ModifyRequest(position_id=pos.position_id, sl=Decimal("4348.00")))).ok
    assert (await a.open_positions())[0].sl == Decimal("4348.00")
    closed = await a.close_position(pos.position_id)
    assert closed.ok and closed.filled_price == Decimal("4350.00") and await a.open_positions() == []
    assert not (await a.close_position(pos.position_id)).ok  # 404: already gone


async def test_pending_orders_expiry_and_cancel() -> None:
    fake = FakeOanda()
    a = adapter(fake)
    exp = datetime(2026, 1, 7, 14, tzinfo=UTC)
    ack = await a.place(
        PlaceRequest(client_order_id=COID, symbol="XAUUSD", side="buy", order_type="limit",
                     lots=Decimal("0.02"),
                     price=Decimal("4330.00"), sl=Decimal("4320.00"), magic=5, expires_at=exp)
    )  # fmt: skip
    sent = json.loads(fake.requests[-1].content)["order"]
    assert (
        sent["type"] == "LIMIT"
        and sent["timeInForce"] == "GTD"
        and sent["gtdTime"].startswith("2026-01-07T14:00:00")
    )
    [o] = await a.pending_orders()  # the trade's own stop-loss order is not an entry
    assert (o.broker_order_id, o.order_type, o.lots, o.comment) == (
        ack.broker_order_id,
        "limit",
        Decimal("0.02"),
        COID,
    )
    assert (await a.cancel(o.broker_order_id)).ok and not (await a.cancel(o.broker_order_id)).ok


async def test_deals_from_fills_closes_financing_and_transfers() -> None:
    fake = FakeOanda()
    a = adapter(fake)
    await a.place(
        PlaceRequest(client_order_id=COID, symbol="XAUUSD", side="buy", order_type="market",
                     lots=Decimal("0.01"),
                     sl=Decimal("4330.00"), magic=9)
    )  # fmt: skip
    await a.close_position((await a.open_positions())[0].position_id)
    deals = await a.deals(datetime(2026, 1, 1, tzinfo=UTC))
    by_kind = {(d.kind, d.entry, d.deal_id.split(":")[-1] if ":" in d.deal_id else "tx"): d for d in deals}
    assert by_kind[("balance", "in", "tx")].profit == Decimal("500.00")
    fin = by_kind[("trade", "out", "fin")]
    assert fin.lots == 0 and fin.swap == Decimal("-0.12")
    opened = by_kind[("trade", "in", "in")]
    assert (opened.comment, opened.lots, opened.commission) == (COID, Decimal("0.01"), Decimal("-0.35"))
    out = by_kind[("trade", "out", "out")]
    assert (out.profit, out.swap, out.lots, out.side) == (
        Decimal("78.00"),
        Decimal("-0.40"),
        Decimal("0.01"),
        "sell",
    )
    assert len({d.deal_id for d in deals}) == len(deals)  # stable, unique ids: de-duplication works


async def test_history_bars_are_complete_bid_ask_candles() -> None:
    a = adapter(FakeOanda())
    start = datetime(2026, 1, 7, 12, tzinfo=UTC)
    bars = await a.history_bars("XAUUSD", Timeframe.M1, start, start + timedelta(minutes=2))
    assert [b.open_time for b in bars] == [
        start,
        start + timedelta(minutes=1),
    ]  # the incomplete one is skipped
    assert bars[0].ask_c == pytest.approx(4340.8) and bars[0].close_time == start + timedelta(minutes=1)


async def test_errors_fail_closed() -> None:
    fake = FakeOanda()
    a = adapter(fake)
    fake.fail_next = 401
    with pytest.raises(PermissionError):
        await a.account()  # a bad token is a configuration error, never retried silently
    fake.fail_next = 503
    with pytest.raises(BrokerUnavailableError):
        await a.open_positions()  # may or may not have happened: the order manager searches first
    fake.fail_next = 400
    ack = await a.place(
        PlaceRequest(client_order_id=COID, symbol="XAUUSD", side="buy", order_type="market",
                     lots=Decimal("0.01"),
                     sl=Decimal("4330.00"), magic=1)
    )  # fmt: skip
    assert not ack.ok and ack.error == "boom"


async def test_the_order_manager_trades_through_oanda(tmp_path: Path) -> None:
    """A signed decision -> a gold order with its stop -> the stop confirmed at OANDA -> the fill recorded."""
    fake = FakeOanda()
    a = adapter(fake)
    clock = SimClock(parse_time(T))
    [xau] = await a.symbols()
    quotes = QuoteBook()
    quotes.update(await anext(aiter(a.stream_quotes(["XAUUSD"]))))
    priv, pub = generate_keypair()
    fills: list[Fill] = []
    om = OrderManager(adapter=a, verifier=DecisionVerifier(load_public(pub)),
                      journal=Journal(tmp_path / "j.json"),
                      alerts=MemoryAlertSink(), clock=clock, config=ExecutionConfig(), quotes=quotes,
                      quality=MemoryQualityLog(), account_id="101-004-1-001", symbols={"XAUUSD": xau},
                      on_fill=fills.append)  # fmt: skip
    await om.initialize()
    sig = Signal(signal_id=uuid4(), strategy_id="s", strategy_version="1", symbol="XAUUSD", side="buy",
                 entry_type="market", entry_price=None, stop_price=4332.0, target_price=None,
                 created_at=clock.now(), reason="r")  # fmt: skip
    it = OrderIntent(
        intent_id=uuid4(), signal=sig, proposed_lots=Decimal("0.01"), risk_fraction=0.001, account_id="a"
    )
    d = RiskDecision(intent_id=it.intent_id, verdict="approve", approved_lots=Decimal("0.01"), reasons=(),
                     limits_snapshot_hash="h", decided_at=clock.now(),
                     expires_at=clock.now() + timedelta(seconds=5),
                     sequence=1)  # fmt: skip
    r = await om.execute(it, sign_decision(load_private(priv), d), Timeframe.H1)
    assert r.placed and r.state == "open", r.reason
    [pos] = await a.open_positions()
    assert (
        pos.comment == client_order_id(it.intent_id)
        and pos.sl == Decimal("4332.00")
        and pos.lots == Decimal("0.01")
    )
    assert om.state.orders[pos.comment].stop_confirmed and fills[0].price == Decimal("4342.20")
    again = await om.execute(
        it, sign_decision(load_private(priv), d.model_copy(update={"sequence": 2})), Timeframe.H1
    )
    assert len(await a.open_positions()) == 1 and not again.placed  # the same intent never places twice


async def test_broker_history_is_cached_by_month_and_readable_by_validation(tmp_path: Path) -> None:
    calls: list[datetime] = []

    class Stub:
        async def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]:
            calls.append(start)
            return [Bar(symbol=symbol, timeframe=tf, open_time=start, close_time=start + timedelta(minutes=1),
                        bid_o=1.0, bid_h=2.0, bid_l=0.5, bid_c=1.5,
                        ask_o=1.1, ask_h=2.1, ask_l=0.6, ask_c=1.6,
                        volume=3.0)]  # fmt: skip

    stub = Stub()
    n = await fetch_history(stub, "XAUUSD", date(2025, 11, 5), date(2026, 1, 3), tmp_path)  # type: ignore[arg-type]
    assert n == 3 and [c.month for c in calls] == [11, 12, 1]
    again = await fetch_history(stub, "XAUUSD", date(2025, 11, 5), date(2026, 1, 3), tmp_path)  # type: ignore[arg-type]
    assert again == 3 and len(calls) == 3  # finished months are never fetched twice
    df = FileSource(tmp_path).m1_bars(
        "XAUUSD", datetime(2025, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )
    assert df.height == 3 and df["ask_c"][0] == 1.6
