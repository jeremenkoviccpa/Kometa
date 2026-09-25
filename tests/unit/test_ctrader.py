"""CTraderAdapter against an in-memory fake of the cTrader Open API JSON protocol (payload types and fields
from spotware/openapi-proto-messages). A real IC Markets demo run is still its acceptance test."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from autotrader.core.alerts import MemoryAlertSink
from autotrader.core.broker import ModifyRequest, PlaceRequest, client_order_id
from autotrader.core.clock import SimClock
from autotrader.core.models import Fill, OrderIntent, RiskDecision, Signal, Timeframe
from autotrader.core.signing import (
    DecisionVerifier,
    generate_keypair,
    load_private,
    load_public,
    sign_decision,
)
from autotrader.execution import ctrader as ct
from autotrader.execution.adapter import BrokerUnavailableError
from autotrader.execution.config import ExecutionConfig
from autotrader.execution.ctrader import CTraderAdapter
from autotrader.execution.journal import Journal
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.quality import MemoryQualityLog
from autotrader.execution.quotes import QuoteBook

ACC = 7001
XAU = 41
NOW_MS = int(datetime(2026, 1, 7, 12, tzinfo=UTC).timestamp() * 1000)
COID = "at" + "b" * 29


class FakeCTrader:
    """Answers each request the way the Open API does; state: positions, orders, deals."""

    def __init__(self, hedged: bool = True, bad_token: bool = False) -> None:
        self.hedged, self.bad_token = hedged, bad_token
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.positions: dict[int, dict[str, Any]] = {}
        self.orders: dict[int, dict[str, Any]] = {}
        self.deals: list[dict[str, Any]] = []
        self.next_id = 500
        self.bid, self.ask = 434180000, 434200000  # 4341.80 / 4342.00 in 1/100,000

    # transport
    async def send(self, text: str) -> None:
        m = json.loads(text)
        self.sent.append(m)
        if m.get("payloadType") == ct.HEARTBEAT:
            return
        for pt, payload in self.handle(m["payloadType"], m["payload"]):
            self.inbox.put_nowait(
                json.dumps({"clientMsgId": m["clientMsgId"], "payloadType": pt, "payload": payload})
            )

    async def recv(self) -> str:
        return await self.inbox.get()

    async def close(self) -> None:
        pass

    def push_spot(self) -> None:
        self.inbox.put_nowait(json.dumps({"payloadType": ct.SPOT, "payload": {
            "ctidTraderAccountId": ACC, "symbolId": XAU, "bid": self.bid, "ask": self.ask, "timestamp": NOW_MS}}))  # fmt: skip

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id

    def handle(self, pt: int, p: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        acc = {"ctidTraderAccountId": ACC}
        if pt == ct.APP_AUTH:
            return [(ct.APP_AUTH_RES, {})]
        if pt == ct.ACC_AUTH:
            if self.bad_token:
                return [
                    (ct.OA_ERROR, {"errorCode": "CH_ACCESS_TOKEN_INVALID", "description": "invalid token"})
                ]
            return [(ct.ACC_AUTH_RES, acc)]
        if pt == ct.SYMBOLS_LIST:
            return [(ct.SYMBOLS_LIST_RES, {**acc, "symbol": [{"symbolId": 1, "symbolName": "EURUSD"}, {"symbolId": XAU, "symbolName": "XAUUSD"}]})]  # fmt: skip
        if pt == ct.SYMBOL_BY_ID:
            return [(ct.SYMBOL_BY_ID_RES, {**acc, "symbol": [{"symbolId": XAU, "digits": 2, "pipPosition": 1,
                     "lotSize": 10000, "minVolume": 100, "stepVolume": 100, "maxVolume": 5000000}]})]  # fmt: skip
        if pt == ct.SUB_SPOTS:
            self.push_spot_later = True
            return [(ct.SUB_SPOTS_RES, acc)]
        if pt == ct.TRADER:
            return [(ct.TRADER_RES, {**acc, "trader": {"ctidTraderAccountId": ACC, "balance": 5000000, "moneyDigits": 2,
                     "depositAssetId": 1, "accountType": 0 if self.hedged else 1}})]  # fmt: skip
        if pt == ct.ASSET_LIST:
            return [(ct.ASSET_LIST_RES, {**acc, "asset": [{"assetId": 1, "name": "USD"}]})]
        if pt == ct.RECONCILE:
            return [(ct.RECONCILE_RES, {**acc, "position": list(self.positions.values()), "order": list(self.orders.values())})]  # fmt: skip
        if pt == ct.PNL:
            return [(ct.PNL_RES, {**acc, "moneyDigits": 2, "positionUnrealizedPnL": [
                {"positionId": pid, "grossUnrealizedPnL": 1500, "netUnrealizedPnL": 1250} for pid in self.positions]})]  # fmt: skip
        if pt == ct.NEW_ORDER:
            return self.new_order(p)
        if pt == ct.AMEND_SLTP:
            pos = self.positions[p["positionId"]]
            pos["stopLoss"] = p["stopLoss"]
            return [(ct.EXECUTION, {**acc, "executionType": ct.REPLACED, "position": pos})]
        if pt == ct.CANCEL_ORDER:
            if self.orders.pop(p["orderId"], None) is None:
                return [(ct.ORDER_ERROR, {**acc, "errorCode": "OA_ORDER_NOT_FOUND"})]
            return [
                (ct.EXECUTION, {**acc, "executionType": ct.CANCELLED, "order": {"orderId": p["orderId"]}})
            ]
        if pt == ct.CLOSE_POSITION:
            pos = self.positions.pop(p["positionId"])
            deal = {"dealId": self._id(), "orderId": self._id(), "positionId": p["positionId"], "volume": p["volume"],
                    "filledVolume": p["volume"], "symbolId": XAU, "createTimestamp": NOW_MS, "executionTimestamp": NOW_MS,
                    "executionPrice": 4350.0, "tradeSide": ct.SELL, "dealStatus": 2, "moneyDigits": 2,
                    "closePositionDetail": {"entryPrice": pos["price"], "grossProfit": 8000, "swap": -40,
                                            "commission": -35, "balance": 5008000, "moneyDigits": 2}}  # fmt: skip
            self.deals.append(deal)
            return [(ct.EXECUTION, {**acc, "executionType": ct.FILLED, "deal": deal})]
        if pt == ct.DEAL_LIST:
            return [(ct.DEAL_LIST_RES, {**acc, "deal": self.deals, "hasMore": False})]
        if pt == ct.ORDER_DETAILS:
            return [
                (
                    ct.ORDER_DETAILS_RES,
                    {**acc, "order": {"orderId": p["orderId"], "tradeData": {"comment": COID}}},
                )
            ]
        if pt == ct.TRENDBARS:
            start = p["fromTimestamp"] // 60000
            bars = [{"volume": 7, "low": 433900000, "deltaOpen": 100000, "deltaHigh": 200000, "deltaClose": 150000,
                     "utcTimestampInMinutes": start + i} for i in range(2)]  # fmt: skip
            return [(ct.TRENDBARS_RES, {**acc, "period": 1, "trendbar": bars})]
        return [(ct.OA_ERROR, {"errorCode": "UNHANDLED", "description": str(pt)})]

    def new_order(self, p: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        acc = {"ctidTraderAccountId": ACC}
        side = p["tradeSide"]
        td = {"symbolId": p["symbolId"], "volume": p["volume"], "tradeSide": side, "openTimestamp": NOW_MS,
              "label": p.get("label"), "comment": p.get("comment")}  # fmt: skip
        oid = self._id()
        order = {"orderId": oid, "tradeData": td, "orderType": p["orderType"], "orderStatus": 1, "clientOrderId": p.get("clientOrderId")}  # fmt: skip
        accepted = (ct.EXECUTION, {**acc, "executionType": ct.ACCEPTED, "order": order})
        if p["orderType"] != ct.MARKET:
            self.orders[oid] = {**order, "limitPrice": p.get("limitPrice"), "stopLoss": p.get("stopLoss"),
                                "expirationTimestamp": p.get("expirationTimestamp")}  # fmt: skip
            return [accepted]
        price = self.ask / 100000 if side == ct.BUY else self.bid / 100000
        dist = p["relativeStopLoss"] / 100000
        pid = self._id()
        pos = {"positionId": pid, "tradeData": td, "positionStatus": 1, "swap": 0, "price": price,
               "stopLoss": round(price - dist if side == ct.BUY else price + dist, 2), "usedMargin": 21500, "moneyDigits": 2}  # fmt: skip
        self.positions[pid] = pos
        deal = {"dealId": self._id(), "orderId": oid, "positionId": pid, "volume": p["volume"], "filledVolume": p["volume"],
                "symbolId": p["symbolId"], "createTimestamp": NOW_MS, "executionTimestamp": NOW_MS, "executionPrice": price,
                "tradeSide": side, "dealStatus": 2, "commission": -35, "moneyDigits": 2}  # fmt: skip
        self.deals.append(deal)
        filled = (
            ct.EXECUTION,
            {**acc, "executionType": ct.FILLED, "order": order, "position": pos, "deal": deal},
        )
        return [accepted, filled]


async def adapter(fake: FakeCTrader) -> CTraderAdapter:
    async def connect(_url: str) -> FakeCTrader:
        return fake

    a = CTraderAdapter(
        "cid", "secret", "token", ACC, ["XAUUSD"], connect=connect, timeout_s=2, heartbeat_s=0.05
    )
    await a.connect()
    fake.push_spot()
    await asyncio.sleep(0.01)
    return a


def market(side: str = "buy", lots: str = "0.05", sl: str = "4332.00") -> PlaceRequest:
    return PlaceRequest(client_order_id=COID, symbol="XAUUSD", side=side, order_type="market", lots=Decimal(lots),
                        sl=Decimal(sl), magic=77)  # fmt: skip


async def test_session_symbols_quotes_and_account() -> None:
    fake = FakeCTrader()
    a = await adapter(fake)
    assert [m["payloadType"] for m in fake.sent[:2]] == [ct.APP_AUTH, ct.ACC_AUTH]
    assert fake.sent[1]["payload"] == {"ctidTraderAccountId": ACC, "accessToken": "token"}
    [xau] = await a.symbols()
    assert (xau.contract_size, xau.min_lot, xau.lot_step, xau.max_lot, xau.digits) == (
        Decimal(100), Decimal("0.01"), Decimal("0.01"), Decimal(500), 2)  # fmt: skip
    q = await anext(aiter(a.stream_quotes(["XAUUSD"])))
    assert (q.bid, q.ask) == (Decimal("4341.80"), Decimal("4342.00"))
    acct = await a.account()
    assert (acct.balance, acct.currency, acct.margin_mode, acct.trade_mode) == (
        Decimal("50000.00"),
        "USD",
        "hedging",
        "demo",
    )
    await asyncio.sleep(0.12)
    assert any(m["payloadType"] == ct.HEARTBEAT for m in fake.sent)  # keeps the session alive
    netted = await adapter(FakeCTrader(hedged=False))
    assert (await netted.account()).margin_mode == "netting"  # execution refuses it at startup


async def test_market_order_relative_stop_and_found_again_by_comment() -> None:
    fake = FakeCTrader()
    a = await adapter(fake)
    ack = await a.place(market())
    sent = next(m for m in fake.sent if m["payloadType"] == ct.NEW_ORDER)["payload"]
    assert sent["volume"] == 500 and sent["orderType"] == ct.MARKET and sent["tradeSide"] == ct.BUY
    assert sent["relativeStopLoss"] == 1000000  # 4342.00 - 4332.00 = 10.00 in 1/100,000
    assert sent["comment"] == COID and sent["clientOrderId"] == COID and sent["label"] == "m77"
    assert ack.ok and ack.filled_price == Decimal("4342.00") and ack.filled_lots == Decimal("0.05")
    [pos] = await a.open_positions()  # after a crash the order manager finds its order by comment
    assert (pos.comment, pos.magic, pos.side, pos.lots, pos.sl) == (
        COID,
        77,
        "buy",
        Decimal("0.05"),
        Decimal("4332.00"),
    )
    assert (await a.account()).equity == Decimal("50012.50")  # balance + net unrealized
    assert (await a.modify(ModifyRequest(position_id=pos.position_id, sl=Decimal("4335.00")))).ok
    assert (await a.open_positions())[0].sl == Decimal("4335.00")
    closed = await a.close_position(pos.position_id)
    assert closed.ok and closed.filled_price == Decimal("4350.00") and await a.open_positions() == []
    assert not (await a.close_position(pos.position_id)).ok


async def test_pending_orders_and_cancel() -> None:
    fake = FakeCTrader()
    a = await adapter(fake)
    exp = datetime(2026, 1, 7, 14, tzinfo=UTC)
    ack = await a.place(
        PlaceRequest(client_order_id=COID, symbol="XAUUSD", side="buy", order_type="limit", lots=Decimal("0.02"),
                     price=Decimal("4330.00"), sl=Decimal("4320.00"), magic=5, expires_at=exp)
    )  # fmt: skip
    sent = [m for m in fake.sent if m["payloadType"] == ct.NEW_ORDER][-1]["payload"]
    assert (
        sent["limitPrice"] == 4330.0
        and sent["stopLoss"] == 4320.0
        and sent["timeInForce"] == ct.GOOD_TILL_DATE
    )
    assert sent["expirationTimestamp"] == int(exp.timestamp() * 1000)
    [o] = await a.pending_orders()
    assert (o.broker_order_id, o.order_type, o.lots, o.comment) == (
        ack.broker_order_id,
        "limit",
        Decimal("0.02"),
        COID,
    )
    assert (await a.cancel(o.broker_order_id)).ok and not (await a.cancel(o.broker_order_id)).ok


async def test_deals_map_entries_to_our_order_and_exits_with_costs() -> None:
    fake = FakeCTrader()
    a = await adapter(fake)
    await a.place(market())
    await a.close_position((await a.open_positions())[0].position_id)
    a._order_comment.clear()  # a restart: the entry's order is looked up again
    deals = await a.deals(datetime(2026, 1, 1, tzinfo=UTC))
    entry = next(d for d in deals if d.entry == "in")
    exit_ = next(d for d in deals if d.entry == "out")
    assert (entry.comment, entry.lots, entry.commission) == (COID, Decimal("0.05"), Decimal("-0.35"))
    assert (exit_.profit, exit_.swap, exit_.commission) == (
        Decimal("80.00"),
        Decimal("-0.40"),
        Decimal("-0.35"),
    )
    assert len({d.deal_id for d in deals}) == len(deals)


async def test_history_is_bid_bars_with_the_current_spread() -> None:
    a = await adapter(FakeCTrader())
    start = datetime(2026, 1, 7, 12, tzinfo=UTC)
    bars = await a.history_bars("XAUUSD", Timeframe.M1, start, start + timedelta(minutes=2))
    assert [b.open_time for b in bars] == [start, start + timedelta(minutes=1)]
    b = bars[0]
    assert (b.bid_l, b.bid_o, b.bid_h, b.bid_c) == pytest.approx((4339.0, 4340.0, 4341.0, 4340.5))
    assert b.ask_c == pytest.approx(4340.7)  # + the 0.20 spread


async def test_a_bad_token_stops_cold_and_a_lost_connection_is_unavailable() -> None:
    with pytest.raises(PermissionError):
        await adapter(FakeCTrader(bad_token=True))
    fake = FakeCTrader()
    a = await adapter(fake)
    fake.handle = lambda pt, p: []  # type: ignore[method-assign]  # the broker stops answering
    with pytest.raises(BrokerUnavailableError):
        await a.account()


async def test_the_order_manager_trades_through_ctrader(tmp_path: Path) -> None:
    """A signed decision -> a gold order with a relative stop -> the exact stop confirmed -> the fill recorded."""
    fake = FakeCTrader()
    fake.ask = 434230000  # the fill will be 0.30 above the quote the stop distance was computed from
    a = await adapter(fake)
    clock = SimClock(datetime(2026, 1, 7, 12, tzinfo=UTC))
    [xau] = await a.symbols()
    quotes = QuoteBook()
    quotes.update(await anext(aiter(a.stream_quotes(["XAUUSD"]))))
    priv, pub = generate_keypair()
    fills: list[Fill] = []
    om = OrderManager(adapter=a, verifier=DecisionVerifier(load_public(pub)),
                      journal=Journal(tmp_path / "j.json"), alerts=MemoryAlertSink(), clock=clock,
                      config=ExecutionConfig(), quotes=quotes, quality=MemoryQualityLog(), account_id=str(ACC),
                      symbols={"XAUUSD": xau}, on_fill=fills.append)  # fmt: skip
    await om.initialize()
    sig = Signal(signal_id=uuid4(), strategy_id="s", strategy_version="1", symbol="XAUUSD", side="buy",
                 entry_type="market", entry_price=None, stop_price=4332.0, target_price=None,
                 created_at=clock.now(), reason="r")  # fmt: skip
    it = OrderIntent(
        intent_id=uuid4(), signal=sig, proposed_lots=Decimal("0.01"), risk_fraction=0.001, account_id="a"
    )
    d = RiskDecision(intent_id=it.intent_id, verdict="approve", approved_lots=Decimal("0.01"), reasons=(),
                     limits_snapshot_hash="h", decided_at=clock.now(),
                     expires_at=clock.now() + timedelta(seconds=5), sequence=1)  # fmt: skip
    r = await om.execute(it, sign_decision(load_private(priv), d), Timeframe.H1)
    assert r.placed and r.state == "open", r.reason
    [pos] = await a.open_positions()
    assert pos.comment == client_order_id(it.intent_id) and pos.lots == Decimal("0.01")
    assert pos.sl is not None and pos.sl >= Decimal("4332.00")  # never looser than the decision's stop
    assert om.state.orders[pos.comment].stop_confirmed and fills
    again = await om.execute(
        it, sign_decision(load_private(priv), d.model_copy(update={"sequence": 2})), Timeframe.H1
    )
    assert len(await a.open_positions()) == 1 and not again.placed  # the same intent never places twice
