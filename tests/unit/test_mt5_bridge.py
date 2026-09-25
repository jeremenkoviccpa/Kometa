"""MT5 bridge and adapter (spec section 12), against a fake `MetaTrader5` module.

The fake follows the documented MetaTrader5 API shapes (namedtuple-like records, server-time
timestamps, order_send request dicts). The last test runs the real order manager through the HTTP
adapter and the bridge app into the fake terminal.
"""

from __future__ import annotations

from datetime import UTC, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest

from autotrader.core.alerts import MemoryAlertSink
from autotrader.core.broker import PlaceRequest, client_order_id
from autotrader.core.clock import SimClock
from autotrader.core.models import OrderIntent, RiskDecision, Signal, Timeframe
from autotrader.core.signing import (
    DecisionVerifier,
    generate_keypair,
    load_private,
    load_public,
    sign_decision,
)
from autotrader.core.timeutil import utc
from autotrader.execution.adapter import BrokerUnavailableError, make_adapter
from autotrader.execution.config import ExecutionConfig
from autotrader.execution.journal import Journal
from autotrader.execution.mt5 import MT5Adapter
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.quality import MemoryQualityLog
from autotrader.execution.quotes import QuoteBook
from autotrader.execution.reconcile import Reconciler
from autotrader.mt5_bridge.app import create_app
from autotrader.mt5_bridge.servertime import ServerTimeZone
from autotrader.mt5_bridge.terminal import (
    ORDER_FILLING_FOK,
    ORDER_TIME_SPECIFIED,
    ORDER_TYPE_BUY,
    ORDER_TYPE_BUY_LIMIT,
    TRADE_ACTION_DEAL,
    TRADE_ACTION_PENDING,
    MT5Terminal,
    TerminalError,
)

TOKEN = "t" * 40
TZ = ServerTimeZone("NY+7")
NOW = utc(2026, 1, 7, 12)  # winter: server = UTC+2
GATE_PRIV, GATE_PUB = generate_keypair()


class FakeMT5:
    """Just enough of the MetaTrader5 package, with server-time timestamps."""

    def __init__(self, trade_mode: int = 0) -> None:
        self.server_s = float(TZ.to_server_seconds(NOW))
        self.trade_mode = trade_mode
        self.balance = 10000.0
        self.bid, self.ask = 4341.80, 4342.00
        self.positions: dict[int, SimpleNamespace] = {}
        self.orders: dict[int, SimpleNamespace] = {}
        self.deals: list[SimpleNamespace] = []
        self.requests: list[dict[str, Any]] = []
        self.ticket = 5000
        self.broken = False

    def _next(self) -> int:
        self.ticket += 1
        return self.ticket

    def last_error(self) -> tuple[int, str]:
        return (-10004, "No IPC connection")

    def initialize(self, **_kw: Any) -> bool:
        return not self.broken

    def symbol_select(self, _s: str, _on: bool) -> bool | None:
        return None if self.broken else True

    def account_info(self) -> SimpleNamespace | None:
        if self.broken:
            return None
        return SimpleNamespace(
            login=123456,
            currency="USD",
            balance=self.balance,
            equity=self.balance,
            margin=0.0,
            margin_free=self.balance,
            trade_mode=self.trade_mode,
            margin_mode=2,
        )

    def symbol_info(self, s: str) -> SimpleNamespace | None:
        if self.broken:
            return None
        return SimpleNamespace(
            name=s,
            digits=2,
            point=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_step=0.01,
            volume_max=50.0,
            trade_mode=4,
            filling_mode=1,
            ask=self.ask,
        )

    def symbol_info_tick(self, _s: str) -> SimpleNamespace | None:
        if self.broken:
            return None
        return SimpleNamespace(bid=self.bid, ask=self.ask, time_msc=int(self.server_s * 1000))

    def order_calc_margin(self, *_a: Any) -> float:
        return 868.4

    def copy_rates_range(self, _s: str, _tf: int, start: Any, _end: Any) -> list[dict[str, float]]:
        t0 = start.replace(tzinfo=UTC).timestamp()  # naive server wall clock -> server seconds
        return [
            {"time": t0, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "tick_volume": 10, "spread": 20}
        ]

    def positions_get(self, ticket: int | None = None) -> tuple[SimpleNamespace, ...] | None:
        if self.broken:
            return None
        return tuple(p for p in self.positions.values() if ticket is None or p.ticket == ticket)

    def orders_get(self, ticket: int | None = None) -> tuple[SimpleNamespace, ...] | None:
        if self.broken:
            return None
        return tuple(o for o in self.orders.values() if ticket is None or o.ticket == ticket)

    def history_deals_get(self, _a: Any, _b: Any) -> tuple[SimpleNamespace, ...] | None:
        return None if self.broken else tuple(self.deals)

    def _deal(self, pos: SimpleNamespace, entry: int, volume: float, price: float, profit: float) -> None:
        self.deals.append(
            SimpleNamespace(
                ticket=self._next(),
                type=pos.type if entry == 0 else 1 - pos.type,
                entry=entry,
                position_id=pos.ticket,
                symbol=pos.symbol,
                volume=volume,
                price=price,
                commission=0.0,
                swap=0.0,
                profit=profit,
                time_msc=int(self.server_s * 1000),
                magic=pos.magic,
                comment=pos.comment,
            )
        )

    def order_send(self, r: dict[str, Any]) -> SimpleNamespace | None:
        if self.broken:
            return None
        self.requests.append(r)
        done = SimpleNamespace(retcode=10009, comment="done", order=0, price=0.0, volume=0.0)
        if r["action"] == TRADE_ACTION_DEAL and "position" not in r:
            t = self._next()
            pos = SimpleNamespace(
                ticket=t,
                symbol=r["symbol"],
                type=r["type"],
                volume=r["volume"],
                price_open=r["price"],
                sl=r["sl"],
                tp=r["tp"],
                magic=r["magic"],
                comment=r["comment"],
                time_msc=int(self.server_s * 1000),
                profit=0.0,
            )
            self.positions[t] = pos
            self._deal(pos, 0, r["volume"], r["price"], 0.0)
            done.order, done.price, done.volume = t, r["price"], r["volume"]
        elif r["action"] == TRADE_ACTION_DEAL:
            pos = self.positions.pop(r["position"])
            profit = (r["price"] - pos.price_open) * 100 * pos.volume * (1 if pos.type == 0 else -1)
            self.balance += profit
            self._deal(pos, 1, r["volume"], r["price"], profit)
            done.price, done.volume = r["price"], r["volume"]
        elif r["action"] == TRADE_ACTION_PENDING:
            t = self._next()
            self.orders[t] = SimpleNamespace(
                ticket=t,
                symbol=r["symbol"],
                type=r["type"],
                volume_current=r["volume"],
                price_open=r["price"],
                sl=r["sl"],
                tp=r["tp"],
                magic=r["magic"],
                comment=r["comment"],
                time_setup_msc=int(self.server_s * 1000),
                time_expiration=r.get("expiration", 0),
            )
            done.retcode, done.order = 10008, t
        elif r["action"] == 6:  # SLTP
            self.positions[r["position"]].sl = r["sl"]
        elif r["action"] == 8:  # REMOVE
            self.orders.pop(r["order"])
        return done


def terminal(mt5: FakeMT5 | None = None, expect: str = "demo") -> MT5Terminal:
    return MT5Terminal(mt5 or FakeMT5(), TZ, ["XAUUSD"], expect_trade_mode=expect)  # type: ignore[arg-type]


def adapter_for(term: MT5Terminal, token: str = TOKEN) -> MT5Adapter:
    transport = httpx.ASGITransport(app=create_app(term, TOKEN))
    return MT5Adapter("http://bridge", token, transport=transport)


# ---------------------------------------------------------------- server time


def test_server_time_ny_plus_7_follows_us_dst() -> None:
    assert TZ.to_server(utc(2026, 1, 7, 22)) == utc(2026, 1, 8, 0).replace(tzinfo=None)  # 17:00 NY = 00:00
    assert TZ.to_server(utc(2026, 7, 7, 21)) == utc(2026, 7, 8, 0).replace(tzinfo=None)  # EDT
    for t in (utc(2026, 1, 7, 12, 30), utc(2026, 7, 7, 3, 15)):
        assert TZ.to_utc(TZ.to_server_seconds(t)) == t


def test_server_time_iana_and_required() -> None:
    athens = ServerTimeZone("Europe/Athens")
    assert athens.to_server(utc(2026, 1, 7, 12)) == utc(2026, 1, 7, 14).replace(tzinfo=None)
    with pytest.raises(ValueError, match="required"):
        ServerTimeZone("")


# ---------------------------------------------------------------- terminal mapping


def test_connect_refuses_wrong_account_type() -> None:
    with pytest.raises(TerminalError, match="real"):
        terminal(FakeMT5(trade_mode=2), expect="demo").connect()
    terminal(FakeMT5(trade_mode=2), expect="real").connect()


def test_market_request_carries_stop_magic_comment_and_filling() -> None:
    mt5 = FakeMT5()
    term = terminal(mt5)
    coid = client_order_id(uuid4())
    req = PlaceRequest(
        client_order_id=coid,
        symbol="XAUUSD",
        side="buy",
        order_type="market",
        lots=Decimal("0.10"),
        sl=Decimal("4332.00"),
        magic=77,
    )
    ack = term.place(req)
    [r] = mt5.requests
    assert r["action"] == TRADE_ACTION_DEAL and r["type"] == ORDER_TYPE_BUY and r["price"] == 4342.0
    assert r["sl"] == 4332.0 and r["magic"] == 77 and r["comment"] == coid
    assert r["type_filling"] == ORDER_FILLING_FOK and r["tp"] == 0.0
    assert ack.ok and ack.position_id == ack.broker_order_id and ack.filled_lots == Decimal("0.1")
    [p] = term.open_positions()
    assert p.sl == Decimal("4332.00") and p.opened_at == NOW and p.comment == coid


def test_pending_expiration_is_server_time() -> None:
    mt5 = FakeMT5()
    term = terminal(mt5)
    expires = NOW + timedelta(hours=2)
    term.place(
        PlaceRequest(
            client_order_id=client_order_id(uuid4()),
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            lots=Decimal("0.10"),
            price=Decimal("4330.00"),
            sl=Decimal("4320.00"),
            magic=1,
            expires_at=expires,
        )
    )
    [r] = mt5.requests
    assert r["type"] == ORDER_TYPE_BUY_LIMIT and r["type_time"] == ORDER_TIME_SPECIFIED
    assert r["expiration"] == TZ.to_server_seconds(expires)
    [o] = term.pending_orders()
    assert o.expires_at == expires and o.order_type == "limit" and o.side == "buy"


def test_history_bars_build_ask_from_spread() -> None:
    [b] = terminal().history_bars("XAUUSD", Timeframe.H1, NOW, NOW + timedelta(hours=1))
    assert b.open_time == NOW and b.close_time == NOW + timedelta(hours=1)
    assert b.ask_o == pytest.approx(1.2)


# ---------------------------------------------------------------- HTTP app and adapter


async def test_bridge_requires_token() -> None:
    with pytest.raises(ValueError, match="at least"):
        create_app(terminal(), "short")
    with pytest.raises(PermissionError):
        await adapter_for(terminal(), token="x" * 40).account()
    acct = await adapter_for(terminal()).account()
    assert acct.account_id == "123456" and acct.server_time == NOW and acct.trade_mode == "demo"


async def test_terminal_failure_is_unavailable_not_rejection() -> None:
    mt5 = FakeMT5()
    a = adapter_for(terminal(mt5))
    mt5.broken = True
    with pytest.raises(BrokerUnavailableError):
        await a.open_positions()


async def test_adapter_registered_by_name() -> None:
    a = make_adapter("mt5", base_url="http://bridge", token=TOKEN)
    assert isinstance(a, MT5Adapter)
    await a.aclose()


async def test_order_manager_end_to_end_through_bridge(tmp_path: Path) -> None:
    mt5 = FakeMT5()
    adapter = adapter_for(terminal(mt5))
    clock = SimClock(NOW)
    quotes = QuoteBook()
    [q] = [x async for x in _first(adapter.stream_quotes(["XAUUSD"]))]
    quotes.update(q)
    symbols = {s.symbol: s for s in await adapter.symbols()}
    assert symbols["XAUUSD"].margin_per_lot == Decimal("868.40")
    om = OrderManager(
        adapter=adapter,
        verifier=DecisionVerifier(load_public(GATE_PUB)),
        journal=Journal(tmp_path / "j.json"),
        alerts=MemoryAlertSink(),
        clock=clock,
        config=ExecutionConfig(),
        quotes=quotes,
        quality=MemoryQualityLog(),
        account_id="123456",
        symbols=symbols,
    )
    await om.initialize()
    sig = Signal(
        signal_id=uuid4(),
        strategy_id="demo_ma_cross",
        strategy_version="0.1.0",
        symbol="XAUUSD",
        side="buy",
        entry_type="market",
        entry_price=None,
        stop_price=4332.0,
        target_price=None,
        created_at=NOW,
        reason="t",
    )
    it = OrderIntent(
        intent_id=uuid4(), signal=sig, proposed_lots=Decimal("0.10"), risk_fraction=0.005, account_id="123456"
    )
    d = RiskDecision(
        intent_id=it.intent_id,
        verdict="approve",
        approved_lots=Decimal("0.10"),
        reasons=(),
        limits_snapshot_hash="h",
        decided_at=NOW,
        expires_at=NOW + timedelta(seconds=5),
        sequence=1,
    )
    r = await om.execute(it, sign_decision(load_private(GATE_PRIV), d), Timeframe.H1)
    assert r.placed and r.state == "open"
    [pos] = mt5.positions.values()
    assert pos.sl == 4332.0 and pos.comment == client_order_id(it.intent_id)

    class NoRisk:
        def enter_recon_halt(self, reason: str, now: object) -> None:
            raise AssertionError(reason)

        def clear_recon_halt(self) -> bool:
            return False

    recon = Reconciler(om, NoRisk())
    assert (await recon.run_once()).ok
    mt5.bid = 4352.0
    assert await om.close(str(pos.ticket), "test")
    assert (await recon.run_once()).ok  # profit deal explains the new balance
    assert [t.state for t in om.state.orders.values()] == ["closed"]
    await adapter.aclose()


async def _first(it: Any) -> Any:
    async for x in it:
        yield x
        return
