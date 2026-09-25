"""Integration (spec section 19): engine, allocator, risk gate and execution over the bus.

Quotes -> engine-live -> signal -> allocator -> intent -> risk gate -> signed decision -> execution ->
broker; the account flows back to the risk gate and allocator, closed trades to the lifecycle, and a
demotion and a loss halt travel the other way. Runs over the in-memory bus and over Redis Streams
(fakeredis); both must behave identically.
"""

from __future__ import annotations

import asyncio
import random
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import fakeredis
import pytest

from autotrader.allocator.allocator import Allocator
from autotrader.allocator.config import load_allocator_config
from autotrader.allocator.service import AllocatorService
from autotrader.allocator.weights import VersionPerformance
from autotrader.core.alerts import MemoryAlertSink
from autotrader.core.broker import SymbolInfo, client_order_id, is_system_comment
from autotrader.core.bus import DECISIONS, INTENTS, Bus, InMemoryBus, RedisStreamsBus, Service, pump
from autotrader.core.clock import SimClock
from autotrader.core.events import BarClosed
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import HaltState, Stage, Timeframe
from autotrader.core.profile import BacktestProfile
from autotrader.core.signing import DecisionVerifier, generate_keypair, load_private, load_public, sign_bytes
from autotrader.core.timeutil import utc
from autotrader.data.instruments import load_instruments
from autotrader.engine.service import EngineLiveService
from autotrader.execution.bus_service import ExecutionBusService
from autotrader.execution.config import ExecutionConfig
from autotrader.execution.fake import FakeBroker
from autotrader.execution.journal import Journal
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.quality import MemoryQualityLog
from autotrader.execution.quotes import QuoteBook
from autotrader.execution.watchdog import Watchdog
from autotrader.lifecycle.config import load_promotion_config
from autotrader.lifecycle.evaluator import Evaluator, MemoryStageData
from autotrader.lifecycle.registry import Registry, VersionInfo
from autotrader.lifecycle.service import LifecycleService
from autotrader.risk.config import load_signed
from autotrader.risk.gate import RiskGate
from autotrader.risk.service import RiskGateService
from autotrader.risk.state import StateStore
from autotrader.strategies_api import Request, Strategy, StrategyContext, StrategyManifest

ROOT = Path(__file__).resolve().parents[2]
INSTR, _ = load_instruments(ROOT / "config" / "instruments.yaml")
T0 = utc(2026, 1, 5, 0)  # Monday
XAU = SymbolInfo(
    symbol="XAUUSD",
    digits=2,
    point=Decimal("0.01"),
    contract_size=Decimal(100),
    min_lot=Decimal("0.01"),
    lot_step=Decimal("0.01"),
    max_lot=Decimal(50),
    margin_per_lot=Decimal(100),
)


class EveryThirdBar(Strategy):
    """Test strategy (not a trading method): buy on every third H1 close, 10 below / 10 above."""

    manifest: ClassVar[StrategyManifest] = StrategyManifest.model_validate(
        {
            "id": "pipe1",
            "version": "1.0.0",
            "origin": "owner",
            "family": "pipeline_test",
            "symbols": ["XAUUSD"],
            "timeframes": ["H1"],
            "expected": {"trades_per_month": 100, "win_rate": 0.5, "avg_r": 0},
            "demo_only": True,
        }
    )

    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        return {("XAUUSD", Timeframe.H1): 1}

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        n = ctx.state["n"] = int(ctx.state.get("n", 0)) + 1
        if n % 3:
            return []
        px = event.bar.ask_c
        return [ctx.signal("XAUUSD", "buy", px - 10.0, target_price=px + 10.0, reason="test")]


class Stack:
    def __init__(self, tmp: Path, bus: Bus) -> None:
        self.bus = bus
        self.clock = SimClock(T0)
        self.alerts = MemoryAlertSink()
        owner_priv, owner_pub = generate_keypair()
        gate_priv, gate_pub = generate_keypair()
        cfg = tmp / "risk.yaml"
        cfg.write_text((ROOT / "config" / "risk.yaml").read_text())
        (tmp / "risk.yaml.sig").write_text(sign_bytes(load_private(owner_priv), cfg.read_bytes()))
        limits, h = load_signed(cfg, tmp / "risk.yaml.sig", load_public(owner_pub))
        self.gate = RiskGate(
            limits, h, load_private(gate_priv), load_public(owner_pub), StateStore(tmp / "rs.json")
        )
        self.gate.roll_day(Decimal(20000), T0, new_week=True)
        self.risk = RiskGateService(self.gate, bus, self.clock, INSTR)

        self.broker = FakeBroker(symbols=[XAU], clock=self.clock, balance=Decimal(20000))
        self.price = Decimal("4342.00")
        self.broker.set_quote("XAUUSD", self.price - Decimal("0.20"), self.price)
        self.om = OrderManager(
            adapter=self.broker,
            verifier=DecisionVerifier(load_public(gate_pub)),
            journal=Journal(tmp / "journal.json"),
            alerts=self.alerts,
            clock=self.clock,
            config=ExecutionConfig(),
            quotes=QuoteBook(),
            quality=MemoryQualityLog(),
            account_id="fake-1",
            symbols={"XAUUSD": XAU},
        )
        self.execution = ExecutionBusService(self.om, Watchdog(self.om, T0), bus)

        acfg = load_allocator_config(ROOT / "config" / "allocator.yaml", ROOT / "config" / "promotion.yaml")
        self.allocator = AllocatorService(Allocator(acfg), bus, self.clock, INSTR)
        self.allocator.rebalance(
            [
                VersionPerformance(
                    strategy_id="pipe1",
                    version="1.0.0",
                    stage=Stage.LIVE,
                    live_trades=0,
                    sharpe_live=0,
                    sharpe_backtest=1.0,
                    daily_r={},
                )
            ]
        )

        self.reg = Registry(JsonlLedger(tmp / "registry.jsonl"), self.clock, self.alerts)
        info = VersionInfo(
            strategy_id="pipe1",
            version="1.0.0",
            family="pipeline_test",
            origin="owner",
            demo_only=False,
            code_hash="c",
            created_by="t",
        )
        self.reg.submit_candidate(
            info,
            BacktestProfile(
                strategy_id="pipe1",
                strategy_version="1.0.0",
                trade_r=(1.0, -1.0) * 50,
                weekly_entries=(20,) * 10,
                mc_dd_p95_r=10.0,
                source="t",
                synthetic=False,
            ),
        )
        for st in (Stage.SHADOW, Stage.MICRO, Stage.LIVE):
            self.reg.transition("pipe1", "1.0.0", st, "setup", actor="evaluator")
        self.data = MemoryStageData()
        ev = Evaluator(
            self.reg,
            load_promotion_config(ROOT / "config" / "promotion.yaml"),
            self.data,
            self.clock,
            self.alerts,
        )
        self.lifecycle = LifecycleService(self.reg, ev, self.data, bus, self.clock)
        self.engine = EngineLiveService(bus, self.clock, money=[(EveryThirdBar, {})])
        self.services: list[Service] = [
            self.lifecycle,
            self.risk,
            self.allocator,
            self.execution,
            self.engine,
        ]
        self.rng = random.Random(4)

    def halt(self) -> HaltState:
        return self.gate.state.halt

    async def start(self) -> None:
        await self.om.initialize()
        await self.lifecycle.publish_snapshot()
        await self.risk.start()
        await pump(self.bus, self.services)

    async def minute(self) -> None:
        self.clock.advance_to(self.clock.now() + timedelta(minutes=1))
        self.price += Decimal(self.rng.randint(-100, 100)) / 100
        self.broker.set_quote("XAUUSD", self.price - Decimal("0.20"), self.price)
        await self.execution.on_quote(self.broker.quotes["XAUUSD"])
        await self.execution.cycle()
        await self.engine.on_time(self.clock.now())
        await pump(self.bus, self.services)

    async def run(self, minutes: int) -> None:
        for _ in range(minutes):
            await self.minute()

    async def messages(self, stream: str) -> list[Any]:
        out = []
        while batch := await self.bus.read(stream, "observer", "o"):
            out += [m for _, m in batch]
        return out


def buses() -> list[Any]:
    return [
        pytest.param(InMemoryBus, id="memory"),
        pytest.param(lambda: RedisStreamsBus(fakeredis.FakeAsyncRedis()), id="redis"),
    ]


@pytest.mark.parametrize("make_bus", buses())
def test_pipeline_over_the_bus(tmp_path: Path, make_bus: Any) -> None:
    async def go() -> None:
        s = Stack(tmp_path, make_bus())
        await s.start()
        await s.run(24 * 60)  # one day

        # orders reached the broker only with approved, signed decisions, sized by the allocator
        decisions = [m.decision for m in await s.messages(DECISIONS)]
        intents = {str(m.intent.intent_id): m.intent for m in await s.messages(INTENTS)}
        approved = {client_order_id(d.intent_id): d for d in decisions if d.verdict != "reject"}
        entries = [d for d in s.broker.deal_log if d.entry == "in" and is_system_comment(d.comment)]
        assert len(entries) >= 3
        for e in entries:
            d = approved[e.comment]
            assert e.lots <= d.approved_lots <= intents[str(d.intent_id)].proposed_lots
            stop = Decimal(str(intents[str(d.intent_id)].signal.stop_price))
            at_risk = (e.price - stop) * XAU.contract_size * e.lots
            assert Decimal(50) <= at_risk <= Decimal(100)  # about 0.5% of 20k (live), never above
        assert all(d.signature for d in decisions)
        # closed trades came back to the lifecycle with R
        real = [t for t in s.data.trade_rows if t.account_id == "fake-1"]
        assert real and all(abs(t.r_multiple) < 1.2 for t in real)

        # loss halt: equity falls 3% -> risk gate halts -> execution closes; intents are refused
        s.broker.balance -= Decimal(600)
        await s.run(2)
        assert s.gate.state.halt == HaltState.DAILY_HALT
        assert not [p for p in s.broker.positions.values() if is_system_comment(p.comment)]
        await s.run(6 * 60)
        new = [m.decision for m in await s.messages(DECISIONS)]
        assert new and all(d.verdict == "reject" and d.reasons == ("halted: DAILY_HALT",) for d in new)
        assert not [p for p in s.broker.positions.values() if is_system_comment(p.comment)]
        await s.risk.roll_day(new_week=False)  # 17:00 New York: the daily halt clears
        assert s.halt() == HaltState.NORMAL

        # demotion: losing trades in the journal -> evaluator -> live to micro -> smaller size at once
        for i in range(13):
            s.data.add_trade(real[0].model_copy(update={"trade_id": f"loss{i}", "r_multiple": -1.0}))
        await s.lifecycle.hourly()
        await pump(s.bus, s.services)
        assert s.reg.get("pipe1", "1.0.0").stage == Stage.MICRO
        n_before = len(s.broker.deal_log)
        await s.run(12 * 60)
        later = [d for d in s.broker.deal_log[n_before:] if d.entry == "in" and is_system_comment(d.comment)]
        assert later
        decisions = [m.decision for m in await s.messages(DECISIONS)]
        intents |= {str(m.intent.intent_id): m.intent for m in await s.messages(INTENTS)}
        approved = {client_order_id(d.intent_id): d for d in decisions if d.verdict != "reject"}
        for e in later:
            stop = Decimal(str(intents[str(approved[e.comment].intent_id)].signal.stop_price))
            assert (e.price - stop) * XAU.contract_size * e.lots <= Decimal("19.40")  # micro: 0.1% of 19.4k

    asyncio.run(go())
