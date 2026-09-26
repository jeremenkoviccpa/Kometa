"""Bus (spec section 8): delivery, acknowledgement and redelivery; service fail-closed paths."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import fakeredis
import pytest

from autotrader.allocator.allocator import Allocator
from autotrader.allocator.config import load_allocator_config
from autotrader.allocator.service import AllocatorService
from autotrader.allocator.weights import VersionPerformance
from autotrader.core.broker import Quote
from autotrader.core.bus import (
    ACCOUNT,
    DECISIONS,
    HALTS,
    INTENTS,
    QUOTES,
    SIGNALS,
    STAGES,
    Bus,
    Handler,
    InMemoryBus,
    RedisStreamsBus,
    decode,
    encode,
    pump,
)
from autotrader.core.clock import SimClock
from autotrader.core.events import (
    AccountUpdate,
    BarClosed,
    Heartbeat,
    OrderIntentCreated,
    QuoteUpdate,
    RiskDecided,
    SignalEmitted,
    StageSnapshot,
    StrategyRequestEmitted,
)
from autotrader.core.indicators.sessions import EventIndex
from autotrader.core.models import (
    CalendarEvent,
    CloseRequest,
    HaltState,
    OrderIntent,
    Signal,
    Stage,
    Timeframe,
)
from autotrader.core.signing import generate_keypair, load_private, load_public, sign_bytes
from autotrader.core.timeutil import utc
from autotrader.data.instruments import load_instruments
from autotrader.engine.service import EngineLiveService
from autotrader.risk.config import load_signed
from autotrader.risk.gate import RiskGate
from autotrader.risk.service import RiskGateService
from autotrader.risk.state import StateStore
from autotrader.strategies_api import Request, Strategy, StrategyContext, StrategyManifest

ROOT = Path(__file__).resolve().parents[2]
T0 = utc(2026, 1, 7, 12)


def signal(symbol: str, stop: float) -> Signal:
    return Signal(
        signal_id=uuid4(),
        strategy_id="s",
        strategy_version="1",
        symbol=symbol,
        side="buy",
        entry_type="market",
        entry_price=None,
        stop_price=stop,
        target_price=None,
        created_at=T0,
        reason="r",
    )


def account(quotes: tuple[Quote, ...] = ()) -> AccountUpdate:
    return AccountUpdate(
        at=T0,
        account_id="a",
        currency="USD",
        balance=Decimal(20000),
        equity=Decimal(20000),
        free_margin=Decimal(20000),
        exposures=(),
        quotes=quotes,
        margin_per_lot={},
    )


def hb(i: int) -> Heartbeat:
    return Heartbeat(at=T0 + timedelta(seconds=i), service=f"s{i}")


def make(kind: str) -> Bus:
    return InMemoryBus() if kind == "memory" else RedisStreamsBus(fakeredis.FakeAsyncRedis())


BUSES = ["memory", "redis"]


class Collect:
    def __init__(self, name: str, stream: str, fail_on: int | None = None) -> None:
        self.name = name
        self.stream = stream
        self.got: list[Any] = []
        self.errors: list[str] = []
        self.fail_on = fail_on

    def handlers(self) -> Mapping[str, Handler]:
        async def h(msg: Any) -> None:
            if self.fail_on is not None and len(self.got) == self.fail_on:
                self.got.append("boom")
                raise RuntimeError("poison")
            self.got.append(msg)

        return {self.stream: h}

    def on_handler_error(self, stream: str, mid: str) -> None:
        self.errors.append(mid)


def test_every_message_kind_round_trips() -> None:
    sig = signal("X", 1.0)
    msgs = [
        hb(1),
        SignalEmitted(at=T0, signal=sig, shadow=True),
        StageSnapshot(at=T0, stages=(("s", "1", Stage.LIVE),)),
        StrategyRequestEmitted(
            at=T0, request=CloseRequest(strategy_id="s", strategy_version="1", position_id="p", reason="x")
        ),
        AccountUpdate(
            at=T0,
            account_id="a",
            currency="USD",
            balance=Decimal("1.5"),
            equity=Decimal(2),
            free_margin=Decimal(1),
            exposures=(),
            quotes=(),
            margin_per_lot={"X": Decimal("0.1")},
        ),
    ]
    for m in msgs:
        assert decode(encode(m)) == m


@pytest.mark.parametrize("kind", BUSES)
async def test_each_service_sees_every_message_once(kind: str) -> None:
    bus = make(kind)
    a, b = Collect("a", "hb"), Collect("b", "hb")
    for i in range(5):
        await bus.publish("hb", hb(i))
    assert await pump(bus, [a, b]) == 10
    assert await pump(bus, [a, b]) == 0
    assert [m.service for m in a.got] == [f"s{i}" for i in range(5)] == [m.service for m in b.got]


@pytest.mark.parametrize("kind", BUSES)
async def test_poison_message_is_acked_and_reported(kind: str) -> None:
    bus = make(kind)
    c = Collect("c", "hb", fail_on=1)
    for i in range(3):
        await bus.publish("hb", hb(i))
    await pump(bus, [c])
    assert len(c.got) == 3 and len(c.errors) == 1  # the stream kept moving
    assert await pump(bus, [c]) == 0


async def test_redis_redelivers_unacked_after_a_crash() -> None:
    r = fakeredis.FakeAsyncRedis()
    bus = RedisStreamsBus(r)
    await bus.publish("hb", hb(1))
    await bus.publish("hb", hb(2))
    got = await bus.read("hb", "svc", "svc-1")
    assert len(got) == 2  # read, then the process dies before acking
    restarted = RedisStreamsBus(r)
    again = await restarted.read("hb", "svc", "svc-1")
    assert [m.service for _, m in again] == ["s1", "s2"]
    await restarted.ack("hb", "svc", [mid for mid, _ in again])
    assert await RedisStreamsBus(r).read("hb", "svc", "svc-1") == []


# ---------------------------------------------------------------- risk gate service


def gate_service(tmp: Path, bus: Bus, clock: SimClock) -> RiskGateService:
    owner_priv, owner_pub = generate_keypair()
    gate_priv, _ = generate_keypair()
    cfg = tmp / "risk.yaml"
    cfg.write_text((ROOT / "config" / "risk.yaml").read_text())
    (tmp / "risk.yaml.sig").write_text(sign_bytes(load_private(owner_priv), cfg.read_bytes()))
    limits, h = load_signed(cfg, tmp / "risk.yaml.sig", load_public(owner_pub))
    gate = RiskGate(limits, h, load_private(gate_priv), load_public(owner_pub), StateStore(tmp / "rs.json"))
    instr, _ = load_instruments(ROOT / "config" / "instruments.yaml")
    return RiskGateService(gate, bus, clock, instr)


def intent_msg() -> OrderIntentCreated:
    sig = signal("XAUUSD", 4332.0)
    return OrderIntentCreated(
        at=T0,
        intent=OrderIntent(
            intent_id=uuid4(), signal=sig, proposed_lots=Decimal("0.1"), risk_fraction=0.005, account_id="a"
        ),
    )


async def test_risk_service_rejects_without_fresh_account(tmp_path: Path) -> None:
    bus, clock = InMemoryBus(), SimClock(T0)
    svc = gate_service(tmp_path, bus, clock)
    await bus.publish(INTENTS, intent_msg())
    await pump(bus, [svc])
    [(_, d)] = await bus.read(DECISIONS, "obs", "o")
    assert d.decision.verdict == "reject" and d.decision.reasons == ("no fresh account data",)
    assert d.decision.signature  # a rejection is signed and audited like any decision


async def test_risk_service_republishes_a_persisted_halt_on_start(tmp_path: Path) -> None:
    bus, clock = InMemoryBus(), SimClock(T0)
    svc = gate_service(tmp_path, bus, clock)
    svc.gate.enter_full_halt("test", T0)
    restarted = gate_service(tmp_path, bus, clock)  # new process, same state file
    assert restarted.gate.state.halt == HaltState.FULL_HALT
    await restarted.start()
    [(_, h)] = await bus.read(HALTS, "obs", "o")
    assert h.state == HaltState.FULL_HALT


async def test_shadow_signals_never_become_intents() -> None:

    bus, clock = InMemoryBus(), SimClock(T0)
    acfg = load_allocator_config(ROOT / "config" / "allocator.yaml", ROOT / "config" / "promotion.yaml")
    instr, _ = load_instruments(ROOT / "config" / "instruments.yaml")
    svc = AllocatorService(Allocator(acfg), bus, clock, instr)
    perf = VersionPerformance(
        strategy_id="s",
        version="1",
        stage=Stage.LIVE,
        live_trades=0,
        sharpe_live=0,
        sharpe_backtest=1.0,
        daily_r={},
    )
    svc.rebalance([perf])
    q = Quote(symbol="XAUUSD", bid=Decimal("4341.80"), ask=Decimal("4342.00"), time=T0)
    await bus.publish(STAGES, StageSnapshot(at=T0, stages=(("s", "1", Stage.LIVE),)))
    await bus.publish(
        ACCOUNT,
        account((q,)),
    )
    sig = intent_msg().intent.signal
    await bus.publish(SIGNALS, SignalEmitted(at=T0, signal=sig, shadow=True))
    await pump(bus, [svc])
    assert await bus.read(INTENTS, "obs", "o") == []
    # control: the same signal, not shadow, does become an intent
    await bus.publish(SIGNALS, SignalEmitted(at=T0, signal=sig, shadow=False))
    await pump(bus, [svc])
    assert len(await bus.read(INTENTS, "obs", "o")) == 1


async def test_risk_service_rejects_on_stale_account(tmp_path: Path) -> None:
    bus, clock = InMemoryBus(), SimClock(T0)
    svc = gate_service(tmp_path, bus, clock)
    await bus.publish(
        ACCOUNT,
        account(),
    )
    await pump(bus, [svc])
    clock.advance_to(T0 + timedelta(seconds=11))
    await bus.publish(INTENTS, intent_msg())
    await pump(bus, [svc])
    [(_, d)] = await bus.read(DECISIONS, "obs", "o")
    assert d.decision.reasons == ("no fresh account data",)


async def test_engine_is_silent_unless_the_version_is_in_a_money_stage() -> None:

    class Always(Strategy):
        manifest: ClassVar[StrategyManifest] = StrategyManifest.model_validate(
            {
                "id": "always",
                "version": "1.0.0",
                "origin": "owner",
                "family": "test",
                "symbols": ["XAUUSD"],
                "timeframes": ["M1"],
                "expected": {"trades_per_month": 1, "win_rate": 0.5, "avg_r": 0},
                "demo_only": True,
            }
        )

        def warmup(self) -> dict[tuple[str, Timeframe], int]:
            return {("XAUUSD", Timeframe.M1): 1}

        def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
            return [ctx.signal("XAUUSD", "buy", event.bar.bid_c - 10)]

    async def minutes(bus: Bus, eng: EngineLiveService, n: int, start: int) -> None:
        for i in range(start, start + n):
            await bus.publish(
                QUOTES,
                QuoteUpdate(at=T0 + timedelta(minutes=i, seconds=30), symbol="XAUUSD", bid=100.0, ask=100.2),
            )
            await pump(bus, [eng])

    bus, clock = InMemoryBus(), SimClock(T0)
    eng = EngineLiveService(bus, clock, money=[(Always, {})])
    await bus.publish(STAGES, StageSnapshot(at=T0, stages=(("always", "1.0.0", Stage.SHADOW),)))
    await minutes(bus, eng, 5, 0)
    assert await bus.read(SIGNALS, "obs", "o") == []  # shadow stage: the money runner stays silent
    await bus.publish(STAGES, StageSnapshot(at=T0, stages=(("always", "1.0.0", Stage.LIVE),)))
    await minutes(bus, eng, 5, 5)
    assert len(await bus.read(SIGNALS, "obs", "o")) >= 4

    # the demo: a shadow version shows what it would trade, marked shadow (never sized); retired stays silent
    bus2 = InMemoryBus()
    demo = EngineLiveService(bus2, clock, money=[(Always, {})], shadow_signals=True)
    await bus2.publish(STAGES, StageSnapshot(at=T0, stages=(("always", "1.0.0", Stage.RETIRED),)))
    await minutes(bus2, demo, 5, 0)
    assert await bus2.read(SIGNALS, "obs", "o") == []
    await bus2.publish(STAGES, StageSnapshot(at=T0, stages=(("always", "1.0.0", Stage.SHADOW),)))
    await minutes(bus2, demo, 5, 5)
    shadow = [m for _, m in await bus2.read(SIGNALS, "obs", "o")]
    assert len(shadow) >= 4 and all(isinstance(m, SignalEmitted) and m.shadow for m in shadow)


async def test_risk_service_news_blackout_and_a_stale_calendar_fail_closed(tmp_path: Path) -> None:
    """Check 8 on the bus path: a high-impact USD event 5 minutes away blocks a gold entry; a configured
    calendar that is stale blocks every entry; the control (fresh calendar, quiet market) is approved."""
    q = Quote(symbol="XAUUSD", bid=Decimal("4341.80"), ask=Decimal("4342.00"), time=T0)
    acct = account((q,)).model_copy(update={"margin_per_lot": {"XAUUSD": Decimal(1000)}})
    nfp = CalendarEvent(
        time=T0 + timedelta(minutes=5), currency="USD", impact="high", name="Non-Farm Payrolls"
    )
    later = CalendarEvent(time=T0 + timedelta(hours=5), currency="USD", impact="high", name="FOMC")

    async def decide(news: object, sub: str) -> RiskDecided:
        bus, clock = InMemoryBus(), SimClock(T0)
        (tmp_path / sub).mkdir()
        svc = gate_service(tmp_path / sub, bus, clock)
        svc.news = news  # type: ignore[assignment]
        svc.stages[("s", "1")] = Stage.LIVE
        await bus.publish(ACCOUNT, acct)
        await bus.publish(INTENTS, intent_msg())
        await pump(bus, [svc])
        [(_, d)] = await bus.read(DECISIONS, "obs", "o")
        return d  # type: ignore[no-any-return]

    control = await decide(lambda: EventIndex([later]), "quiet")
    assert control.decision.verdict == "approve", control.decision.reasons
    blackout = await decide(lambda: EventIndex([nfp]), "nfp")
    assert blackout.decision.verdict == "reject" and "news blackout" in blackout.decision.reasons
    stale = await decide(lambda: None, "stale")
    assert stale.decision.reasons == ("economic calendar unavailable or stale",) and stale.decision.signature
    unconfigured = await decide(None, "sim")  # simulated market: no calendar at all
    assert unconfigured.decision.verdict == "approve"
