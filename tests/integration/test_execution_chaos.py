"""Chaos tests (spec section 19): the real risk gate + order manager + reconciliation on a FakeBroker.

Kill the bridge mid order, drop quotes, duplicate fills, restart services during a halt. The system
must end in a safe, consistent state: every system position has its stop, the journal matches the
broker, nothing was placed twice, nothing larger than approved, and no halt was cleared that should
have stayed.
"""

from __future__ import annotations

import asyncio
import os
import random
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from autotrader.core.alerts import MemoryAlertSink, Severity
from autotrader.core.broker import SymbolInfo, client_order_id, is_system_comment
from autotrader.core.clock import SimClock
from autotrader.core.models import EntryType, HaltState, OrderIntent, Signal, Stage, Timeframe
from autotrader.core.signing import DecisionVerifier, generate_keypair, load_private, load_public, sign_bytes
from autotrader.core.timeutil import utc
from autotrader.data.instruments import load_instruments
from autotrader.execution.config import ExecutionConfig
from autotrader.execution.fake import FakeBroker
from autotrader.execution.journal import Journal
from autotrader.execution.order_manager import ExecResult, OrderManager
from autotrader.execution.quality import MemoryQualityLog
from autotrader.execution.quotes import QuoteBook
from autotrader.execution.reconcile import Reconciler
from autotrader.execution.service import ExecutionService
from autotrader.execution.watchdog import Watchdog
from autotrader.risk.config import load_signed
from autotrader.risk.gate import Exposure, RiskGate, Snapshot
from autotrader.risk.state import StateStore

ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS, _ = load_instruments(ROOT / "config" / "instruments.yaml")
OWNER_PRIV, OWNER_PUB = generate_keypair()
GATE_PRIV, GATE_PUB = generate_keypair()
T0 = utc(2026, 1, 7, 12)  # Wednesday
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


class Stack:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        cfg = tmp / "risk.yaml"
        cfg.write_text((ROOT / "config" / "risk.yaml").read_text())
        (tmp / "risk.yaml.sig").write_text(sign_bytes(load_private(OWNER_PRIV), cfg.read_bytes()))
        self.limits = load_signed(cfg, tmp / "risk.yaml.sig", load_public(OWNER_PUB))
        self.clock = SimClock(T0)
        self.broker = FakeBroker(symbols=[XAU], clock=self.clock)
        self.alerts = MemoryAlertSink()
        self.quotes = QuoteBook()
        self.feed_alive = True
        self.approved: dict[str, Decimal] = {}
        self.last_reasons: tuple[str, ...] = ()
        self.price = Decimal("4342.00")
        self.n = 0
        self.quote()
        self.start()
        self.gate.roll_day(Decimal(10000), T0, new_week=True)

    def start(self) -> None:
        """(Re)start the risk gate and execution from their persisted state."""
        self.gate = RiskGate(
            self.limits[0],
            self.limits[1],
            load_private(GATE_PRIV),
            load_public(OWNER_PUB),
            StateStore(self.tmp / "risk_state.json"),
        )
        self.om = OrderManager(
            adapter=self.broker,
            verifier=DecisionVerifier(load_public(GATE_PUB)),
            journal=Journal(self.tmp / "journal.json"),
            alerts=self.alerts,
            clock=self.clock,
            config=ExecutionConfig(),
            quotes=self.quotes,
            quality=MemoryQualityLog(),
            account_id="acc",
            symbols={"XAUUSD": XAU},
        )
        self.recon = Reconciler(self.om, self.gate)
        self.service = ExecutionService(self.om, self.recon, Watchdog(self.om, self.clock.now()))

    def quote(self, move: str = "0") -> None:
        self.price += Decimal(move)
        was_down, self.broker.down = self.broker.down, False  # the market moves even if the bridge is dead
        self.broker.set_quote("XAUUSD", self.price - Decimal("0.20"), self.price)
        self.broker.down = was_down
        if self.feed_alive:
            self.quotes.update(self.broker.quotes["XAUUSD"])

    def advance(self, s: float) -> None:
        self.clock.advance_to(self.clock.now() + timedelta(seconds=s))

    def snapshot(self) -> Snapshot:
        b = self.broker
        exposures = [
            Exposure(
                symbol=p.symbol,
                side=p.side,
                lots=p.lots,
                entry=p.price_open,
                stop=p.sl,
                strategy_id="s1" if is_system_comment(p.comment) else None,
                external=not is_system_comment(p.comment),
            )
            for p in b.positions.values()
        ] + [
            Exposure(
                symbol=o.symbol,
                side=o.side,
                lots=o.lots,
                entry=o.price,
                stop=o.sl,
                strategy_id="s1",
                pending=True,
            )
            for o in b.orders.values()
        ]
        q = b.quotes["XAUUSD"]
        floating = sum(
            (
                (q.bid - p.price_open if p.side == "buy" else p.price_open - q.ask) * 100 * p.lots
                for p in b.positions.values()
            ),
            Decimal(0),
        )
        equity = b.balance + floating
        return Snapshot(
            now=self.clock.now(),
            equity=equity,
            free_margin=equity,
            bid=q.bid,
            ask=q.ask,
            exposures=exposures,
            instruments=INSTRUMENTS,
            to_account=lambda _ccy: Decimal(1),
            margin_per_lot={"XAUUSD": Decimal(100)},
            stages={("s1", "1.0.0"): Stage.LIVE},
        )

    async def trade(self, entry_type: EntryType = "market", dist: int = 10) -> ExecResult:
        self.n += 1
        side = "buy" if self.n % 2 else "sell"
        sign = Decimal(1) if side == "buy" else Decimal(-1)
        entry = None
        if entry_type == "limit":
            entry = float(self.price - sign * 2)
        ref = Decimal(str(entry)) if entry is not None else self.price
        sig = Signal(
            signal_id=uuid4(),
            strategy_id="s1",
            strategy_version="1.0.0",
            symbol="XAUUSD",
            side=side,
            entry_type=entry_type,
            entry_price=entry,
            stop_price=float(ref - sign * dist),
            target_price=float(ref + sign * dist),
            expiry_bars=3 if entry_type == "limit" else None,
            created_at=self.clock.now(),
            reason="chaos",
        )
        it = OrderIntent(
            intent_id=uuid4(),
            signal=sig,
            proposed_lots=Decimal("0.05"),
            risk_fraction=0.005,
            account_id="acc",
        )
        d = self.gate.decide(it, self.snapshot())
        self.last_reasons = d.reasons
        if d.approved_lots > 0:
            self.approved[client_order_id(it.intent_id)] = d.approved_lots
        return await self.om.execute(it, d, Timeframe.H1)

    async def heal(self) -> None:
        """Faults stop; time passes; periodic jobs run. Everything must converge without the owner."""
        b = self.broker
        b.down = False
        b.ignore_sl_on_fill = False
        b.modify_failures = 0
        b.clear_faults()
        self.feed_alive = True
        for _ in range(3):
            self.advance(61)
            self.quote()
            await self.service.cycle()

    def assert_safe_and_consistent(self) -> None:
        b = self.broker
        system = [p for p in b.positions.values() if is_system_comment(p.comment)]
        # every system position is protected by a broker-side stop
        assert all(p.sl is not None for p in system), "system position without stop"
        # nothing placed twice, nothing larger than approved
        by_coid: dict[str, Decimal] = {}
        for d in b.deal_log:
            if d.entry == "in" and is_system_comment(d.comment):
                by_coid[d.comment] = by_coid.get(d.comment, Decimal(0)) + d.lots
        for coid, lots in by_coid.items():
            assert coid in self.approved, "order reached the broker without an approval"
            assert lots <= self.approved[coid], "filled more than approved"
        assert len({p.comment for p in system}) == len(system)
        # the journal matches the broker
        tracked_open = {t.position_id for t in self.om.state.active() if t.state == "open"}
        tracked_pending = {t.broker_order_id for t in self.om.state.active() if t.state == "pending"}
        assert tracked_open == {p.position_id for p in system}
        assert tracked_pending == {
            o.broker_order_id for o in b.orders.values() if is_system_comment(o.comment)
        }
        assert not [t for t in self.om.state.active() if t.state == "sending"]


def halt_of(g: RiskGate) -> HaltState:
    """Read through a call so mypy does not narrow the state across awaits."""
    return g.state.halt


def run(coro: object) -> None:
    asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------- named scenarios


def test_kill_bridge_mid_order(tmp_path: Path) -> None:
    async def go() -> None:
        s = Stack(tmp_path)
        await s.om.initialize()
        s.broker.fail("place", "after")  # the order lands at the broker ...
        s.broker.fail("open_positions", "before", skip=1)  # ... and the bridge dies before we can look
        r = await s.trade()
        assert r.state == "sending" and len(s.broker.positions) == 1
        s.broker.down = True
        report = await s.recon.run_once()
        assert report.mismatches[0].kind == "broker_unreachable"
        assert halt_of(s.gate) == HaltState.RECON_HALT
        s.advance(1)
        s.quote()
        r2 = await s.trade()  # halted: the gate rejects, nothing is sent
        assert not r2.placed and len(s.broker.positions) == 1
        await s.heal()
        assert halt_of(s.gate) == HaltState.NORMAL
        s.assert_safe_and_consistent()
        assert len(s.broker.positions) == 1

    run(go())


def test_dropped_quotes_block_entries_and_stops_still_protect(tmp_path: Path) -> None:
    async def go() -> None:
        s = Stack(tmp_path)
        await s.om.initialize()
        assert (await s.trade()).placed
        s.feed_alive = False
        s.advance(30)
        s.quote("-1")
        r = await s.trade()
        assert not r.placed and r.reason == "no fresh quote"
        s.quote("-12")  # through the broker-side stop while our feed is dead
        assert not s.broker.positions
        await s.heal()
        s.assert_safe_and_consistent()
        assert halt_of(s.gate) == HaltState.NORMAL

    run(go())


def test_duplicate_fills(tmp_path: Path) -> None:
    async def go() -> None:
        s = Stack(tmp_path)
        await s.om.initialize()
        s.broker.duplicate_deals = True
        assert (await s.trade("limit")).state == "pending"
        s.quote("-3")  # fills the buy limit
        await s.service.cycle()
        await s.service.cycle()
        await s.heal()
        s.assert_safe_and_consistent()
        [t] = s.om.state.orders.values()
        assert t.state == "open"
        assert halt_of(s.gate) == HaltState.NORMAL

    run(go())


def test_restart_during_halt_keeps_halt_and_state(tmp_path: Path) -> None:
    async def go() -> None:
        s = Stack(tmp_path)
        await s.om.initialize()
        assert (await s.trade()).placed
        s.advance(1)
        s.quote()
        assert (await s.trade()).placed
        s.quote("-4")  # both positions under water
        s.gate.state.day_ref_equity = Decimal(10000)
        halt = s.gate.on_account(Decimal(9790), s.clock.now())  # -2.1%: daily halt
        assert halt is not None and halt.close_positions
        s.broker.fail("close_position", "before")  # the first close fails
        await s.om.apply_halt(halt)
        assert "halt_close_failed" in s.alerts.kinds(Severity.CRITICAL)
        # restart everything mid-halt
        s.start()
        assert halt_of(s.gate) == HaltState.DAILY_HALT
        s.advance(1)
        s.quote()
        r = await s.trade()
        assert not r.placed and not s.broker.orders
        # the risk gate re-issues the halt command on restart; execution finishes the job
        await s.om.apply_halt(halt)
        assert not [p for p in s.broker.positions.values() if is_system_comment(p.comment)]
        await s.heal()
        s.assert_safe_and_consistent()
        assert halt_of(s.gate) == HaltState.DAILY_HALT  # reconciliation never clears a loss halt
        s.gate.roll_day(Decimal(9790), s.clock.now(), new_week=False)
        s.advance(1)
        s.quote()
        assert (await s.trade()).placed

    run(go())


def test_restart_between_send_and_ack(tmp_path: Path) -> None:
    async def go() -> None:
        s = Stack(tmp_path)
        await s.om.initialize()
        s.broker.fail("place", "after")
        s.broker.fail("open_positions", "before", skip=1)
        assert (await s.trade()).state == "sending"
        s.start()  # crash + restart: the write-ahead journal remembers the send
        await s.heal()
        s.assert_safe_and_consistent()
        assert [t.state for t in s.om.state.orders.values()] == ["open"]

    run(go())


# ---------------------------------------------------------------- random fault schedules

OPS = [
    *["market"] * 4,  # entries are weighted up so most schedules actually trade (see the events)
    *["limit"] * 2,
    "fail_place_after",
    "fail_place_before",
    "fail_lookup",
    "bridge_down",
    "bridge_up",
    "price_up",
    "price_down",
    "crash_down",
    "drop_sl",
    "modify_fail",
    "dup_deals",
    "feed_dead",
    "feed_alive",
    "cycle",
    "restart",
    "advance",
]


@settings(
    max_examples=int(os.environ.get("CHAOS_EXAMPLES", "60")),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(ops=st.lists(st.sampled_from(OPS), min_size=5, max_size=30))
# found by this test (Phase 5): a restart forgot that reconciliation had halted for an unreachable broker
@example(ops=["market", "market", "bridge_down", "cycle", "restart"])
def test_random_fault_schedules_end_safe_and_consistent(tmp_path_factory: object, ops: list[str]) -> None:
    tmp = tmp_path_factory.mktemp("chaos")  # type: ignore[attr-defined]

    async def go() -> None:
        s = Stack(tmp)
        await s.om.initialize()
        b = s.broker
        placed = 0
        for op in ops:
            s.advance(1)
            if s.feed_alive:
                s.quote()
            if op in ("market", "limit"):
                placed += (await s.trade(op)).placed  # type: ignore[arg-type]
            elif op == "fail_place_after":
                b.fail("place", "after")
            elif op == "fail_place_before":
                b.fail("place", "before")
            elif op == "fail_lookup":
                b.fail("open_positions", "before", skip=len(ops) % 2)
            elif op == "bridge_down":
                b.down = True
            elif op == "bridge_up":
                b.down = False
            elif op == "price_up":
                s.quote("3")
            elif op == "price_down":
                s.quote("-3")
            elif op == "crash_down":
                s.quote("-15")
            elif op == "drop_sl":
                b.ignore_sl_on_fill = True
            elif op == "modify_fail":
                b.modify_failures += 2
            elif op == "dup_deals":
                b.duplicate_deals = True
            elif op == "feed_dead":
                s.feed_alive = False
            elif op == "feed_alive":
                s.feed_alive = True
            elif op == "cycle":
                await s.service.cycle()
            elif op == "restart":
                s.start()
            elif op == "advance":
                s.advance(61)
        await s.heal()
        s.assert_safe_and_consistent()
        assert s.recon.episode == "clean"
        assert halt_of(s.gate) == HaltState.NORMAL
        event(f"orders placed: {min(placed, 3)}{'+' if placed > 3 else ''}")
        event(f"system positions at end: {len(b.positions)}")

    run(go())


# ---------------------------------------------------------------- 48 hour simulated soak


def test_48h_simulated_soak_has_no_unreconciled_states(tmp_path: Path) -> None:
    """Stand-in for the Phase 5 paper run (open question 23) until a demo account and the Windows
    bridge exist: 48 simulated hours, one cycle per minute, an entry every hour, loss halts wired,
    scheduled faults. Every reconciliation must be clean except while the bridge is down."""

    async def go() -> None:
        rng = random.Random(20260925)
        s = Stack(tmp_path)
        await s.om.initialize()
        b = s.broker
        unreconciled: list[str] = []
        entries = 0
        rejections: list[str] = []
        for minute in range(48 * 60):
            s.advance(60)
            s.quote(str(Decimal(rng.choice((-1, 1)) * rng.randint(0, 80)) / 100))
            if minute == 10 * 60:
                b.down = True
            if minute == 10 * 60 + 5:
                b.down = False
            if minute == 20 * 60:
                b.fail("place", "after")
            if minute == 30 * 60:
                b.ignore_sl_on_fill = True
            if minute == 31 * 60:
                b.ignore_sl_on_fill = False
            if minute == 40 * 60:
                b.duplicate_deals = True
            if minute % (24 * 60) == 0 and minute:
                s.gate.roll_day(s.snapshot().equity, s.clock.now(), new_week=False)
            if not b.down:
                halt = s.gate.on_account(s.snapshot().equity, s.clock.now())
                if halt is not None:
                    await s.om.apply_halt(halt)
            if minute % 60 == 30:
                r = await s.trade(dist=4)
                entries += r.placed
                if not r.placed:
                    rejections.append(f"{r.reason}: {'; '.join(s.last_reasons)}")
            report = (await s.service.cycle()).recon
            if report is not None and not report.ok and not b.down:
                unreconciled += [f"{minute}: {m.kind} {m.ref}" for m in report.mismatches]
        await s.heal()
        s.assert_safe_and_consistent()
        assert unreconciled == []
        assert entries >= 20, (entries, rejections)
        # the scheduled faults really happened (a soak that dodges its faults proves nothing)
        crit, info = s.alerts.kinds(Severity.CRITICAL), s.alerts.kinds(Severity.INFO)
        assert "recon_mismatch" in crit and "recon_resolved" in info  # bridge down, then back
        assert "missing_stop" in crit and "stop_repaired" in info
        lost_ack = [c for c in b.calls if c[0] == "place"]
        assert len(lost_ack) == entries  # the lost ack at hour 20 did not cause a second send
        assert s.recon.episode == "clean"
        assert halt_of(s.gate) != HaltState.RECON_HALT

    run(go())
