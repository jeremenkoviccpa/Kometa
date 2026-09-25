"""Execution (spec section 12): order manager, stop confirmation, reconciliation, watchdog, startup."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from autotrader.core.alerts import MemoryAlertSink, Severity
from autotrader.core.broker import SymbolInfo, client_order_id, is_system_comment, magic_number
from autotrader.core.bus import REQUESTS, InMemoryBus, pump
from autotrader.core.clock import SimClock
from autotrader.core.events import StrategyRequestEmitted
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import (
    CloseRequest,
    EntryType,
    Fill,
    HaltCommand,
    HaltState,
    OrderIntent,
    RiskDecision,
    Signal,
    Timeframe,
    Trade,
)
from autotrader.core.signing import (
    DecisionVerifier,
    generate_keypair,
    load_private,
    load_public,
    sign_decision,
)
from autotrader.core.timeutil import utc
from autotrader.execution.bus_service import ExecutionBusService
from autotrader.execution.config import ExecutionConfig, load_execution_config
from autotrader.execution.fake import FakeBroker
from autotrader.execution.journal import Journal, JournalCorruptError
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.quality import JsonlQualityLog, MemoryQualityLog
from autotrader.execution.quotes import QuoteBook
from autotrader.execution.reconcile import Reconciler
from autotrader.execution.service import StartupRefusedError, startup_checks
from autotrader.execution.watchdog import Watchdog
from autotrader.monitor.audit import AuditLog, AuditService

ROOT = Path(__file__).resolve().parents[2]
GATE_PRIV, GATE_PUB = generate_keypair()
OTHER_PRIV, _ = generate_keypair()
T0 = utc(2026, 1, 7, 12)
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


class FakeRisk:
    def __init__(self) -> None:
        self.halt = HaltState.NORMAL
        self.entered: list[str] = []

    def enter_recon_halt(self, reason: str, now: datetime) -> HaltCommand | None:
        if self.halt != HaltState.NORMAL:
            return None
        self.halt = HaltState.RECON_HALT
        self.entered.append(reason)
        return HaltCommand.for_state(HaltState.RECON_HALT, reason, now)

    def current(self) -> HaltState:
        return self.halt

    def clear_recon_halt(self) -> bool:
        if self.halt != HaltState.RECON_HALT:
            return False
        self.halt = HaltState.NORMAL
        return True


class H:
    """Test harness: fake broker, clock, a signing 'risk gate', and an order manager on a journal."""

    def __init__(self, tmp: Path, **broker_kw: object) -> None:
        self.tmp = tmp
        self.clock = SimClock(T0)
        self.broker = FakeBroker(symbols=[XAU], clock=self.clock, **broker_kw)  # type: ignore[arg-type]
        self.alerts = MemoryAlertSink()
        self.quality = MemoryQualityLog()
        self.quotes = QuoteBook()
        self.fills: list[Fill] = []
        self.closed: list[Trade] = []
        self.seq = 0
        self.quote("4341.80", "4342.00")
        self.om = self.new_om()
        self.risk = FakeRisk()
        self.recon = Reconciler(self.om, self.risk)

    def new_om(self) -> OrderManager:
        return OrderManager(
            adapter=self.broker,
            verifier=DecisionVerifier(load_public(GATE_PUB)),
            journal=Journal(self.tmp / "journal.json"),
            alerts=self.alerts,
            clock=self.clock,
            config=ExecutionConfig(),
            quotes=self.quotes,
            quality=self.quality,
            account_id="acc",
            symbols={"XAUUSD": XAU},
            on_fill=self.fills.append,
            on_trade=self.closed.append,
        )

    def quote(self, bid: str, ask: str) -> None:
        self.broker.set_quote("XAUUSD", bid, ask)
        self.quotes.update(self.broker.quotes["XAUUSD"])

    def advance(self, seconds: float) -> None:
        self.clock.advance_to(self.clock.now() + timedelta(seconds=seconds))

    def intent(
        self,
        lots: str = "0.10",
        side: str = "buy",
        stop: float = 4332.0,
        entry_type: EntryType = "market",
        entry: float | None = None,
        expiry_bars: int | None = None,
    ) -> OrderIntent:
        sig = Signal(
            signal_id=uuid4(),
            strategy_id="s1",
            strategy_version="1.0.0",
            symbol="XAUUSD",
            side=side,
            entry_type=entry_type,
            entry_price=entry,
            stop_price=stop,
            target_price=None,
            expiry_bars=expiry_bars,
            created_at=self.clock.now(),
            reason="t",
        )
        return OrderIntent(
            intent_id=uuid4(), signal=sig, proposed_lots=Decimal(lots), risk_fraction=0.005, account_id="acc"
        )

    def decision(
        self, it: OrderIntent, lots: str | None = None, key: str = GATE_PRIV, seq: int | None = None
    ) -> RiskDecision:
        if seq is None:
            self.seq += 1
            seq = self.seq
        now = self.clock.now()
        approved = Decimal(lots) if lots is not None else it.proposed_lots
        d = RiskDecision(
            intent_id=it.intent_id,
            verdict="approve" if approved == it.proposed_lots else ("resize" if approved > 0 else "reject"),
            approved_lots=approved,
            reasons=(),
            limits_snapshot_hash="h",
            decided_at=now,
            expires_at=now + timedelta(seconds=5),
            sequence=seq,
        )
        return sign_decision(load_private(key), d)

    async def enter(self, it: OrderIntent | None = None, lots: str | None = None) -> OrderIntent:
        it = it or self.intent()
        r = await self.om.execute(it, self.decision(it, lots=lots), Timeframe.H1)
        assert r.placed, r.reason
        return it

    def system_positions(self) -> list[str]:
        return [p.position_id for p in self.broker.positions.values() if is_system_comment(p.comment)]


@pytest.fixture
def h(tmp_path: Path) -> H:
    return H(tmp_path)


# ---------------------------------------------------------------- entries


async def test_market_entry_attaches_stop_magic_and_client_id(h: H) -> None:
    it = await h.enter()
    [pos] = h.broker.positions.values()
    assert pos.sl == Decimal("4332.00") and pos.lots == Decimal("0.10")
    assert pos.comment == client_order_id(it.intent_id) and len(pos.comment) == 31
    assert pos.magic == magic_number("s1", "1.0.0")
    t = h.om.state.orders[pos.comment]
    assert t.state == "open" and t.stop_confirmed
    [q] = h.quality.rows
    assert q.requested_price == Decimal("4342.00") and q.filled_price == Decimal("4342.00")
    assert q.slippage == 0 and q.spread_at_fill == Decimal("0.20")
    assert len(h.fills) == 1


async def test_never_sends_more_than_approved(h: H) -> None:
    it = h.intent("0.10")
    r = await h.om.execute(it, h.decision(it, lots="0.05"), Timeframe.H1)
    assert r.placed
    assert [p.lots for p in h.broker.positions.values()] == [Decimal("0.05")]


async def test_decision_larger_than_intent_is_refused(h: H) -> None:
    it = h.intent("0.10")
    # a validly signed decision can still never raise size above the intent
    r = await h.om.execute(it, h.decision(it, lots="0.20"), Timeframe.H1)
    assert not r.placed and not h.broker.positions


@pytest.mark.parametrize("case", ["forged", "expired", "other_intent", "reject", "replayed_sequence"])
async def test_unverified_decisions_never_reach_the_broker(h: H, case: str) -> None:
    it = h.intent()
    if case == "forged":
        d = h.decision(it, key=OTHER_PRIV)
    elif case == "expired":
        d = h.decision(it)
        h.advance(6)
    elif case == "other_intent":
        d = h.decision(h.intent())
    elif case == "reject":
        d = h.decision(it, lots="0")
    else:
        await h.enter()
        d = h.decision(it, seq=h.seq)  # same sequence as the one already used
    n_calls = len([c for c in h.broker.calls if c[0] == "place"])
    r = await h.om.execute(it, d, Timeframe.H1)
    assert not r.placed
    assert len([c for c in h.broker.calls if c[0] == "place"]) == n_calls
    if case == "forged":
        assert "forged_decision" in h.alerts.kinds(Severity.CRITICAL)


async def test_decision_sequence_survives_restart(h: H) -> None:
    await h.enter()
    used = h.seq
    om2 = h.new_om()
    it = h.intent()
    r = await om2.execute(it, h.decision(it, seq=used), Timeframe.H1)
    assert not r.placed and "replayed" in r.reason


async def test_duplicate_intent_places_once(h: H) -> None:
    it = await h.enter()
    r = await h.om.execute(it, h.decision(it), Timeframe.H1)
    assert not r.placed and r.reason == "duplicate intent"
    assert len(h.broker.positions) == 1


async def test_stale_quote_blocks_market_entry(h: H) -> None:
    h.advance(11)
    it = h.intent()
    r = await h.om.execute(it, h.decision(it), Timeframe.H1)
    assert not r.placed and r.reason == "no fresh quote"
    assert "stale_quotes" in h.alerts.kinds(Severity.WARNING)
    assert not h.broker.positions


async def test_lost_ack_is_not_placed_twice(h: H) -> None:
    h.broker.fail("place", "after")  # applied at the broker, answer lost
    it = h.intent()
    r = await h.om.execute(it, h.decision(it), Timeframe.H1)
    assert r.placed and r.reason == "found at broker"
    assert len(h.broker.positions) == 1
    assert len([c for c in h.broker.calls if c[0] == "place"]) == 1


async def test_unanswered_send_stays_sending_then_reconciles_as_failed(h: H) -> None:
    h.broker.fail("place", "before")
    h.broker.fail("place", "before")
    it = h.intent()
    r = await h.om.execute(it, h.decision(it), Timeframe.H1)
    assert not r.placed and r.state == "sending"
    assert "order_unconfirmed" in h.alerts.kinds(Severity.WARNING)
    h.advance(121)
    h.quote("4341.80", "4342.00")
    report = await h.recon.run_once()
    assert report.ok  # resync proved it never reached the broker
    assert h.om.state.orders[client_order_id(it.intent_id)].state == "failed"
    assert h.risk.current() == HaltState.NORMAL and h.risk.entered  # halted, then cleared


async def test_stop_rounded_toward_entry(h: H) -> None:
    await h.enter(h.intent(stop=4332.004))
    [pos] = h.broker.positions.values()
    assert pos.sl == Decimal("4332.01")
    await h.enter(h.intent(side="sell", stop=4351.996))
    sell = next(p for p in h.broker.positions.values() if p.side == "sell")
    assert sell.sl == Decimal("4351.99")


# ---------------------------------------------------------------- stop confirmation


async def test_missing_stop_is_set_again(h: H) -> None:
    h.broker.ignore_sl_on_fill = True
    await h.enter()
    [pos] = h.broker.positions.values()
    assert pos.sl == Decimal("4332.00")
    assert "missing_stop" in h.alerts.kinds(Severity.CRITICAL)
    assert "stop_repaired" in h.alerts.kinds(Severity.INFO)


async def test_stop_that_cannot_be_set_closes_the_position(h: H) -> None:
    h.broker.ignore_sl_on_fill = True
    h.broker.modify_failures = 2
    await h.enter()
    assert not h.broker.positions
    assert h.alerts.kinds(Severity.CRITICAL).count("missing_stop") >= 2
    await h.om.sync_deals()
    assert [t.state for t in h.om.state.orders.values()] == ["closed"]


async def test_tighten_only(h: H) -> None:
    await h.enter()
    [pid] = list(h.broker.positions)
    assert not await h.om.tighten_stop(pid, 4330.0)  # looser: refused
    assert h.broker.positions[pid].sl == Decimal("4332.00")
    assert await h.om.tighten_stop(pid, 4336.0)
    assert h.broker.positions[pid].sl == Decimal("4336.00")


# ---------------------------------------------------------------- pending orders, deals


async def test_pending_limit_fills_via_deals_and_expires(h: H) -> None:
    it = h.intent(entry_type="limit", entry=4340.0, stop=4330.0, expiry_bars=2)
    r = await h.om.execute(it, h.decision(it), Timeframe.H1)
    assert r.state == "pending"
    [o] = h.broker.orders.values()
    assert o.expires_at == T0 + timedelta(hours=2) and o.price == Decimal("4340.00")
    h.quote("4339.70", "4339.90")  # ask through the limit
    assert await h.om.sync_deals() == 1
    t = h.om.state.orders[client_order_id(it.intent_id)]
    assert t.state == "open" and t.stop_confirmed
    assert h.quality.rows[-1].filled_price == Decimal("4340.00")

    it2 = h.intent(entry_type="limit", entry=4330.0, stop=4320.0, expiry_bars=1)
    await h.om.execute(it2, h.decision(it2), Timeframe.H1)
    h.advance(3601)
    assert await h.om.expire_pending() == 1
    assert h.om.state.orders[client_order_id(it2.intent_id)].state == "expired"


async def test_duplicate_deals_count_once(h: H) -> None:
    await h.om.initialize()
    h.broker.duplicate_deals = True
    it = h.intent(entry_type="limit", entry=4340.0, stop=4330.0)
    await h.om.execute(it, h.decision(it), Timeframe.H1)
    h.quote("4339.70", "4339.90")
    await h.om.sync_deals()
    await h.om.sync_deals()
    assert len(h.fills) == 1
    h.quote("4329.00", "4329.20")  # stop hit
    await h.om.sync_deals()
    await h.om.sync_deals()
    t = h.om.state.orders[client_order_id(it.intent_id)]
    assert t.state == "closed" and t.lots == 0
    assert (await h.recon.run_once()).ok  # cash counted once: balance still matches


async def test_halt_cancels_entries_closes_positions_and_leaves_external(h: H) -> None:
    await h.enter()
    it = h.intent(entry_type="limit", entry=4330.0, stop=4320.0)
    await h.om.execute(it, h.decision(it), Timeframe.H1)
    ext = h.broker.open_external("XAUUSD", "sell", Decimal("0.01"))
    await h.om.apply_halt(HaltCommand.for_state(HaltState.DAILY_HALT, "daily loss", h.clock.now()))
    assert not h.broker.orders
    assert list(h.broker.positions) == [ext]
    assert "external_position_open_in_halt" in h.alerts.kinds(Severity.CRITICAL)

    # RECON_HALT cancels entries but keeps positions
    h2 = H(h.tmp / "b")
    await h2.enter()
    await h2.om.apply_halt(HaltCommand.for_state(HaltState.RECON_HALT, "recon", h2.clock.now()))
    assert len(h2.broker.positions) == 1


# ---------------------------------------------------------------- reconciliation


async def test_clean_state_reconciles(h: H) -> None:
    await h.om.initialize()
    await h.enter()
    report = await h.recon.run_once()
    assert report.ok and not h.risk.entered


async def test_external_position_flagged_once_not_halting(h: H) -> None:
    await h.om.initialize()
    h.broker.open_external("XAUUSD", "buy", Decimal("0.01"))
    r1 = await h.recon.run_once()
    r2 = await h.recon.run_once()
    assert r1.ok and r2.ok and len(r1.external) == 1
    assert h.alerts.kinds(Severity.CRITICAL).count("external_position") == 1
    assert h.risk.current() == HaltState.NORMAL


async def test_position_closed_at_broker_by_stop_reconciles(h: H) -> None:
    await h.om.initialize()
    await h.enter()
    h.quote("4331.00", "4331.20")
    report = await h.recon.run_once()
    assert report.ok
    assert [t.state for t in h.om.state.orders.values()] == ["closed"]


async def test_unexplained_missing_position_stays_halted_until_owner(h: H) -> None:
    await h.om.initialize()
    await h.enter()
    h.broker.positions.clear()  # vanished without a deal
    report = await h.recon.run_once()
    assert not report.ok and h.recon.episode == "stuck"
    assert h.risk.current() == HaltState.RECON_HALT
    assert {"recon_mismatch", "recon_stuck"} <= set(h.alerts.kinds(Severity.CRITICAL))
    await h.recon.run_once()
    assert h.risk.current() == HaltState.RECON_HALT  # no automatic clear
    report = await h.recon.owner_resync()
    assert report.ok and h.risk.current() == HaltState.NORMAL


async def test_ack_lost_and_lookup_failed_is_adopted_by_resync(h: H) -> None:
    await h.om.initialize()
    h.broker.fail("open_positions", "before")  # first pre-send lookup fails
    h.broker.fail("place", "after")  # second attempt is applied but unanswered
    it = h.intent()
    r = await h.om.execute(it, h.decision(it), Timeframe.H1)
    assert r.state == "sending" and len(h.broker.positions) == 1
    report = await h.recon.run_once()
    assert report.ok and h.risk.current() == HaltState.NORMAL
    t = h.om.state.orders[client_order_id(it.intent_id)]
    assert t.state == "open" and t.position_id in h.broker.positions


async def test_unreachable_broker_halts_then_clears(h: H) -> None:
    await h.om.initialize()
    h.broker.down = True
    r = await h.recon.run_once()
    assert r.mismatches[0].kind == "broker_unreachable" and h.risk.current() == HaltState.RECON_HALT
    await h.recon.run_once()
    assert h.alerts.kinds(Severity.CRITICAL).count("recon_mismatch") == 1  # not re-alerted every minute
    h.broker.down = False
    r = await h.recon.run_once()
    assert r.ok and h.risk.current() == HaltState.NORMAL


async def test_deposit_is_explained_but_unexplained_balance_halts(h: H) -> None:
    await h.om.initialize()
    h.broker.deposit(Decimal(500))
    assert (await h.recon.run_once()).ok
    assert "balance_operation" in h.alerts.kinds(Severity.WARNING)
    h.broker.balance -= Decimal(100)  # no deal explains it
    r = await h.recon.run_once()
    assert [m.kind for m in r.mismatches] == ["balance"] and h.risk.current() == HaltState.RECON_HALT


async def test_missing_stop_found_by_reconciliation_is_repaired(h: H) -> None:
    await h.om.initialize()
    await h.enter()
    [pid] = list(h.broker.positions)
    h.broker.positions[pid] = h.broker.positions[pid].model_copy(update={"sl": None})
    r = await h.recon.run_once()
    assert r.ok and h.broker.positions[pid].sl == Decimal("4332.00")


# ---------------------------------------------------------------- watchdog


async def test_watchdog_cancels_entries_when_engine_silent(h: H) -> None:
    wd = Watchdog(h.om, started_at=h.clock.now())
    it = h.intent(entry_type="limit", entry=4330.0, stop=4320.0)
    await h.om.execute(it, h.decision(it), Timeframe.H1)
    await h.enter()
    ext = h.broker.open_external("XAUUSD", "buy", Decimal("0.01"))
    h.advance(100)
    wd.beat("engine", h.clock.now())
    wd.beat("risk-gate", h.clock.now())
    h.advance(100)
    assert await wd.check() == []
    h.advance(30)
    wd.beat("risk-gate", h.clock.now())
    assert await wd.check() == ["engine"]
    assert not h.broker.orders and len(h.broker.positions) == 2  # entries cancelled, positions kept
    assert ext in h.broker.positions
    assert await wd.check() == ["engine"]
    assert h.alerts.kinds(Severity.CRITICAL).count("heartbeat_lost") == 1
    wd.beat("engine", h.clock.now())
    assert await wd.check() == []
    assert "heartbeat_restored" in h.alerts.kinds(Severity.INFO)


# ---------------------------------------------------------------- startup, journal, config


@pytest.mark.parametrize(
    ("env", "kw", "err"),
    [
        ("paper", {"server_time_offset_s": 3.0}, "clock skew"),
        ("paper", {"trade_mode": "real"}, "demo account"),
        ("live", {"trade_mode": "demo"}, "real account"),
        ("dev", {"trade_mode": "real"}, "demo account"),
        ("paper", {"margin_mode": "netting"}, "hedging"),
        ("paper", {"account_id": "someone-else"}, "not the configured"),
    ],
)
async def test_startup_refuses(tmp_path: Path, env: str, kw: dict[str, object], err: str) -> None:
    h = H(tmp_path, **kw)
    with pytest.raises(StartupRefusedError, match=err):
        await startup_checks(h.broker, h.clock, env=env, expected_account_id="fake-1", max_skew_seconds=2)


async def test_startup_accepts_demo_for_paper(h: H) -> None:
    acct = await startup_checks(
        h.broker, h.clock, env="paper", expected_account_id="fake-1", max_skew_seconds=2
    )
    assert acct.trade_mode == "demo"


async def test_corrupt_journal_fails_closed(h: H) -> None:
    await h.enter()
    p = h.tmp / "journal.json"
    p.write_text(p.read_text().replace('"open"', '"closed"'))
    with pytest.raises(JournalCorruptError):
        h.new_om()


def test_execution_config_loads(tmp_path: Path) -> None:
    cfg = load_execution_config(ROOT / "config" / "execution.yaml")
    assert cfg == ExecutionConfig()  # the file documents the defaults
    log = JsonlQualityLog(tmp_path / "q.jsonl")
    assert log.path.parent.exists()


# ---------------------------------------------------------------- lifecycle demotions


async def test_demotion_actions_touch_only_that_version(h: H) -> None:
    await h.enter()
    it = h.intent(entry_type="limit", entry=4330.0, stop=4320.0)
    await h.om.execute(it, h.decision(it), Timeframe.H1)
    other = h.intent()
    other = other.model_copy(update={"signal": other.signal.model_copy(update={"strategy_id": "s2"})})
    await h.enter(other)
    await h.om.cancel_pending_of("s1", "1.0.0", "demoted")
    assert not h.broker.orders and len(h.broker.positions) == 2  # one money stage down: stops stay
    await h.om.close_positions_of("s1", "1.0.0", "retired")
    [left] = h.broker.positions.values()
    assert left.magic == magic_number("s2", "1.0.0")


async def test_demotion_action_failure_raises(h: H) -> None:
    it = h.intent(entry_type="limit", entry=4330.0, stop=4320.0)
    await h.om.execute(it, h.decision(it), Timeframe.H1)
    h.broker.orders.clear()  # the broker lost it: cancel is refused
    with pytest.raises(RuntimeError, match="cancel failed"):
        await h.om.cancel_pending_of("s1", "1.0.0", "demoted")


async def test_closed_position_becomes_a_trade_with_r(h: H) -> None:
    h.broker.commission = Decimal(3)  # per lot per side
    await h.enter()  # buy 0.10 at 4342.00, stop 4332.00: 100 USD at risk
    [pid] = list(h.broker.positions)
    await h.om.tighten_stop(pid, 4336.0)  # R stays measured on the initial stop
    h.quote("4352.00", "4352.20")
    assert await h.om.close(pid, "target")
    await h.om.sync_deals()
    [t] = h.closed
    assert t.money_at_risk == Decimal("100.00") and t.stop_price == Decimal("4332.00")
    assert t.pnl_gross == Decimal("100.00") and t.costs == Decimal("0.60")
    assert t.r_multiple == pytest.approx(0.994)
    assert t.exit_price == Decimal("4352.00") and t.strategy_id == "s1"


async def test_strategy_cannot_close_another_strategys_position(h: H) -> None:

    await h.enter()
    [pid] = list(h.broker.positions)
    bus = InMemoryBus()
    svc = ExecutionBusService(h.om, Watchdog(h.om, h.clock.now()), bus)
    foreign = CloseRequest(strategy_id="intruder", strategy_version="1.0.0", position_id=pid, reason="x")
    await bus.publish(REQUESTS, StrategyRequestEmitted(at=h.clock.now(), request=foreign))
    await pump(bus, [svc])
    assert pid in h.broker.positions and "foreign_request" in h.alerts.kinds(Severity.WARNING)
    own = foreign.model_copy(update={"strategy_id": "s1"})
    await bus.publish(REQUESTS, StrategyRequestEmitted(at=h.clock.now(), request=own))
    await pump(bus, [svc])
    assert pid not in h.broker.positions


async def test_every_order_fill_and_cancel_reaches_the_audit_log(h: H, tmp_path: Path) -> None:
    """Spec section 16: every order, fill and cancel is audited. Execution reports each change after the
    journal write; a restart does not report old changes again."""
    bus = InMemoryBus()
    svc = ExecutionBusService(h.om, Watchdog(h.om, h.clock.now()), bus)
    audit = AuditLog(JsonlLedger(tmp_path / "audit.jsonl"))
    it = await h.enter()
    lim = h.intent(entry_type="limit", entry=4330.0, stop=4320.0, expiry_bars=1)
    await h.om.execute(lim, h.decision(lim), Timeframe.H1)
    h.advance(3601)
    await h.om.expire_pending()
    await svc.flush()
    await pump(bus, [AuditService(audit)])
    rows = [(r["kind"], r["payload"]["data"]) for r in audit.tail(100)]
    assert all(r["payload"]["actor"] == "execution" for r in audit.tail(100))
    mkt, pend = client_order_id(it.intent_id), client_order_id(lim.intent_id)

    def of(coid: str) -> list[str]:
        out = []
        for kind, d in rows:
            order = d.get("order") or d.get("fill") or d
            if order.get("client_order_id") == coid:
                out.append(f"{kind}:{order.get('status', '')}".rstrip(":"))
        return out

    assert of(mkt) == ["OrderPlaced:new", "OrderModified:filled", "OrderFilled"]
    assert of(pend) == ["OrderPlaced:new", "OrderModified:placed", "OrderCancelled"]
    assert [d["reason"] for k, d in rows if k == "OrderCancelled"] == ["expired"]
    reported: list[object] = []
    restarted = h.new_om()
    restarted.on_order = reported.append
    restarted.save()
    assert reported == []  # already reported before the restart


async def test_realistic_fake_broker_slips_market_fills_and_stop_outs_adversely(tmp_path: Path) -> None:
    """Demo realism: seeded slippage on market fills and stop-outs, never on limit fills; latency reported."""
    h = H(tmp_path, slippage=lambda _s, _side, spread: spread / 2, latency_ms=lambda: 120.0)
    it = await h.enter()
    [pos] = h.broker.positions.values()
    assert pos.price_open == Decimal("4342.10")  # ask 4342.00 + half the 0.20 spread
    assert h.quality.rows[-1].latency_ms == 120.0
    h.quote("4331.00", "4331.20")  # through the stop at 4332.00
    [out] = [d for d in h.broker.deal_log if d.entry == "out"]
    assert out.price == Decimal("4330.90")  # the sell at 4331.00 slipped by half the spread
    assert client_order_id(it.intent_id) in h.om.state.orders
