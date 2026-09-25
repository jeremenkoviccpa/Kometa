"""Monitoring (spec section 16): alert routing, audit chain, daily summary; every critical alert fires."""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from autotrader.core.alerts import Alert, MemoryAlertSink, Severity
from autotrader.core.broker import SymbolInfo, client_order_id, intent_id_for
from autotrader.core.bus import ALERTS, CONFIG, HALTS, InMemoryBus, pump
from autotrader.core.clock import SimClock
from autotrader.core.configs import config_events, read_config
from autotrader.core.events import (
    AccountUpdate,
    AlertRaised,
    HaltEntered,
    OrderFilled,
    OrderIntentCreated,
    RiskDecided,
    SignalEmitted,
)
from autotrader.core.hashing import sha256_hex
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import Fill, HaltState, OrderIntent, RiskDecision, Signal, Timeframe
from autotrader.core.signing import (
    DecisionVerifier,
    generate_keypair,
    load_private,
    load_public,
    sign_decision,
)
from autotrader.core.timeutil import utc
from autotrader.execution.config import ExecutionConfig
from autotrader.execution.fake import FakeBroker
from autotrader.execution.journal import Journal
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.quality import MemoryQualityLog
from autotrader.execution.quotes import QuoteBook
from autotrader.execution.reconcile import Reconciler
from autotrader.execution.watchdog import Watchdog
from autotrader.monitor.alerts import AlertRouter, TelegramNotifier
from autotrader.monitor.audit import AuditLog, AuditService, check_chain
from autotrader.monitor.candles import Candles
from autotrader.monitor.journey import Journeys
from autotrader.monitor.state import AccountSnapshot, MonitorService, SlippageWatch
from autotrader.risk.config import ConfigSignatureError
from autotrader.risk.service import load_limits_or_alert

ROOT = Path(__file__).resolve().parents[2]
T0 = utc(2026, 1, 7, 12)


class Recorder:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[str] = []

    async def notify(self, text: str) -> bool:
        self.sent.append(text)
        return self.ok


def alert(sev: Severity, kind: str = "k") -> Alert:
    return Alert(severity=sev, kind=kind, message="m", at=T0)


async def test_critical_repeats_every_10_minutes_until_acked() -> None:
    clock = SimClock(T0)
    tg = Recorder()
    r = AlertRouter(clock, primary=tg)
    r.send(alert(Severity.CRITICAL, "halt"))
    assert await r.tick() == 1 and "/ack A1" in tg.sent[0]
    clock.advance_to(T0 + timedelta(minutes=9))
    assert await r.tick() == 0
    clock.advance_to(T0 + timedelta(minutes=10))
    assert await r.tick() == 1  # repeated
    assert r.ack("A1") == 1
    clock.advance_to(T0 + timedelta(minutes=30))
    assert await r.tick() == 0


async def test_non_critical_once_and_fallback_when_telegram_fails() -> None:
    tg, mail = Recorder(ok=False), Recorder()
    r = AlertRouter(SimClock(T0), primary=tg, fallback=mail)
    r.send(alert(Severity.WARNING))
    assert await r.tick() == 1 and len(mail.sent) == 1
    assert await r.tick() == 0


async def test_undeliverable_alert_stays_queued() -> None:
    tg = Recorder(ok=False)
    r = AlertRouter(SimClock(T0), primary=tg)
    r.send(alert(Severity.INFO))
    await r.tick()
    tg.ok = True
    assert await r.tick() == 1


async def test_telegram_send_and_ack_from_owner_chat_only() -> None:
    calls: list[httpx.Request] = []
    by_poll = {  # the stranger alone first: a check that the owner's ack could mask proves nothing
        2: [{"update_id": 5, "message": {"chat": {"id": 999}, "text": "/ack A1"}}],
        3: [{"update_id": 6, "message": {"chat": {"id": 42}, "text": "/ack A1"}}],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if req.url.path.endswith("/getUpdates"):
            polls = sum(1 for c in calls if c.url.path.endswith("/getUpdates"))
            return httpx.Response(200, json={"result": by_poll.get(polls, [])})
        return httpx.Response(200, json={"ok": True})

    tg = TelegramNotifier("123:abc", "42", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = AlertRouter(SimClock(T0), primary=tg)
    r.send(alert(Severity.CRITICAL))
    await r.tick()  # no acks yet: sends
    sent = [c for c in calls if c.url.path.endswith("/sendMessage")]
    assert len(sent) == 1 and json.loads(sent[0].content)["chat_id"] == "42"
    await r.tick()  # a stranger's /ack is ignored
    assert list(r.active) == ["A1"]
    await r.tick()  # the owner's /ack
    assert r.active == {}
    await r.tick()
    assert calls[-1].url.params["offset"] == "7"  # never re-reads old updates


async def test_audit_records_every_message_and_detects_tampering(tmp_path: Path) -> None:
    bus = InMemoryBus()
    log = AuditLog(JsonlLedger(tmp_path / "audit.jsonl"))
    await bus.publish(HALTS, HaltEntered(at=T0, state=HaltState.DAILY_HALT, reason="daily loss"))
    await bus.publish(ALERTS, AlertRaised(at=T0, alert=alert(Severity.WARNING)))
    await pump(bus, [AuditService(log)])
    assert [r["kind"] for r in log.tail()] == ["HaltEntered", "AlertRaised"]
    assert log.tail()[0]["payload"]["actor"] == "risk-gate"
    alerts = MemoryAlertSink()
    assert check_chain(log, alerts, T0) == 2
    p = tmp_path / "audit.jsonl"
    p.write_text(p.read_text().replace("daily loss", "nothing happened"))
    assert check_chain(log, alerts, T0) is None
    assert alerts.kinds(Severity.CRITICAL) == ["audit_chain_break"]


async def test_halts_become_critical_alerts_and_remote_alerts_are_routed() -> None:
    bus = InMemoryBus()
    r = AlertRouter(SimClock(T0))
    mon = MonitorService(r, SimClock(T0))
    await bus.publish(HALTS, HaltEntered(at=T0, state=HaltState.WEEKLY_HALT, reason="weekly loss"))
    await bus.publish(ALERTS, AlertRaised(at=T0, alert=alert(Severity.CRITICAL, "missing_stop")))
    await pump(bus, [mon])
    assert [a.kind for _, a in r.history] == ["halt", "missing_stop"]
    assert mon.state.halt == HaltState.WEEKLY_HALT and len(r.active) == 2


async def test_daily_summary_reports_the_day_and_resets() -> None:
    clock = SimClock(T0)
    r = AlertRouter(clock)
    mon = MonitorService(r, clock)
    upd = AccountUpdate(
        at=T0,
        account_id="a",
        currency="USD",
        balance=Decimal(10000),
        equity=Decimal(10000),
        free_margin=Decimal(10000),
        exposures=(),
        quotes=(),
        margin_per_lot={},
    )
    await mon._account(upd)
    await mon._account(upd.model_copy(update={"at": T0 + timedelta(hours=5), "equity": Decimal(9900)}))
    text = mon.daily_summary(T0 + timedelta(hours=5))
    assert "day P&L -100.00" in text and "drawdown from peak 1.00%" in text
    assert "day P&L +0.00" in mon.summary_text(T0 + timedelta(hours=5))


# ---------------------------------------------------------------- spec: every critical alert fires


class _Risk:
    def __init__(self) -> None:
        self.halted = False

    def enter_recon_halt(self, reason: str, now: object) -> None:
        self.halted = True

    def clear_recon_halt(self) -> bool:
        return False


async def critical_kinds_from_execution(tmp: Path) -> set[str]:
    """Missing stop, heartbeat loss, external position and a reconciliation mismatch, for real."""
    clock = SimClock(T0)
    xau = SymbolInfo(
        symbol="XAUUSD",
        digits=2,
        point=Decimal("0.01"),
        contract_size=Decimal(100),
        min_lot=Decimal("0.01"),
        lot_step=Decimal("0.01"),
        max_lot=Decimal(50),
        margin_per_lot=Decimal(100),
    )
    broker = FakeBroker(symbols=[xau], clock=clock)
    broker.set_quote("XAUUSD", "4341.80", "4342.00")
    quotes = QuoteBook()
    quotes.update(broker.quotes["XAUUSD"])
    priv, pub = generate_keypair()
    sink = MemoryAlertSink()
    om = OrderManager(
        adapter=broker,
        verifier=DecisionVerifier(load_public(pub)),
        journal=Journal(tmp / "j.json"),
        alerts=sink,
        clock=clock,
        config=ExecutionConfig(),
        quotes=quotes,
        quality=MemoryQualityLog(),
        account_id="a",
        symbols={"XAUUSD": xau},
    )
    await om.initialize()
    broker.ignore_sl_on_fill, broker.modify_failures = True, 2  # the broker drops the stop and refuses it
    sig = Signal(
        signal_id=uuid4(),
        strategy_id="s",
        strategy_version="1",
        symbol="XAUUSD",
        side="buy",
        entry_type="market",
        entry_price=None,
        stop_price=4332.0,
        target_price=None,
        created_at=T0,
        reason="r",
    )
    it = OrderIntent(
        intent_id=uuid4(), signal=sig, proposed_lots=Decimal("0.1"), risk_fraction=0.005, account_id="a"
    )
    d = RiskDecision(
        intent_id=it.intent_id,
        verdict="approve",
        approved_lots=Decimal("0.1"),
        reasons=(),
        limits_snapshot_hash="h",
        decided_at=T0,
        expires_at=T0 + timedelta(seconds=5),
        sequence=1,
    )
    await om.execute(it, sign_decision(load_private(priv), d), Timeframe.H1)
    wd = Watchdog(om, T0)
    clock.advance_to(T0 + timedelta(minutes=3))
    await wd.check()
    broker.open_external("XAUUSD", "sell", Decimal("0.01"))
    broker.balance -= Decimal(50)  # unexplained cash: a mismatch
    await Reconciler(om, _Risk()).run_once()
    return set(sink.kinds(Severity.CRITICAL))


SPEC_CRITICAL = {
    "halt",
    "recon_mismatch",
    "missing_stop",
    "heartbeat_lost",
    "bad_config_signature",
    "external_position",
    "audit_chain_break",
}


async def test_every_critical_alert_in_the_spec_fires(tmp_path: Path) -> None:
    """Spec section 16's critical list, each produced by the component that owns it."""
    fired: set[str] = set()
    # halt (monitor)
    r = AlertRouter(SimClock(T0))
    mon = MonitorService(r, SimClock(T0))
    await mon._halt(HaltEntered(at=T0, state=HaltState.DAILY_HALT, reason="x"))
    fired |= {a.kind for _, a in r.history if a.severity == Severity.CRITICAL}
    # bad config signature (risk gate start-up)
    sink = MemoryAlertSink()
    cfg = tmp_path / "risk.yaml"
    cfg.write_text((ROOT / "config" / "risk.yaml").read_text())
    (tmp_path / "risk.yaml.sig").write_text("00" * 64)
    with pytest.raises(ConfigSignatureError):
        load_limits_or_alert(cfg, tmp_path / "risk.yaml.sig", load_public(generate_keypair()[1]), sink, T0)
    fired |= set(sink.kinds(Severity.CRITICAL))
    # audit chain break (monitor job)
    log = AuditLog(JsonlLedger(tmp_path / "a.jsonl"))
    log.record("x", "t", {"v": 1})
    (tmp_path / "a.jsonl").write_text((tmp_path / "a.jsonl").read_text().replace('"v":1', '"v":2'))
    check_chain(log, sink, T0)
    fired |= set(sink.kinds(Severity.CRITICAL))
    # reconciliation, missing stop, heartbeat, external position (execution)
    fired |= await critical_kinds_from_execution(tmp_path / "exec")
    assert fired >= SPEC_CRITICAL, SPEC_CRITICAL - fired


async def test_config_hashes_are_audited_with_the_reading_service(tmp_path: Path) -> None:
    cfg = tmp_path / "execution.yaml"
    cfg.write_text("stop_confirm_attempts: 3\n")
    read_config(cfg)
    [e] = [x for x in config_events("execution", T0) if x.name == "execution.yaml"]
    assert e.config_hash == sha256_hex(cfg.read_bytes()) and e.path == str(cfg)
    bus = InMemoryBus()
    log = AuditLog(JsonlLedger(tmp_path / "audit.jsonl"))
    await bus.publish(CONFIG, e)
    await pump(bus, [AuditService(log)])
    [row] = log.tail(event_type="ConfigChanged")
    assert row["payload"]["actor"] == "execution"
    assert row["payload"]["data"]["config_hash"] == e.config_hash


def fill(slip: str, side: str = "buy", spread: str = "0.20") -> Fill:
    req = Decimal("4342.00")
    price = req + Decimal(slip) if side == "buy" else req - Decimal(slip)
    return Fill(client_order_id="c", account_id="a", symbol="XAUUSD", side=side, price=price,
                lots=Decimal("0.1"), commission=Decimal(0), spread_at_fill=Decimal(spread),
                requested_price=req, latency_ms=5.0, filled_at=T0)  # fmt: skip


async def test_slippage_above_model_warns_once_per_breach() -> None:
    """Model: 0.2 x spread, adverse. 0.20 spread -> 0.04 modelled per fill."""
    r = AlertRouter(SimClock(T0))
    mon = MonitorService(r, SimClock(T0), slippage=SlippageWatch(mult=0.2, window=5))

    async def feed(*fs: Fill) -> list[str]:
        for f in fs:
            await mon._order(OrderFilled(at=T0, fill=f))
        return [a.kind for _, a in r.history]

    assert await feed(*[fill("0.04")] * 5, *[fill("0.04", "sell")] * 5) == []  # control: at the model
    assert await feed(*[fill("0.10")] * 5) == ["slippage_above_model"]  # above: warned
    assert await feed(*[fill("0.10")] * 5) == ["slippage_above_model"]  # still above: not again
    assert await feed(*[fill("0.00")] * 5, *[fill("0.10")] * 5) == ["slippage_above_model"] * 2  # re-armed
    assert await feed(fill("-0.50", "sell")) == ["slippage_above_model"] * 2  # price improvement is fine


async def test_snapshots_once_a_minute_and_a_failed_write_only_warns() -> None:
    r = AlertRouter(SimClock(T0))
    rows: list[AccountSnapshot] = []
    mon = MonitorService(r, SimClock(T0), snapshots=rows.append)
    upd = AccountUpdate(at=T0, account_id="a", currency="USD", balance=Decimal(10000), equity=Decimal(10000),
                        free_margin=Decimal(10000), exposures=(), quotes=(), margin_per_lot={})  # fmt: skip
    for sec in (0, 20, 40, 60):
        await mon._account(upd.model_copy(update={"at": T0 + timedelta(seconds=sec)}))
    assert [s.at for s in rows] == [T0, T0 + timedelta(minutes=1)] and rows[0].open_risk == 0

    def broken(_: AccountSnapshot) -> None:
        raise OSError("database down")

    mon.snapshots = broken
    await mon._account(upd.model_copy(update={"at": T0 + timedelta(minutes=2), "equity": Decimal(9000)}))
    assert [a.kind for _, a in r.history] == ["snapshot_write_failed"]
    assert mon.state.account is not None and mon.state.account.equity == Decimal(9000)  # kept going


async def test_feed_tells_the_pipeline_in_order_and_skips_state() -> None:
    mon = MonitorService(AlertRouter(SimClock(T0)), SimClock(T0))
    bus = InMemoryBus()
    await bus.publish(HALTS, HaltEntered(at=T0, state=HaltState.DAILY_HALT, reason="daily loss"))
    await bus.publish(ALERTS, AlertRaised(at=T0, alert=alert(Severity.WARNING, "data_quality")))
    await pump(bus, [mon])
    await mon._account(
        AccountUpdate(at=T0, account_id="a", currency="USD", balance=Decimal(1), equity=Decimal(1),
                      free_margin=Decimal(1), exposures=(), quotes=(), margin_per_lot={})
    )  # fmt: skip
    items = mon.state.feed.after(0)
    assert [(i.step, i.tone) for i in items] == [("halt", "bad"), ("alert", "warn")]  # account: state only
    assert "DAILY_HALT" in items[0].text and mon.state.feed.after(items[-1].seq) == []


def test_candles_align_like_the_strategies_bars_and_never_rewrite_the_past() -> None:
    c = Candles()
    t = utc(2026, 1, 7, 21, 58, 10)  # 16:58 New York: the D1 bucket closes at 17:00
    for i, px in enumerate((10.0, 12.0, 9.0, 11.0)):
        c.quote("X", t + timedelta(seconds=10 * i), px, px + 0.1)
    c.quote("X", t + timedelta(minutes=3), 13.0, 13.1)  # 17:01 New York: next trading day
    c.quote("X", t, 99.0, 99.1)  # a late quote for a passed minute is dropped
    [m1a, _] = c.series("X", Timeframe.M1, 10)
    assert (m1a.o, m1a.h, m1a.l, m1a.c, m1a.ticks) == (10.0, 12.0, 9.0, 11.0, 4)
    days = c.series("X", Timeframe.D1, 10)
    assert len(days) == 2 and days[0].c == 11.0 and days[1].o == 13.0  # split at 17:00 New York


def test_journey_joins_every_step_of_one_signal() -> None:
    sig = Signal(signal_id=uuid4(), strategy_id="s", strategy_version="1", symbol="XAUUSD", side="buy",
                 entry_type="market", entry_price=None, stop_price=4332.0, target_price=None, created_at=T0,
                 reason="cross")  # fmt: skip
    iid = intent_id_for(sig.signal_id)
    coid = client_order_id(iid)
    it = OrderIntent(
        intent_id=iid, signal=sig, proposed_lots=Decimal("0.1"), risk_fraction=0.005, account_id="a"
    )
    d = RiskDecision(intent_id=iid, verdict="approve", approved_lots=Decimal("0.1"), reasons=(),
                     limits_snapshot_hash="h", decided_at=T0, expires_at=T0, sequence=7)  # fmt: skip
    f = Fill(client_order_id=coid, account_id="a", symbol="XAUUSD", side="buy", price=Decimal("4342.10"),
             lots=Decimal("0.1"), commission=Decimal("-0.35"), spread_at_fill=Decimal("0.20"),
             requested_price=Decimal("4342.00"), latency_ms=80.0, filled_at=T0)  # fmt: skip
    j = Journeys(slippage_mult=0.2)
    j.add(SignalEmitted(at=T0, signal=sig))
    j.add(
        SignalEmitted(at=T0, signal=sig.model_copy(update={"signal_id": uuid4()}), shadow=True)
    )  # no journey
    j.add(OrderIntentCreated(at=T0, intent=it))
    j.add(RiskDecided(at=T0, decision=d))
    j.add(OrderFilled(at=T0, fill=f))
    [one] = j.summaries(10)
    assert one.ref == coid and [s.step for s in one.steps] == ["signal", "intent", "decision", "fill"]
    assert one.steps[2].details["reasons"] == "all 9 checks passed" and one.status == "approved"
    [row] = j.fills
    assert row.slippage == pytest.approx(0.10) and row.modelled == pytest.approx(0.04)  # worse than the model
