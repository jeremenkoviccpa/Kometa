"""The audit log in Postgres (spec section 16), against a real database migrated with Alembic.

Needs AT_TEST_DATABASE_URL (a role that may create databases). Each run creates and drops its own
database. Skipped locally without it; in CI (env CI set) a missing URL is a failure, not a skip.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from autotrader.cli.main import main
from autotrader.core.alerts import MemoryAlertSink, Severity
from autotrader.core.broker import Quote
from autotrader.core.bus import CONFIG, CONTROL, DECISIONS, HALTS, STAGES, TRADES, InMemoryBus, pump
from autotrader.core.clock import SimClock
from autotrader.core.events import (
    AccountUpdate,
    ConfigChanged,
    Exposure,
    HaltCleared,
    HaltEntered,
    ModelVersionActivated,
    PositionClosed,
    RiskDecided,
    StageChanged,
)
from autotrader.core.ledger import LedgerCorruptError
from autotrader.core.models import HaltState, RiskDecision, Stage, Trade
from autotrader.core.timeutil import utc
from autotrader.monitor.alerts import AlertRouter
from autotrader.monitor.audit import AuditLog, AuditService, check_chain
from autotrader.monitor.pg_ledger import PgLedger, sync_url
from autotrader.monitor.pg_snapshots import PgSnapshots
from autotrader.monitor.state import MonitorService

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "grafana" / "provisioning" / "dashboards" / "json"
URL = os.environ.get("AT_TEST_DATABASE_URL", "")


def _with_db(url: str, db: str) -> str:
    return url.rsplit("/", 1)[0] + "/" + db


@pytest.fixture(scope="module")
def db_url() -> Iterator[str]:
    if not URL:
        if os.environ.get("CI"):
            pytest.fail("AT_TEST_DATABASE_URL is not set in CI: the Postgres audit tests must run")
        pytest.skip("AT_TEST_DATABASE_URL not set")
    name = f"at_test_{uuid4().hex[:12]}"
    admin = sync_url(URL)
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = _with_db(URL, name)
    try:
        env = {**os.environ, "AT_DATABASE_URL": url}
        r = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        yield url
    finally:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def ledger(db_url: str) -> PgLedger:
    tenant = f"t{uuid4().hex[:8]}"  # a fresh chain per test in the shared database
    return PgLedger(db_url, tenant_id=tenant)


def test_chain_appends_and_verifies(ledger: PgLedger) -> None:
    log = AuditLog(ledger)
    log.record("HaltEntered", "risk-gate", {"state": "DAILY_HALT", "reason": "daily loss", "x": 0.1})
    log.record("OrderPlaced", "execution", {"order": {"client_order_id": "at1", "lots": "0.01"}})
    assert log.verify() == 2
    assert [r["kind"] for r in log.tail()] == ["HaltEntered", "OrderPlaced"]
    assert [r["payload"]["actor"] for r in log.tail(event_type="OrderPlaced")] == ["execution"]


def test_database_refuses_update_delete_and_truncate(ledger: PgLedger) -> None:
    AuditLog(ledger).record("x", "t", {"v": 1})
    for sql in (
        "UPDATE audit_log SET actor = 'someone else'",
        "DELETE FROM audit_log",
        "TRUNCATE audit_log",
    ):
        with (
            psycopg.connect(ledger.url) as c,
            pytest.raises(psycopg.errors.RaiseException, match="append-only"),
        ):
            c.execute(sql)
    assert ledger.verify() == 1


def test_application_role_may_only_insert_and_select(ledger: PgLedger) -> None:
    AuditLog(ledger).record("x", "t", {"v": 1})
    with psycopg.connect(ledger.url) as c:
        c.execute("SET ROLE autotrader_app")
        assert c.execute("SELECT count(*) FROM audit_log").fetchone() is not None  # control: may read
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("UPDATE audit_log SET actor = 'x'")


@pytest.mark.parametrize(
    "tamper",
    [
        "UPDATE audit_log SET payload = jsonb_set(payload, '{v}', '2') WHERE event_type = 'b'",
        "UPDATE audit_log SET actor = 'owner' WHERE event_type = 'b'",
        "UPDATE audit_log SET canonical = replace(canonical, '\"v\":1', '\"v\":2') WHERE event_type = 'b'",
        "DELETE FROM audit_log WHERE event_type = 'b'",
    ],
)
def test_verify_catches_a_superuser_edit(ledger: PgLedger, tamper: str) -> None:
    """Even someone who can switch the trigger off cannot change history without `verify` noticing."""
    log = AuditLog(ledger)
    for kind in ("a", "b", "c"):
        log.record(kind, "t", {"v": 1})
    alerts = MemoryAlertSink()
    assert check_chain(log, alerts, utc(2026, 1, 7)) == 3  # control: intact before the edit
    with psycopg.connect(ledger.url) as c:
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
        c.execute(tamper + f" AND tenant_id = '{ledger.tenant_id}'")
        c.execute("ALTER TABLE audit_log ENABLE TRIGGER USER")
    with pytest.raises(LedgerCorruptError):
        ledger.verify()
    assert check_chain(log, alerts, utc(2026, 1, 7)) is None
    assert alerts.kinds(Severity.CRITICAL) == ["audit_chain_break"]


def test_writers_extend_one_chain(ledger: PgLedger) -> None:
    """Two writers (monitor and API) interleaving never fork the chain."""
    other = PgLedger(ledger.url, tenant_id=ledger.tenant_id)
    for i in range(10):
        (ledger if i % 2 else other).append("e", {"actor": "t", "data": {"i": i}})
    assert ledger.verify() == 10


def test_at_audit_verify_on_postgres(
    ledger: PgLedger, db_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger.tenant_id = "default"  # the CLI verifies the default tenant's chain
    with psycopg.connect(ledger.url) as c:
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
        c.execute("DELETE FROM audit_log WHERE tenant_id = 'default'")
        c.execute("ALTER TABLE audit_log ENABLE TRIGGER USER")
    AuditLog(ledger).record("x", "t", {"v": 1})
    monkeypatch.setenv("AT_DATABASE_URL", db_url)
    assert main(["audit", "verify", "--db"]) == 0
    assert "1 records, chain intact" in capsys.readouterr().out
    with psycopg.connect(ledger.url) as c:
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
        c.execute("UPDATE audit_log SET actor = 'owner' WHERE tenant_id = 'default'")
        c.execute("ALTER TABLE audit_log ENABLE TRIGGER USER")
    assert main(["audit", "verify", "--db"]) == 1
    assert "BROKEN" in capsys.readouterr().out


# ---------------------------------------------------------------- dashboards on real data


def _macros(sql: str) -> str:
    """Grafana's $__timeFilter(col) as plain SQL over the last 30 days."""
    return re.sub(
        r"\$__timeFilter\((\w+)\)", r"\1 BETWEEN now() - interval '30 days' AND now() + interval '1 day'", sql
    )


def trade(r: float, t: datetime, shadow: bool) -> Trade:
    return Trade(
        trade_id=f"t-{r}-{shadow}",
        account_id="shadow" if shadow else "acc",
        strategy_id="s",
        strategy_version="1",
        symbol="EURUSD",
        side="buy",
        lots=Decimal("0.1"),
        entry_time=t,
        entry_price=Decimal("1.1"),
        stop_price=Decimal("1.09"),
        exit_time=t + timedelta(minutes=30),
        exit_price=Decimal("1.1"),
        pnl_gross=Decimal(str(r * 100)),
        costs=Decimal(0),
        pnl_net=Decimal(str(r * 100)),
        money_at_risk=Decimal(100),
        r_multiple=r,
        mae=0.0,
        mfe=0.0,
    )


async def _seed(url: str) -> None:
    """Every source a dashboard reads, written the way the running system writes it."""
    now = datetime.now(UTC) - timedelta(minutes=5)
    log = AuditLog(PgLedger(url))
    bus = InMemoryBus()
    clock = SimClock(now)
    router = AlertRouter(clock, audit=log)
    snaps = PgSnapshots(url)
    mon = MonitorService(
        router, clock, snapshots=snaps, contract_size={"EURUSD": Decimal(100000)}, quote_ccy={"EURUSD": "USD"}
    )
    for i, eq in enumerate(("10000", "9900")):
        await mon._account(
            AccountUpdate(
                at=now + timedelta(minutes=i),
                account_id="acc",
                currency="USD",
                balance=Decimal(10000),
                equity=Decimal(eq),
                free_margin=Decimal(eq),
                exposures=(
                    Exposure(
                        symbol="EURUSD",
                        side="buy",
                        lots=Decimal("0.1"),
                        entry=Decimal("1.1"),
                        stop=Decimal("1.09"),
                        strategy_id="s",
                        strategy_version="1",
                        position_id="p",
                    ),
                ),
                quotes=(Quote(symbol="EURUSD", bid=Decimal("1.1"), ask=Decimal("1.1001"), time=now),),
                margin_per_lot={},
            )
        )
    decision = RiskDecision(
        intent_id=uuid4(),
        verdict="resize",
        approved_lots=Decimal("0.05"),
        reasons=("risk fraction capped to stage limit 0.001",),
        limits_snapshot_hash="h",
        decided_at=now,
        expires_at=now + timedelta(seconds=5),
        sequence=1,
    )
    msgs = [
        (STAGES, StageChanged(at=now, strategy_id="s", strategy_version="1", from_stage=Stage.SHADOW,
                              to_stage=Stage.MICRO, reason="shadow exit passed")),
        (STAGES, StageChanged(at=now, strategy_id="s", strategy_version="1", from_stage=Stage.MICRO,
                              to_stage=Stage.SHADOW, reason="demoted: drawdown")),
        (TRADES, PositionClosed(at=now, trade=trade(1.5, now - timedelta(hours=1), shadow=False))),
        (TRADES, PositionClosed(at=now, trade=trade(-1.0, now - timedelta(hours=1), shadow=True))),
        (DECISIONS, RiskDecided(at=now, decision=decision)),
        (HALTS, HaltEntered(at=now, state=HaltState.DAILY_HALT, reason="daily loss")),
        (HALTS, HaltCleared(at=now, previous=HaltState.DAILY_HALT, actor="rollover")),
        (CONFIG, ConfigChanged(at=now, service="risk-gate", name="risk.yaml", path="config/risk.yaml",
                               config_hash="ab" * 32)),
        (CONTROL, ModelVersionActivated(at=now, model_kind="cost_model", model_version="c2")),
    ]  # fmt: skip
    for stream, m in msgs:
        await bus.publish(stream, m)
    await pump(bus, [AuditService(log), mon])
    log.record("OwnerCommand", "owner:api", {"action": "pause_learning", "paused": True})
    with psycopg.connect(sync_url(url)) as c:
        for i in range(3):
            c.execute(
                "INSERT INTO execution_quality (account_id, client_order_id, strategy_id, strategy_version,"
                " symbol, side, order_type, lots, requested_price, filled_price, spread_at_fill, slippage,"
                " latency_ms, filled_at) VALUES ('acc', %s, 's', '1', 'EURUSD', 'buy', 'market', 0.1,"
                " 1.1, 1.1002, 0.0001, 0.0002, %s, %s)",
                (f"at{i:029d}", 20.0 + i, now),
            )
        c.execute(
            "INSERT INTO data_quality_issues (symbol, start_time, end_time, issue_type, severity, detail)"
            " VALUES ('EURUSD', %s, %s, 'gap', 'high', '45 bars')",
            (now - timedelta(hours=2), now - timedelta(hours=1)),
        )


def test_every_dashboard_panel_shows_data_to_the_grafana_role(db_url: str) -> None:
    """Spec section 16: each dashboard shows what it names. Every panel's SQL runs as the read-only role
    Grafana uses, on data written by the real audit service and monitor, and returns rows."""
    import asyncio  # noqa: PLC0415

    asyncio.run(_seed(db_url))
    empty = []
    for f in sorted(DASHBOARD_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        for p in d["panels"]:
            for t in p["targets"]:
                with psycopg.connect(sync_url(db_url)) as c:
                    c.execute("SET ROLE autotrader_readonly")
                    rows = c.execute(_macros(t["rawSql"])).fetchall()
                if not any(all(v is not None for v in r) for r in rows):  # a wrong JSON path reads null
                    empty.append(f"{d['title']} / {p['title']}")
    assert empty == []
    with psycopg.connect(sync_url(db_url)) as c:  # control: the role really is read-only
        c.execute("SET ROLE autotrader_readonly")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("INSERT INTO data_quality_issues (symbol, start_time, end_time, issue_type, severity)"
                      " VALUES ('X', now(), now(), 'gap', 'low')")  # fmt: skip
