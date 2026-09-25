"""API (spec section 16): read-only views, owner-only audited controls, no way to change risk limits."""

from __future__ import annotations

import json
import secrets
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from autotrader.api.app import create_app
from autotrader.core.broker import Quote
from autotrader.core.bus import InMemoryBus, pump
from autotrader.core.clock import SimClock
from autotrader.core.events import AccountUpdate, Exposure
from autotrader.core.hashing import canonical_json
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import HaltState, Stage
from autotrader.core.signing import generate_keypair, load_private, load_public, sign_bytes
from autotrader.core.timeutil import utc
from autotrader.lifecycle.registry import Registry, VersionInfo
from autotrader.monitor.alerts import AlertRouter
from autotrader.monitor.audit import AuditLog
from autotrader.monitor.state import MonitorService
from autotrader.risk.config import load_signed
from autotrader.risk.gate import RiskGate
from autotrader.risk.service import RiskGateService
from autotrader.risk.state import StateStore

ROOT = Path(__file__).resolve().parents[2]
T0 = utc(2026, 1, 7, 12)
TOKEN = secrets.token_urlsafe(32)
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Env:
    def __init__(self, tmp: Path, token: str | None = TOKEN) -> None:
        self.clock = SimClock(T0)
        self.bus = InMemoryBus()
        self.audit = AuditLog(JsonlLedger(tmp / "audit.jsonl"))
        self.router = AlertRouter(self.clock, audit=self.audit)
        self.monitor = MonitorService(self.router, self.clock)
        self.reg = Registry(JsonlLedger(tmp / "reg.jsonl"), self.clock, self.router)
        self.reg.submit_candidate(
            VersionInfo(
                strategy_id="demo_ma_cross",
                version="1.0.0",
                family="d",
                origin="owner",
                demo_only=True,
                code_hash="c",
                created_by="t",
            )
        )
        self.reg.promote_candidate("demo_ma_cross", "1.0.0", validation_passed=False, synthetic=True)
        self.freeze = tmp / "learning.freeze"
        self.app = create_app(
            self.monitor,
            audit=self.audit,
            bus=self.bus,
            clock=self.clock,
            registry=self.reg,
            owner_token=token,
            learning_freeze_path=self.freeze,
            contract_size={"XAUUSD": Decimal(100)},
            quote_ccy={"XAUUSD": "USD"},
            info={"mode": "TEST"},
            research_dir=tmp / "reports",
        )
        self.client = TestClient(self.app)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


async def test_read_endpoints(env: Env) -> None:
    q = Quote(symbol="XAUUSD", bid=Decimal("4341.80"), ask=Decimal("4342.00"), time=T0)
    await env.monitor._account(
        AccountUpdate(
            at=T0,
            account_id="a",
            currency="USD",
            balance=Decimal(20000),
            equity=Decimal(20000),
            free_margin=Decimal(20000),
            quotes=(q,),
            margin_per_lot={},
            exposures=(
                Exposure(
                    symbol="XAUUSD",
                    side="buy",
                    lots=Decimal("0.1"),
                    entry=Decimal(4342),
                    stop=Decimal(4332),
                    strategy_id="s",
                    strategy_version="1",
                    position_id="p",
                ),
            ),
        )
    )
    c = env.client
    st = c.get("/api/status").json()
    assert st["account"]["equity"] == 20000 and st["halt"] == "NORMAL" and st["info"]["mode"] == "TEST"
    assert st["open_risk"] == pytest.approx(100.0)  # 10 x 100 x 0.1
    assert c.get("/api/positions").json()[0]["mark"] == pytest.approx(4341.8)
    assert c.get("/api/versions").json()[0]["stage"] == "shadow"
    assert "Kometa Trading Hub" in c.get("/").text
    for path in ("/api/trades", "/api/equity", "/api/decisions", "/api/alerts", "/api/audit", "/api/summary"):
        assert c.get(path).status_code == 200, path


def test_controls_need_the_owner_token_and_are_audited(env: Env) -> None:
    c = env.client
    body = {"strategy_id": "demo_ma_cross", "version": "1.0.0", "reason": "demo over"}
    assert c.post("/api/control/retire", json=body).status_code == 401
    assert (
        c.post("/api/control/retire", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    )
    assert c.post("/api/control/retire", json=body, headers=AUTH).json() == {"stage": "retired"}
    assert env.reg.get("demo_ma_cross", "1.0.0").stage == Stage.RETIRED
    assert c.post("/api/control/retire", json=body, headers=AUTH).status_code == 409
    assert c.post("/api/control/pause-learning", json={"paused": True}, headers=AUTH).json() == {
        "paused": True
    }
    assert env.freeze.exists()
    actions = [r["payload"]["data"]["action"] for r in env.audit.tail(event_type="OwnerCommand")]
    assert actions == ["retire", "retire", "pause_learning"]


def test_controls_disabled_without_a_token(tmp_path: Path) -> None:
    e = Env(tmp_path, token=None)
    assert (
        e.client.post("/api/control/pause-learning", json={"paused": True}, headers=AUTH).status_code == 403
    )
    with pytest.raises(ValueError, match="at least"):
        Env(tmp_path / "x", token="short")  # noqa: S106


def test_no_endpoint_can_change_risk_limits(env: Env) -> None:
    writes = [
        r.path
        for r in env.app.routes
        if isinstance(r, APIRoute) and (r.methods or set()) & {"POST", "PUT", "PATCH", "DELETE"}
    ]
    assert writes and not [p for p in writes if "risk" in p or "limit" in p]


async def test_full_halt_resume_through_the_api(tmp_path: Path, env: Env) -> None:
    owner_priv, owner_pub = generate_keypair()
    gate_priv, _ = generate_keypair()
    cfg = tmp_path / "risk.yaml"
    cfg.write_text((ROOT / "config" / "risk.yaml").read_text())
    (tmp_path / "risk.yaml.sig").write_text(sign_bytes(load_private(owner_priv), cfg.read_bytes()))
    limits, h = load_signed(cfg, tmp_path / "risk.yaml.sig", load_public(owner_pub))
    gate = RiskGate(
        limits, h, load_private(gate_priv), load_public(owner_pub), StateStore(tmp_path / "rs.json")
    )
    gate.enter_full_halt("peak drawdown", T0)
    risk = RiskGateService(gate, env.bus, env.clock, {})
    await risk.start()
    await pump(env.bus, [risk, env.monitor])
    assert env.monitor.state.halt == HaltState.FULL_HALT

    token = {"action": "resume_full_halt", "nonce": "n1", "issued_at": T0.isoformat(),
             "expires_at": (T0 + timedelta(minutes=15)).isoformat()}  # fmt: skip
    forged = sign_bytes(load_private(generate_keypair()[0]), canonical_json(token).encode())
    assert (
        env.client.post(
            "/api/control/resume", json={"token": token, "signature": forged}, headers=AUTH
        ).status_code
        == 200
    )
    await pump(env.bus, [risk, env.monitor])
    assert gate.state.halt == HaltState.FULL_HALT  # the API only forwards; the gate refused a forged token
    good = sign_bytes(load_private(owner_priv), canonical_json(token).encode())
    env.client.post("/api/control/resume", json={"token": token, "signature": good}, headers=AUTH)
    await pump(env.bus, [risk, env.monitor])
    after = (gate.state.halt.value, env.monitor.state.halt.value)  # mypy narrowed the earlier asserts
    assert after == (HaltState.NORMAL.value, HaltState.NORMAL.value)


def test_research_lists_reports_and_serves_only_their_html(env: Env, tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "swing_1.0.0_abc.json").write_text(
        json.dumps(
            {
                "strategy_id": "swing",
                "version": "1.0.0",
                "passed": False,
                "synthetic": False,
                "checks": [
                    {"name": "oos_trades", "value": 120, "threshold": 200, "passed": False, "op": ">="}
                ],
                "windows": [{"params": {"trend_len": 50}, "test_pf": 1.1}],
                "oos_trades": [{"r_multiple": 1.0}, {"r_multiple": -1.0}],
            }
        )
    )
    (reports / "swing_1.0.0_abc.html").write_text("<html>report</html>")
    (tmp_path / "secret.html").write_text("do not serve")
    [r] = env.client.get("/api/research").json()
    assert r["strategy_id"] == "swing" and r["oos_trades"] == 2 and r["oos_r"] == [1.0, -1.0] and r["html"]
    assert "report" in env.client.get("/research/swing_1.0.0_abc.html").text
    for bad in ("secret", "..%2Fsecret", "nope"):
        assert env.client.get(f"/research/{bad}.html").status_code == 404


def test_patterns_marks_a_bullish_engulfing_on_its_completing_candle(env: Env) -> None:
    c = env.monitor.state.candles
    for i, px in enumerate((100.0, 101.0, 99.5, 99.6)):  # minute 1: bearish, open 100 close 99.6
        c.quote("X", T0 + timedelta(seconds=10 * i), px, px + 0.1)
    for i, px in enumerate((99.4, 99.3, 100.8, 100.6)):  # minute 2: bullish, body 99.4..100.6 engulfs it
        c.quote("X", T0 + timedelta(minutes=1, seconds=10 * i), px, px + 0.1)
    first, hit = env.client.get("/api/patterns?symbol=X&tf=M1").json()
    assert first["names"] == ["inverted_hammer"]  # minute 1 alone: long upper wick, small body at the bottom
    assert "bullish_engulfing" in hit["names"] and hit["bias"] == "bull"
    assert hit["t"] == int((T0 + timedelta(minutes=1)).timestamp())
