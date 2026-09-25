"""`at demo run` end to end (headless): the real services on a simulated market for three weeks.

This is what the demo shows, so it must keep working: paper trades at minimum size go through the
signed risk gate, trades come back with R, the audit chain holds, no critical alert fires in a quiet
market, and the dashboard's API answers.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from autotrader.cli.demo import DemoConfig, build, play_sim, stage_of
from autotrader.core.alerts import Severity
from autotrader.core.broker import is_system_comment
from autotrader.core.models import Stage
from autotrader.execution.fake import FakeBroker

ROOT = Path(__file__).resolve().parents[2]


def test_demo_runs_end_to_end(tmp_path: Path) -> None:
    async def go() -> None:
        s = await build(
            DemoConfig(root=ROOT, var=tmp_path / "demo", serve=False, catchup_days=21, run_days=1)
        )
        await play_sim(s, until=s.clock.now() + timedelta(days=21))
        st = s.monitor.state
        # the whole library paper-trades from the start; the plumbing test is parked in shadow
        assert stage_of(s) == Stage.SHADOW
        for sid in ("swing_trend_pullback", "candle_sr_reversal", "scalp_session_breakout"):
            assert stage_of(s, sid) == Stage.DEMO_ONLY
        traded = {t.strategy_id for t in st.trades}
        assert "demo_ma_cross" not in traded and traded, traded
        assert isinstance(s.adapter, FakeBroker)
        entries = [d for d in s.adapter.deal_log if d.entry == "in" and is_system_comment(d.comment)]
        assert len(entries) >= 2 and all(d.lots == Decimal("0.01") for d in entries)  # minimum size only
        assert len(st.trades) >= 2 and all(t.money_at_risk > 0 for t in st.trades)
        assert not [a for _, a in s.router.history if a.severity == Severity.CRITICAL]
        assert s.audit.verify() > 0
        rows = s.audit.tail(100_000)
        kinds = {r["kind"] for r in rows}
        assert {  # spec section 16: signal, proposal, decision, order, fill, close, config
            "SignalEmitted",
            "OrderIntentCreated",
            "RiskDecided",
            "OrderPlaced",
            "OrderModified",
            "OrderFilled",
            "PositionClosed",
            "ConfigChanged",
        } <= kinds
        configs = {r["payload"]["data"]["name"] for r in rows if r["kind"] == "ConfigChanged"}
        assert {"risk.paper.yaml", "instruments.yaml", "execution.yaml", "promotion.yaml"} <= configs
        c = TestClient(s.app("t" * 40))
        status = c.get("/api/status").json()
        assert status["info"]["mode"].startswith("SIMULATED XAUUSD") and status["halt"] == "NORMAL"
        assert set(status["heartbeats"]) >= {"engine", "risk-gate", "execution"}
        cat = {x["strategy_id"]: x for x in c.get("/api/catalog").json()}
        assert (
            cat["scalp_session_breakout"]["trading"] and cat["scalp_session_breakout"]["style"] == "Scalping"
        )
        assert not cat["demo_ma_cross"]["trading"] and cat["demo_ma_cross"]["plumbing"]
        # the owner's Claude tracks: registered with a switch, off by default (each call costs money)
        assert not cat["claude_smc_judge"]["trading"] and not cat["claude_smc_free"]["trading"]
        assert cat["claude_smc_free"]["style"] == "Claude AI"
        ai = c.get("/api/ai").json()
        assert ai["tracks"] == {"claude_smc_free": False, "claude_smc_judge": False}
        auth = {"Authorization": "Bearer " + "t" * 40}
        body = {"strategy_id": "scalp_session_breakout", "version": "1.0.0", "on": False}
        assert c.post("/api/control/paper-trade", json=body).status_code == 401  # owner only
        assert c.post("/api/control/paper-trade", json=body, headers=auth).json() == {"stage": "shadow"}
        assert s.engine.stages[("scalp_session_breakout", "1.0.0")] == Stage.SHADOW  # the engine knows
        on = {**body, "on": True}
        assert c.post("/api/control/paper-trade", json=on, headers=auth).json() == {"stage": "demo_only"}
        assert len(c.get("/api/trades").json()) == len(st.trades)
        assert "Kometa Trading Hub" in c.get("/").text
        # the hub: every pipeline step happened, the feed tells it in order, the chart has prices and trades
        assert all(n > 0 for n in status["pipeline"].values()), status["pipeline"]
        feed = c.get("/api/feed?after=0&limit=500").json()
        seqs = [i["seq"] for i in feed["items"]]
        assert seqs == sorted(seqs) and feed["seq"] == seqs[-1]
        assert c.get(f"/api/feed?after={feed['seq']}").json()["items"] == []
        [mkt] = c.get("/api/market").json()
        assert mkt["symbol"] == "XAUUSD" and len(mkt["mids"]) > 100 and mkt["bid"] < mkt["ask"]
        assert mkt["trades"] and all(t["r"] is not None for t in mkt["trades"])

    asyncio.run(go())
