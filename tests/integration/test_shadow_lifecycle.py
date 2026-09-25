"""Phase 6 acceptance: demo_ma_cross reaches shadow, runs on live quotes, and stays capped.

Quotes -> live bars -> strategy -> ShadowBroker -> trades and signals -> lifecycle evaluator; the
risk gate, reading stages from the registry, refuses every order a shadow version could send.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from autotrader.core.alerts import MemoryAlertSink, Severity
from autotrader.core.broker import Quote
from autotrader.core.clock import SimClock
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import OrderIntent, Stage
from autotrader.core.signing import generate_keypair, load_private, load_public, sign_bytes
from autotrader.core.timeutil import utc
from autotrader.data.instruments import load_instruments
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.costs import InstrumentCosts
from autotrader.engine.shadow import ShadowSession, run_shadow
from autotrader.lifecycle.config import load_promotion_config
from autotrader.lifecycle.evaluator import Evaluator, MemoryStageData
from autotrader.lifecycle.registry import IllegalTransitionError, Registry, VersionInfo
from autotrader.risk.config import load_signed
from autotrader.risk.gate import RiskGate, Snapshot
from autotrader.risk.state import StateStore
from autotrader.strategies_api.loader import load_strategy

ROOT = Path(__file__).resolve().parents[2]
FAST = {"fast": 5, "slow": 20}  # more crosses, so the evaluator's signal minimum is reached


def quotes(days: int) -> list[Quote]:
    df = generate(SyntheticSpec(days=days, seed=8))
    return [
        Quote(symbol="SYNTH", bid=Decimal(str(b)), ask=Decimal(str(a)), time=t + timedelta(seconds=30))
        for t, b, a in df.select("open_time", "bid_c", "ask_c").iter_rows()
    ]


async def feed(qs: list[Quote], clock: SimClock) -> AsyncIterator[Quote]:
    for q in qs:
        clock.advance_to(q.time)
        yield q
        await asyncio.sleep(0)


def test_demo_ma_cross_reaches_shadow_and_stays_capped(tmp_path: Path) -> None:
    async def go() -> None:
        demo = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
        m = demo.manifest
        qs = quotes(40)
        clock = SimClock(qs[0].time)
        alerts = MemoryAlertSink()
        reg = Registry(JsonlLedger(tmp_path / "registry.jsonl"), clock, alerts)
        reg.submit_candidate(
            VersionInfo(
                strategy_id=m.id,
                version=m.version,
                family=m.family,
                origin=m.origin,
                demo_only=m.demo_only,
                code_hash=demo.code_hash,
                params=FAST,
                created_by="owner",
            )
        )
        reg.promote_candidate(m.id, m.version, validation_passed=False, synthetic=True)
        assert reg.get(m.id, m.version).stage == Stage.SHADOW

        data = MemoryStageData()
        ev = Evaluator(reg, load_promotion_config(ROOT / "config" / "promotion.yaml"), data, clock, alerts)
        changes = []

        def on_trade(t):  # type: ignore[no-untyped-def]
            data.add_trade(t)
            changes.append(asyncio.ensure_future(ev.on_trade_closed(t)))

        session = ShadowSession(
            demo.cls,
            {"SYNTH": InstrumentCosts.from_instrument(synthetic_instrument())},
            params=FAST,
            on_signal=lambda s: data.add_signal(s.strategy_id, s.strategy_version, s.created_at),
            on_trade=on_trade,
        )
        await run_shadow(feed(qs, clock), [session], clock, tick_seconds=3600)
        await asyncio.gather(*changes)
        assert await ev.evaluate_all() == []

        v = reg.get(m.id, m.version)
        n_signals = data.signal_count(m.id, m.version, v.stage_since)
        weeks = (clock.now() - v.stage_since) / timedelta(weeks=1)
        # the evaluator really looked: enough weeks and signals to promote a normal version
        assert weeks >= 4 and n_signals >= 30, (weeks, n_signals)
        assert data.trade_rows and all(t.account_id == "shadow" for t in data.trade_rows)
        assert v.stage == Stage.SHADOW  # capped
        assert not [a for a in alerts.alerts if a.kind == "promotion" and "micro" in a.message]
        try:
            reg.transition(m.id, m.version, Stage.MICRO, "force", actor="evaluator")
            raise AssertionError("demo_only left shadow")
        except IllegalTransitionError:
            assert "illegal_transition" in alerts.kinds(Severity.CRITICAL)

        # and the risk gate refuses anything a shadow version could send
        owner_priv, owner_pub = generate_keypair()
        gate_priv, _ = generate_keypair()
        cfg = tmp_path / "risk.yaml"
        cfg.write_text((ROOT / "config" / "risk.yaml").read_text())
        (tmp_path / "risk.yaml.sig").write_text(sign_bytes(load_private(owner_priv), cfg.read_bytes()))
        limits, h = load_signed(cfg, tmp_path / "risk.yaml.sig", load_public(owner_pub))
        gate = RiskGate(
            limits, h, load_private(gate_priv), load_public(owner_pub), StateStore(tmp_path / "rs.json")
        )
        sig = session.signals[-1]
        instruments, _ = load_instruments(ROOT / "config" / "instruments.yaml")
        instruments = {**instruments, "SYNTH": synthetic_instrument()}

        def decide(stages: dict[tuple[str, str], Stage]) -> tuple[str, tuple[str, ...]]:
            it = OrderIntent(
                intent_id=uuid4(),
                signal=sig,
                proposed_lots=Decimal("0.01"),
                risk_fraction=0.001,
                account_id="a",
            )
            snap = Snapshot(
                now=utc(2026, 1, 7, 12),  # a Wednesday: no weekend cutoff
                equity=Decimal(100000),
                free_margin=Decimal(100000),
                bid=Decimal(str(sig.stop_price)) + Decimal(5)
                if sig.side == "buy"
                else Decimal(str(sig.stop_price)) - Decimal(5),
                ask=(
                    Decimal(str(sig.stop_price)) + Decimal(5)
                    if sig.side == "buy"
                    else Decimal(str(sig.stop_price)) - Decimal(5)
                )
                + Decimal("0.01"),
                exposures=(),
                instruments=instruments,
                to_account=lambda _c: Decimal(1),
                margin_per_lot={"SYNTH": Decimal(10)},
                stages=stages,
            )
            d = gate.decide(it, snap)
            return d.verdict, d.reasons

        verdict, reasons = decide(reg.stages())
        assert verdict == "reject" and reasons == ("stage shadow does not trade",), reasons
        # the same intent from a live version passes: the stage alone blocks it
        verdict, reasons = decide({(m.id, m.version): Stage.LIVE})
        assert verdict in ("approve", "resize"), reasons

    asyncio.run(go())
