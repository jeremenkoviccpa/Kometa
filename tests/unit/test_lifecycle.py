"""Lifecycle (spec section 10): registry, state machine, evaluator promotion and demotion scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from autotrader.core.alerts import MemoryAlertSink, Severity
from autotrader.core.broker import ExecutionQuality
from autotrader.core.clock import SimClock
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import Stage, Trade
from autotrader.core.profile import BacktestProfile
from autotrader.core.timeutil import utc
from autotrader.lifecycle.config import PromotionConfig, load_promotion_config
from autotrader.lifecycle.evaluator import Evaluator, MemoryStageData
from autotrader.lifecycle.registry import IllegalTransitionError, Registry, VersionInfo

ROOT = Path(__file__).resolve().parents[2]
CFG = load_promotion_config(ROOT / "config" / "promotion.yaml")
T0 = utc(2026, 1, 5, 12)  # a Monday
RNG = np.random.default_rng(1)
BT_R = tuple(float(x) for x in RNG.normal(0.2, 1.0, 400))  # the "backtest": avg 0.2R, about 55% winners
PROFILE_KW = {
    "trade_r": BT_R,
    "weekly_entries": (5, 4, 6, 5, 5, 4, 6, 5, 5, 5, 4, 6),
    "mc_dd_p95_r": 8.0,
    "model_slippage": {"EURUSD": 0.00002},
    "source": "test",
    "synthetic": False,
}


def info(sid: str = "s1", demo: bool = False) -> VersionInfo:
    return VersionInfo(
        strategy_id=sid,
        version="1.0.0",
        family=sid,
        origin="owner",
        demo_only=demo,
        code_hash="c",
        created_by="t",
    )


def profile(sid: str = "s1") -> BacktestProfile:
    return BacktestProfile(strategy_id=sid, strategy_version="1.0.0", **PROFILE_KW)


@dataclass
class Actions:
    calls: list[tuple[str, str]] = field(default_factory=list)
    fail: bool = False

    async def cancel_pending_of(self, strategy_id: str, version: str, reason: str) -> None:
        self.calls.append(("cancel", strategy_id))
        if self.fail:
            raise RuntimeError("bridge down")

    async def close_positions_of(self, strategy_id: str, version: str, reason: str) -> None:
        self.calls.append(("close", strategy_id))


class H:
    def __init__(self, tmp: Path, cfg: PromotionConfig = CFG) -> None:
        self.tmp = tmp
        self.clock = SimClock(T0)
        self.alerts = MemoryAlertSink()
        self.reg = Registry(JsonlLedger(tmp / "registry.jsonl"), self.clock, self.alerts)
        self.data = MemoryStageData()
        self.actions = Actions()
        self.ev = Evaluator(self.reg, cfg, self.data, self.clock, self.alerts, self.actions)

    def advance(self, **kw: float) -> None:
        self.clock.advance_to(self.clock.now() + timedelta(**kw))

    def new(self, sid: str = "s1", demo: bool = False, prof: bool = True) -> None:
        self.reg.submit_candidate(info(sid, demo), profile(sid) if prof else None)

    def to(self, sid: str, *stages: Stage) -> None:
        for s in stages:
            self.reg.transition(sid, "1.0.0", s, "test", actor="evaluator")

    def trades(
        self, sid: str, rs: list[float], *, shadow: bool, start: datetime | None = None, every_h: float = 4
    ) -> None:
        t = start or self.clock.now()
        for r in rs:
            self.data.add_trade(trade(sid, r, t, shadow))
            t += timedelta(hours=every_h)

    def stage(self, sid: str = "s1") -> Stage:
        return self.reg.get(sid, "1.0.0").stage


def trade(sid: str, r: float, t: datetime, shadow: bool) -> Trade:
    mar = Decimal(100)
    return Trade(
        trade_id=f"{sid}-{t.isoformat()}",
        account_id="shadow" if shadow else "acc",
        strategy_id=sid,
        strategy_version="1.0.0",
        symbol="EURUSD",
        side="buy",
        lots=Decimal("0.1"),
        entry_time=t,
        entry_price=Decimal("1.1"),
        stop_price=Decimal("1.09"),
        exit_time=t + timedelta(hours=1),
        exit_price=Decimal("1.1"),
        pnl_gross=Decimal(str(round(r * 100, 6))),
        costs=Decimal(0),
        pnl_net=Decimal(str(round(r * 100, 6))),
        money_at_risk=mar,
        r_multiple=r,
        mae=0.0,
        mfe=0.0,
    )


def fills(sid: str, n: int, slip: float, t: datetime) -> list[ExecutionQuality]:
    return [
        ExecutionQuality(
            client_order_id=f"at{i:029d}",
            account_id="acc",
            strategy_id=sid,
            strategy_version="1.0.0",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            lots=Decimal("0.1"),
            requested_price=Decimal("1.1"),
            filled_price=Decimal("1.1") + Decimal(str(slip)),
            spread_at_fill=Decimal("0.0001"),
            slippage=Decimal(str(slip)),
            latency_ms=5,
            filled_at=t + timedelta(minutes=i),
        )
        for i in range(n)
    ]


def typical(n: int, shift: float = 0.0) -> list[float]:
    """n evenly spaced quantiles of the backtest R: a sample that looks exactly like the backtest."""
    q = np.quantile(np.asarray(BT_R), (np.arange(n) + 0.5) / n)
    order = np.random.default_rng(n).permutation(n)  # interleave winners and losers
    return [float(x) + shift for x in q[order]]


def sample(n: int, seed: int, shift: float = 0.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) + shift for x in rng.choice(np.asarray(BT_R), n)]


@pytest.fixture
def h(tmp_path: Path) -> H:
    return H(tmp_path)


# ---------------------------------------------------------------- registry and state machine


def test_submit_starts_as_candidate_and_versions_are_immutable(h: H) -> None:
    h.new()
    assert h.stage() == Stage.CANDIDATE
    with pytest.raises(ValueError, match="immutable"):
        h.new()


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        ([], Stage.LIVE),
        ([], Stage.MICRO),
        ([Stage.SHADOW], Stage.LIVE),
        ([Stage.SHADOW], Stage.SCALED),
        ([Stage.SHADOW, Stage.MICRO], Stage.SCALED),
        ([Stage.SHADOW, Stage.MICRO, Stage.LIVE], Stage.SHADOW),
        ([Stage.SHADOW, Stage.RETIRED], Stage.SHADOW),
    ],
)
def test_illegal_transitions_raise_and_alert(h: H, path: list[Stage], bad: Stage) -> None:
    h.new()
    h.to("s1", *path)
    with pytest.raises(IllegalTransitionError):
        h.to("s1", bad)
    assert "illegal_transition" in h.alerts.kinds(Severity.CRITICAL)


def test_full_ladder_and_owner_retire(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE, Stage.SCALED, Stage.LIVE, Stage.MICRO)
    h.reg.retire("s1", "1.0.0", "owner says so")
    assert h.stage() == Stage.RETIRED
    with pytest.raises(IllegalTransitionError):
        h.reg.retire("s1", "1.0.0", "twice")


def test_candidate_needs_a_profile_for_shadow(h: H) -> None:
    h.new(prof=False)
    with pytest.raises(IllegalTransitionError):
        h.to("s1", Stage.SHADOW)


def test_validation_verdicts(h: H) -> None:
    for sid, passed, synth, want in [
        ("a", True, False, Stage.SHADOW),
        ("b", False, False, Stage.RETIRED),
        ("c", True, True, Stage.RETIRED),
    ]:
        h.new(sid)
        h.reg.promote_candidate(sid, "1.0.0", validation_passed=passed, synthetic=synth)
        assert h.stage(sid) == want


def test_registry_replays_from_ledger(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW)
    h.advance(days=1)
    h.to("s1", Stage.MICRO)
    again = Registry(JsonlLedger(h.tmp / "registry.jsonl"), h.clock, MemoryAlertSink())
    v = again.get("s1", "1.0.0")
    assert v.stage == Stage.MICRO and v.stage_since == T0 + timedelta(days=1) and v.profile == profile()
    assert [r.to_stage for r in v.history] == [Stage.CANDIDATE, Stage.SHADOW, Stage.MICRO]
    assert JsonlLedger(h.tmp / "registry.jsonl").verify() == 4


# ---------------------------------------------------------------- demo_ma_cross stays capped


async def test_demo_only_reaches_shadow_and_stays_capped(h: H) -> None:
    h.new("demo_ma_cross", demo=True, prof=False)
    h.reg.promote_candidate("demo_ma_cross", "1.0.0", validation_passed=False, synthetic=True)
    assert h.stage("demo_ma_cross") == Stage.SHADOW
    for i in range(60):
        h.data.add_signal("demo_ma_cross", "1.0.0", T0 + timedelta(hours=i))
    h.trades("demo_ma_cross", sample(40, 3), shadow=True)
    h.advance(weeks=8)
    assert await h.ev.evaluate_all() == []
    assert h.stage("demo_ma_cross") == Stage.SHADOW
    with pytest.raises(IllegalTransitionError):
        h.to("demo_ma_cross", Stage.MICRO)
    h.reg.retire("demo_ma_cross", "1.0.0", "owner")  # retiring is always allowed


# ---------------------------------------------------------------- shadow exit


async def test_shadow_promotes_when_inside_band(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW)
    for i in range(32):
        h.data.add_signal("s1", "1.0.0", T0 + timedelta(hours=5 * i))
    h.trades("s1", typical(25), shadow=True, every_h=26)  # 25 entries over 5 weeks = 5 per week
    h.advance(weeks=3)
    assert await h.ev.evaluate("s1", "1.0.0") is None  # too early
    h.advance(weeks=2)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.MICRO
    rec = h.reg.get("s1", "1.0.0").history[-1]
    assert rec.metrics["trades"] == 25 and "rate_band_lo" in rec.metrics


@pytest.mark.parametrize("why", ["losing", "too_few_entries"])
async def test_shadow_that_diverges_is_retired(h: H, why: str) -> None:
    h.new()
    h.to("s1", Stage.SHADOW)
    for i in range(32):
        h.data.add_signal("s1", "1.0.0", T0 + timedelta(hours=5 * i))
    if why == "losing":
        h.trades("s1", typical(25, shift=-1.0), shadow=True, every_h=26)
    else:
        h.trades("s1", typical(3), shadow=True, every_h=26)
    h.advance(weeks=5)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.RETIRED and "diverges" in ev.reason


async def test_shadow_trades_never_count_as_money_trades(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO)
    h.trades("s1", sample(80, 5), shadow=True)  # shadow rows only: micro has no real trades yet
    h.advance(weeks=4)
    assert await h.ev.evaluate("s1", "1.0.0") is None


# ---------------------------------------------------------------- money stages


async def test_micro_to_live_and_weekly_promotion_limit(h: H) -> None:
    for sid in ("a", "b", "c"):
        h.new(sid)
        h.to(sid, Stage.SHADOW, Stage.MICRO)
        h.trades(sid, typical(65), shadow=False)
        h.data.fill_rows += fills(sid, 25, 0.00002, T0)
    h.advance(weeks=3)
    results = [await h.ev.evaluate(sid, "1.0.0") for sid in ("a", "b", "c")]
    assert [r.to_stage if r else None for r in results] == [Stage.LIVE, Stage.LIVE, None]  # max 2 per week
    h.advance(days=8)
    ev = await h.ev.evaluate("c", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.LIVE


async def test_live_to_scaled_then_back_when_sharpe_slips(h: H) -> None:
    # isolate the Sharpe rule: PF and drawdown rules relaxed for this scenario
    relaxed = CFG.demotion.model_copy(update={"rolling_pf_min": 0.01, "dd_vs_mc_p95": 100.0})
    h = H(h.tmp / "x", CFG.model_copy(update={"demotion": relaxed}))
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE)
    h.trades("s1", typical(160), shadow=False, every_h=6)  # behaves like the backtest: Sharpe well above 1
    h.advance(weeks=6)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.SCALED
    h.trades("s1", typical(160, shift=-0.2), shadow=False, every_h=6)  # same shape, no edge
    h.advance(weeks=6)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.LIVE and "Sharpe" in ev.reason, ev
    assert h.actions.calls == [("cancel", "s1")]  # one money stage down: positions keep their stops


async def test_outperforming_the_backtest_is_drift_too(h: H) -> None:
    """Spec: outside the backtest interval on either side. A version that wins far more than its
    backtest is not the strategy that was validated."""
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE)
    h.trades("s1", [abs(r) + 0.3 for r in typical(60)], shadow=False)
    h.advance(weeks=3)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.MICRO and "outside backtest band" in ev.reason


async def test_live_demotion_by_profit_factor_keeps_positions(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE)
    h.trades("s1", [0.5, -1.0] * 25, shadow=False)  # PF 0.5 over 50 trades
    h.advance(weeks=2)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.MICRO and "PF" in ev.reason
    assert h.actions.calls == [("cancel", "s1")]
    assert "demotion" in h.alerts.kinds(Severity.WARNING)


async def test_drawdown_demotion(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE)
    h.trades("s1", [-1.0] * 13, shadow=False)  # 13R > 1.5 x 8R
    h.advance(days=4)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and "drawdown" in ev.reason


async def test_slippage_demotion(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE)
    h.data.fill_rows += fills("s1", 20, 0.00005, T0)  # 2.5 x model
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and "slippage" in ev.reason


async def test_drift_demotion(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE)
    h.trades("s1", [2.5 if i % 5 == 0 else -0.4 for i in range(50)], shadow=False)  # PF ok, win rate 20%
    h.advance(weeks=2)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and "win rate" in ev.reason and "PF" not in ev.reason


async def test_micro_demotions_shadow_then_retired_within_window(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO)
    h.trades("s1", [-1.0] * 13, shadow=False)
    h.advance(days=4)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.SHADOW
    assert h.actions.calls == [("cancel", "s1"), ("close", "s1")]  # out of money: close at market
    h.advance(days=30)
    h.to("s1", Stage.MICRO)
    h.trades("s1", [-1.0] * 13, shadow=False)
    h.advance(days=4)
    ev = await h.ev.evaluate("s1", "1.0.0")
    assert ev is not None and ev.to_stage == Stage.RETIRED  # second demotion within 6 months


async def test_failed_demotion_action_alerts_but_stage_stands(h: H) -> None:
    h.new()
    h.to("s1", Stage.SHADOW, Stage.MICRO, Stage.LIVE)
    h.actions.fail = True
    h.trades("s1", [-1.0] * 13, shadow=False)
    h.advance(days=4)
    await h.ev.evaluate("s1", "1.0.0")
    assert h.stage() == Stage.MICRO
    assert "demotion_action_failed" in h.alerts.kinds(Severity.CRITICAL)


def test_promotion_config_loads() -> None:
    assert CFG.shadow.risk_per_trade == 0.0
    assert CFG.global_.max_promotions_to_live_per_week == 2
