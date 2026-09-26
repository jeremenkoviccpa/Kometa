"""Champion vs challenger through the real registry and evaluator (spec 14.8): "a challenger replaces the
champion only if ...", "every swap test, passed or failed, is recorded", "on swap, the old champion is kept
in shadow for 4 more weeks; a demotion of the new champion in that period rolls back automatically".
Every refusal has an accepted control."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from autotrader.core.alerts import MemoryAlertSink
from autotrader.core.clock import SimClock
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import Stage, Trade
from autotrader.core.profile import BacktestProfile
from autotrader.core.timeutil import utc
from autotrader.lifecycle.config import load_promotion_config
from autotrader.lifecycle.evaluator import Evaluator, MemoryStageData
from autotrader.lifecycle.registry import IllegalTransitionError, Registry, VersionInfo

ROOT = Path(__file__).resolve().parents[2]
T0 = utc(2026, 1, 5, 12)
BASE = [1.0, -1.0] * 25  # the champion: 50 trades, 0R a trade
BETTER = [x + 0.3 for x in BASE]  # the challenger: +0.3R a trade (bootstrap p about 0.07)


def prof(v: str) -> BacktestProfile:
    return BacktestProfile(
        strategy_id="s1",
        strategy_version=v,
        trade_r=tuple([0.2, -1.0, 1.5, 0.4] * 100),
        weekly_entries=(5,) * 12,
        mc_dd_p95_r=8.0,
        model_slippage={"EURUSD": 0.00002},
        source="test",
        synthetic=False,
    )


class H:
    def __init__(self, tmp: Path) -> None:
        self.clock = SimClock(T0)
        self.alerts = MemoryAlertSink()
        self.reg = Registry(JsonlLedger(tmp / "registry.jsonl"), self.clock, self.alerts)
        self.data = MemoryStageData()
        self.ev = Evaluator(
            self.reg,
            load_promotion_config(ROOT / "config" / "promotion.yaml"),
            self.data,
            self.clock,
            self.alerts,
        )
        self.reg.submit_candidate(self._info("1.0.0", "owner", None), prof("1.0.0"))
        for s in (Stage.SHADOW, Stage.MICRO, Stage.LIVE):
            self.reg.transition("s1", "1.0.0", s, "test", actor="evaluator")

    @staticmethod
    def _info(v: str, origin: str, parent: str | None) -> VersionInfo:
        return VersionInfo(
            strategy_id="s1",
            version=v,
            family="f",
            origin=origin,
            demo_only=False,
            code_hash="c",
            params={},
            parent_version=parent,
            created_by="t",
        )

    def challenger(self, v: str = "1.0.1", parent: str = "1.0.0", origin: str = "learning_reopt") -> None:
        self.reg.submit_candidate(self._info(v, origin, parent), prof(v))
        self.reg.promote_candidate("s1", v, validation_passed=True, synthetic=False)

    def trades(self, v: str, rs: list[float], *, shadow: bool) -> None:
        t = self.clock.now()
        for i, r in enumerate(rs):
            at = t + timedelta(hours=8 * i)
            self.data.add_trade(trade(v, r, at, shadow))

    def advance(self, **kw: float) -> None:
        self.clock.advance_to(self.clock.now() + timedelta(**kw))

    def stage(self, v: str) -> Stage:
        return self.reg.get("s1", v).stage


def trade(v: str, r: float, t: datetime, shadow: bool) -> Trade:
    return Trade(
        trade_id=f"{v}-{t.isoformat()}",
        account_id="shadow" if shadow else "acc",
        strategy_id="s1",
        strategy_version=v,
        symbol="EURUSD",
        side="buy",
        lots=Decimal("0.1"),
        entry_time=t,
        entry_price=Decimal("1.1"),
        stop_price=Decimal("1.09"),
        exit_time=t + timedelta(hours=1),
        exit_price=Decimal("1.1"),
        pnl_gross=Decimal(100 * r),
        costs=Decimal(0),
        pnl_net=Decimal(100 * r),
        money_at_risk=Decimal(100),
        r_multiple=r,
        mae=0.5,
        mfe=1.0,
    )


def race(h: H, champion: list[float], challenger: list[float], weeks: float = 6) -> None:
    h.challenger()
    h.trades("1.0.0", champion, shadow=False)
    h.trades("1.0.1", challenger, shadow=True)
    h.advance(weeks=weeks)


def test_a_better_challenger_takes_the_champions_stage_and_the_champion_waits_in_shadow(
    tmp_path: Path,
) -> None:
    h = H(tmp_path)
    race(h, BASE, BETTER)
    events = h.ev.review_challengers()
    assert (h.stage("1.0.1"), h.stage("1.0.0")) == (Stage.LIVE, Stage.SHADOW)
    assert [(e.strategy_version, e.to_stage) for e in events] == [
        ("1.0.1", Stage.LIVE),
        ("1.0.0", Stage.SHADOW),
    ]
    [test] = h.reg.swap_tests
    assert test["swap"] and test["challenger_signals"] == 50 and test["reasons"] == []


@pytest.mark.parametrize(
    ("champion", "challenger", "weeks", "why"),
    [
        (BASE, BETTER, 3, "weeks"),  # too early
        (BASE, BETTER[:30], 6, "signals"),  # too few signals
        (BETTER, BASE, 6, "not better"),
    ],
    ids=["3-weeks", "30-signals", "worse"],
)
def test_no_swap_without_the_evidence_but_the_test_is_recorded(
    tmp_path: Path, champion: list[float], challenger: list[float], weeks: float, why: str
) -> None:
    h = H(tmp_path)
    race(h, champion, challenger, weeks)
    assert h.ev.review_challengers() == []
    assert (h.stage("1.0.1"), h.stage("1.0.0")) == (Stage.SHADOW, Stage.LIVE)
    [test] = h.reg.swap_tests
    assert not test["swap"] and any(why in r for r in test["reasons"])


def test_swap_tests_survive_a_restart(tmp_path: Path) -> None:
    h = H(tmp_path)
    race(h, BASE, BETTER, weeks=3)
    h.ev.review_challengers()
    again = Registry(JsonlLedger(tmp_path / "registry.jsonl"), h.clock, MemoryAlertSink())
    assert again.swap_tests == h.reg.swap_tests and len(again.swap_tests) == 1


def test_only_a_reoptimized_child_in_shadow_may_replace_a_champion(tmp_path: Path) -> None:
    h = H(tmp_path)
    h.challenger("2.0.0", origin="owner")  # a new owner version is not a challenger: it climbs the ladder
    with pytest.raises(IllegalTransitionError):
        h.reg.swap("s1", "1.0.0", "2.0.0", "test")
    h.challenger("1.0.1", parent="0.9.0")  # someone else's child
    with pytest.raises(IllegalTransitionError):
        h.reg.swap("s1", "1.0.0", "1.0.1", "test")
    h.challenger("1.0.2")  # control
    h.reg.swap("s1", "1.0.0", "1.0.2", "test")
    assert h.stage("1.0.2") == Stage.LIVE


def test_a_new_champion_demoted_within_4_weeks_is_rolled_back(tmp_path: Path) -> None:
    h = H(tmp_path)
    race(h, BASE, BETTER)
    h.ev.review_challengers()
    h.advance(weeks=2)
    ev = h.reg.transition("s1", "1.0.1", Stage.MICRO, "drawdown", actor="evaluator")
    h.ev._maybe_rollback(ev)
    assert (h.stage("1.0.0"), h.stage("1.0.1")) == (Stage.LIVE, Stage.SHADOW)


def test_no_rollback_after_4_weeks(tmp_path: Path) -> None:
    h = H(tmp_path)
    race(h, BASE, BETTER)
    h.ev.review_challengers()
    h.advance(weeks=5)
    ev = h.reg.transition("s1", "1.0.1", Stage.MICRO, "drawdown", actor="evaluator")
    assert h.ev._maybe_rollback(ev) == []
    assert (h.stage("1.0.0"), h.stage("1.0.1")) == (Stage.SHADOW, Stage.MICRO)
