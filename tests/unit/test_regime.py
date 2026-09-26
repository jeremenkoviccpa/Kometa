"""L4 regime (spec 14.6): each day's regime from closed daily bars, a strategy's results by regime, a regime
proposed as blocked only with the evidence, and a filter version that only drops signals and sees no future.
Refusals have accepted controls."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from autotrader.core.models import Timeframe
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.backtest import run_backtest
from autotrader.learning.features import FeatureSnapshot, snapshot
from autotrader.learning.journal import JournalEntry, Outcome
from autotrader.learning.lessons import diagnose
from autotrader.learning.regime import blocked, by_regime, finding, regime, regime_filtered
from autotrader.strategies_api.base import Strategy
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.inputs import prepare
from autotrader.validation.poisoning import future_poisoning_test

ROOT = Path(__file__).resolve().parents[2]
DEMO = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
T0 = datetime(2026, 1, 5, tzinfo=UTC)


def feats(eff: float | None, atr_pct: float | None = 0.5) -> FeatureSnapshot:
    return FeatureSnapshot(
        atr_h1=1.0, atr_pct_h1=0.5, slope_h1=None, slope_h4=None, slope_d1=None, level_dist_atr=None,
        realized_vol_d=None, spread_ratio=None, minutes_to_event=None, session=("london",), hour=10,
        weekday=1, efficiency_d1=eff, atr_pct_d1=atr_pct,
    )  # fmt: skip


def entry(i: int, r: float, eff: float) -> JournalEntry:
    return JournalEntry(
        signal_id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(i))),
        strategy_id="s",
        strategy_version="1.0.0",
        symbol="XAUUSD",
        side="buy",
        timeframe=Timeframe.H1,
        created_at=T0 + timedelta(hours=i),
        entry=100.0,
        stop=99.0,
        target=102.0,
        shadow=False,
        strategy_win_rate_20=None,
        features=feats(eff),
        outcome=Outcome(
            label=1 if r > 0 else -1,
            r=r,
            mae_r=0.5,
            mfe_r=1.0,
            minutes=60,
            resolved_at=T0 + timedelta(hours=i, minutes=60),
        ),
    )


def test_the_regime_rules() -> None:
    assert regime(feats(0.5, 0.9)) == ("trending", "high vol")
    assert regime(feats(0.1, 0.1)) == ("ranging", "low vol")
    assert regime(feats(0.3, 0.5)) == ("mixed", "normal vol")
    assert regime(feats(None, None)) == (None, None)  # too little history is unknown, not "ranging"


def test_the_regime_is_known_from_closed_daily_bars_only() -> None:
    frame = generate(SyntheticSpec(symbol="SYNTH", days=300, seed=3))
    m = DEMO.manifest.model_copy(update={"timeframes": (Timeframe.D1,)})
    d1 = prepare({"SYNTH": frame}, m, {"SYNTH": synthetic_instrument()}, synthetic=True).series[
        ("SYNTH", Timeframe.D1)
    ]
    now = frame["open_time"][0] + timedelta(days=220, hours=13)  # > 100 daily ATRs of history
    f = snapshot({Timeframe.D1: d1}, now, spread=0.0)
    assert f.efficiency_d1 is not None and 0 <= f.efficiency_d1 <= 1 and f.atr_pct_d1 is not None
    later = snapshot({Timeframe.D1: d1}, now + timedelta(days=10), spread=0.0)
    assert later.efficiency_d1 != f.efficiency_d1  # it moves with the days it is allowed to see


def losing_in_trends() -> list[JournalEntry]:
    return [entry(i, -1.0 if i % 5 else 1.0, 0.5) for i in range(25)] + [
        entry(100 + i, 1.5 if i % 2 else -1.0, 0.1) for i in range(40)
    ]


def test_a_regime_it_loses_in_is_proposed_as_blocked() -> None:
    stats = by_regime(losing_in_trends())
    [bad] = blocked(stats)
    assert bad.regime == "trending" and bad.n == 25 and bad.avg_r < 0 < bad.rest_avg_r
    assert "trending" in finding(stats) and "hypothesis" in finding(stats)


def test_nothing_is_blocked_without_the_evidence() -> None:
    few = losing_in_trends()[5:25] + losing_in_trends()[25:]  # 19 signals in trends: too few
    assert blocked(by_regime(few[1:])) == []
    even = [entry(i, 1.5 if i % 2 else -1.0, 0.5 if i % 3 else 0.1) for i in range(90)]
    assert blocked(by_regime(even)) == [] and "no regime stands out" in finding(by_regime(even))
    assert blocked(by_regime(losing_in_trends()))  # control


def test_lessons_see_the_regime_too() -> None:
    d = diagnose(losing_in_trends())
    assert d.worst is not None and d.worst.condition == "in a trending market"


def _trades(cls: type[Strategy]) -> list[Any]:
    frame = generate(SyntheticSpec(symbol="SYNTH", days=300, seed=4))
    inp = prepare({"SYNTH": frame}, cls.manifest, {"SYNTH": synthetic_instrument()}, synthetic=True)
    return run_backtest(cls, inp.m1, inp.series, inp.instruments, inp.cost_model).trades


def test_the_filter_is_a_new_version_that_only_drops_signals() -> None:
    cls = regime_filtered(DEMO.cls, ["trending"], "1.0.1")
    assert (cls.manifest.version, cls.manifest.origin) == ("1.0.1", "learning_regime")
    plain, filt = _trades(DEMO.cls), _trades(cls)
    ready = min(t.entry_time_ns for t in filt)

    def key(ts: list[Any]) -> set[tuple[int, str]]:
        return {(t.entry_time_ns, t.side) for t in ts if t.entry_time_ns >= ready}

    assert 0 < len(key(filt)) < len(key(plain)) and key(filt) <= key(plain)
    nothing = regime_filtered(DEMO.cls, ["no such regime"], "1.0.2")
    assert key(_trades(nothing)) == key(plain)  # control: skipping nothing changes nothing


def test_the_filter_sees_no_future() -> None:
    cls = regime_filtered(DEMO.cls, ["trending"], "1.0.1")
    frame = generate(SyntheticSpec(symbol="SYNTH", days=220, seed=6))
    rep = future_poisoning_test(
        cls, {"SYNTH": frame}, {"SYNTH": synthetic_instrument()}, frame["open_time"][0] + timedelta(days=180)
    )
    assert rep.passed, rep.first_difference
    assert rep.signals_checked > 0
