"""L2 re-optimization and the champion/challenger rule, written from the spec's sentences (14.4, 14.8):
"a challenger only if they differ by more than the stability band", "parameter changes capped at 25 percent
per step", "at least 4 weeks and at least 40 signals for both", "p below 0.10 / k", "drawdown not worse
than 1.2 times", "at most one swap per month", "every attempt is counted". Refusals have accepted controls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from autotrader.core.ledger import JsonlLedger
from autotrader.core.series import to_ns
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.learning.challenger import Record, bootstrap_p, max_drawdown_r, rollback_due, swap_test
from autotrader.learning.reopt import challenger_class, fit_recent, next_version, step_params
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.inputs import prepare
from autotrader.validation.runner import param_candidates
from autotrader.validation.store import TrialRegistry

ROOT = Path(__file__).resolve().parents[2]
DEMO = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
M = DEMO.manifest
CHAMP = M.param_values()  # fast 12, slow 48, atr_len 14, stop_atr 2.0, reward_risk 2.0
T0 = datetime(2026, 1, 5, tzinfo=UTC)


# ---------------------------------------------------------------- re-optimization


def test_no_challenger_inside_the_stability_band() -> None:
    near = {**CHAMP, "fast": 14, "stop_atr": 2.3}  # +16.7%, +15%: inside 20%
    assert step_params(M, CHAMP, near, band=0.20) is None
    far = {**CHAMP, "fast": 15}  # +25%: outside
    assert step_params(M, CHAMP, far, band=0.20) is not None  # control


def test_each_parameter_moves_at_most_25_percent_a_step_and_stays_in_range() -> None:
    target = {**CHAMP, "fast": 30, "slow": 20, "stop_atr": 5.0, "reward_risk": 2.1}
    out = step_params(M, CHAMP, target, band=0.20)
    assert out is not None
    assert out["fast"] == 15  # 12 + 25% of 12
    assert out["slow"] == 36  # 48 - 25% of 48
    assert out["stop_atr"] == pytest.approx(2.5)
    assert out["reward_risk"] == 2.0  # inside the band: left alone
    for k, p in M.params.items():
        if p.min is not None and p.max is not None:
            assert p.min <= float(out[k]) <= p.max


def test_only_tunable_parameters_move() -> None:
    out = step_params(M, CHAMP, {**CHAMP, "atr_len": 50, "fast": 30}, band=0.20)
    assert out is not None and out["atr_len"] == 14 and out["fast"] == 15


def test_the_challenger_is_the_same_code_as_a_new_version() -> None:
    assert next_version("1.0.2", ["1.0.0", "1.0.3", "1.1.4"]) == "1.0.4"
    cls = challenger_class(DEMO.cls, "1.0.1", {**CHAMP, "fast": 15})
    m = cls.manifest
    assert (m.version, m.origin, m.param_values()["fast"]) == ("1.0.1", "learning_reopt", 15)
    assert issubclass(cls, DEMO.cls) and DEMO.cls.manifest.version == "1.0.0"  # the champion is untouched


def test_every_parameter_set_tried_is_counted_as_a_trial(tmp_path: Path) -> None:
    frame = generate(SyntheticSpec(symbol="SYNTH", days=200, seed=4))
    inp = prepare({"SYNTH": frame}, M, {"SYNTH": synthetic_instrument()}, synthetic=True)
    reg = TrialRegistry(JsonlLedger(tmp_path / "trials.jsonl"))
    start, end = to_ns(frame["open_time"][0]), to_ns(frame["open_time"][-1])
    fit = fit_recent(
        DEMO.cls, inp, start_ns=start, end_ns=end, budget=4, min_trades=5, registry=reg, code_hash="c"
    )
    trials = reg.trials(M.family)
    assert fit.tried == 4 and len(trials) == 4  # the count, after the last trial is recorded
    assert all(t.note == "reopt" and t.kind == "wf_train" for t in trials)
    assert fit.params in param_candidates(M, 4, 0)  # the winner is one of the sets actually tried


# ---------------------------------------------------------------- champion vs challenger


def rec(r: list[float], weeks: float = 6.0) -> Record:
    return Record(r=r, first_at=T0, last_at=T0 + timedelta(weeks=weeks))


BASE = [1.0, -1.0] * 25  # 50 signals, mean 0
BETTER = [x + 0.3 for x in BASE]  # +0.3R a signal: bootstrap p about 0.07


def test_a_significantly_better_challenger_swaps() -> None:
    v = swap_test(rec(BASE), rec(BETTER), now=T0 + timedelta(weeks=6))
    assert v.swap and v.reasons == [] and v.diff == pytest.approx(0.3)
    assert v.p_value is not None and 0.034 < v.p_value < 0.10


def test_both_need_4_weeks_and_40_signals() -> None:
    assert not swap_test(rec(BASE), rec(BETTER, weeks=3.5), now=T0 + timedelta(weeks=6)).swap
    assert not swap_test(rec(BASE[:38]), rec(BETTER), now=T0 + timedelta(weeks=6)).swap
    assert swap_test(rec(BASE[:40]), rec(BETTER), now=T0 + timedelta(weeks=6)).swap  # control: exactly 40


def test_more_challengers_need_a_smaller_p_bonferroni() -> None:
    now = T0 + timedelta(weeks=6)
    assert swap_test(rec(BASE), rec(BETTER), now=now, k=1).swap
    v = swap_test(rec(BASE), rec(BETTER), now=now, k=3)  # threshold 0.033: the same data no longer swaps
    assert not v.swap and any("not significant" in r for r in v.reasons)


def test_not_better_never_swaps() -> None:
    v = swap_test(rec(BETTER), rec(BASE), now=T0 + timedelta(weeks=6))
    assert not v.swap and any("not better" in r for r in v.reasons)


def test_a_deeper_drawdown_blocks_the_swap() -> None:
    losing_streak = [-1.0] * 10 + [x + 0.6 for x in BASE[:40]]  # better on average, deeper hole
    v = swap_test(rec(BASE), rec(losing_streak), now=T0 + timedelta(weeks=6))
    assert v.challenger_dd is not None and v.champion_dd is not None
    assert v.challenger_dd > 1.2 * v.champion_dd and not v.swap and any("drawdown" in r for r in v.reasons)


def test_at_most_one_swap_a_month() -> None:
    now = T0 + timedelta(weeks=6)
    assert not swap_test(rec(BASE), rec(BETTER), now=now, last_swap=now - timedelta(days=10)).swap
    assert swap_test(rec(BASE), rec(BETTER), now=now, last_swap=now - timedelta(days=31)).swap  # control


def test_rollback_only_for_a_demotion_within_4_weeks_of_the_swap() -> None:
    assert rollback_due(T0, T0 + timedelta(days=27))
    assert not rollback_due(T0, T0 + timedelta(days=29))
    assert not rollback_due(T0, None)


def test_helpers() -> None:
    assert max_drawdown_r([1.0, -2.0, -1.0, 3.0]) == pytest.approx(3.0)
    assert bootstrap_p([0.0] * 50, [1.0] * 50, seed=1) == 0.0
    assert bootstrap_p(BASE, BETTER, seed=1) == bootstrap_p(BASE, BETTER, seed=1)  # deterministic
