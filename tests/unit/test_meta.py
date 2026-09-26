"""L3 meta-labeling (spec 14.5), from its sentences: "cross-validation must be purged and embargoed", "the
meta model can only reduce or skip", "at least 300 signals", "activates only if, out of sample, the filtered
strategy has higher expectancy and a higher deflated Sharpe", "models are stored with training window,
features, hyperparameters, seed, metrics and file hash"; and the historical journal sees no future."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.learning.history import journal_from_backtest
from autotrader.learning.meta import (
    EMBARGO,
    MIN_SIGNALS,
    Dataset,
    evaluate,
    purged_folds,
    train_and_register,
)
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.poisoning import poison_after

ROOT = Path(__file__).resolve().parents[2]
HOUR = 3_600_000_000_000


def synthetic(n: int, predictive: bool, seed: int = 3) -> Dataset:
    """Signals an hour apart, each resolved within 3 hours. With `predictive`, feature 0 says who wins."""
    rng = np.random.default_rng(seed)
    good = rng.random(n) < 0.4
    win = np.where(good, rng.random(n) < 0.75, rng.random(n) < 0.2) if predictive else rng.random(n) < 0.4
    x = np.column_stack([good.astype(float) if predictive else rng.random(n), rng.normal(size=n)])
    start = np.arange(n, dtype=np.int64) * HOUR
    return Dataset(
        names=["f0", "f1"],
        x=x,
        y=win.astype(np.int64),
        r=np.where(win, 2.0, -1.0),
        start=start,
        end=start + 3 * HOUR,
    )


def test_folds_are_purged_and_embargoed_and_test_every_signal_once() -> None:
    ds = synthetic(1000, True)
    folds = purged_folds(ds.start, ds.end)
    tested = np.concatenate([te for _, te in folds])
    assert sorted(tested.tolist()) == list(range(1000))
    gap = int(np.ceil(EMBARGO * 1000))
    for train, test in folds:
        t0, t1 = ds.start[test[0]], ds.end[test].max()
        assert not ((ds.end[train] >= t0) & (ds.start[train] <= t1)).any()  # no outcome window overlaps
        after = set(range(test[-1] + 1, test[-1] + 1 + gap))
        assert not after & set(train.tolist())  # the embargo right after the test fold
        assert set(train.tolist()) & set(range(test[-1] + 1 + gap, 1000)) or test[-1] + 1 + gap >= 1000


def test_a_predictive_feature_passes_the_out_of_sample_gate() -> None:
    rep, _keep = evaluate(synthetic(900, True))
    assert rep.passed and rep.reasons == []
    assert rep.filtered_expectancy > rep.base_expectancy and rep.filtered_dsr > rep.base_dsr
    assert (
        0.25 < rep.kept_share < 0.6
        and rep.kept_win_rate is not None
        and rep.kept_win_rate > rep.base_win_rate
    )


def test_noise_does_not_pass() -> None:
    rep, _ = evaluate(synthetic(900, False))
    assert not rep.passed and rep.reasons  # control: the predictive case above passes


def test_it_only_keeps_or_skips_never_adds() -> None:
    ds = synthetic(900, True)
    _rep, keep = evaluate(ds)
    assert keep.dtype == bool  # a size factor of 1 or 0, nothing above 1
    filtered = np.where(keep, ds.r, 0.0)
    assert np.all((filtered == ds.r) | (filtered == 0.0))


def test_fewer_than_300_signals_are_refused() -> None:
    rep, keep = evaluate(synthetic(MIN_SIGNALS - 1, True))
    assert not rep.passed and "need 300" in rep.reasons[0] and keep.all()
    assert evaluate(synthetic(MIN_SIGNALS + 300, True))[0].signals == 600  # control


def test_a_losing_strategy_that_the_filter_improves_is_passed_with_a_warning() -> None:
    ds = synthetic(900, True)
    losing = Dataset(ds.names, ds.x, ds.y, np.where(ds.y == 1, 1.0, -1.0), ds.start, ds.end)
    rep, _ = evaluate(losing)
    assert rep.warnings == [] if rep.filtered_expectancy > 0 else "still loses" in rep.warnings[0]


def test_the_model_is_stored_with_what_it_learned_from(tmp_path: Path) -> None:
    ls = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
    frame = generate(SyntheticSpec(symbol="SYNTH", days=900, seed=4))
    entries = journal_from_backtest(ls.cls, frame, "SYNTH", synthetic_instrument())
    assert len(entries) >= MIN_SIGNALS, len(entries)
    rec, rep = train_and_register(entries, tmp_path, seed=1)
    assert rec is not None and rec.status == ("passed_oos" if rep.passed else "refused")
    blob = (tmp_path / rec.file).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == rec.sha256
    [line] = (tmp_path / "registry.jsonl").read_text().splitlines()
    row = json.loads(line)
    assert row["features"] and row["params"]["max_iter"] == 200 and row["seed"] == 1
    assert row["window_start"] < row["window_end"] and row["report"]["signals"] == len(entries)


def test_the_historical_journal_sees_no_future() -> None:
    ls = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
    frame = generate(SyntheticSpec(symbol="SYNTH", days=240, seed=9))
    cut = frame["open_time"][0] + timedelta(days=200)
    clean = journal_from_backtest(ls.cls, frame, "SYNTH", synthetic_instrument())
    poisoned = journal_from_backtest(
        ls.cls, poison_after(frame, cut, seed=2), "SYNTH", synthetic_instrument()
    )
    before = [(e.created_at, e.features) for e in clean if e.outcome and e.outcome.resolved_at < cut]
    assert (
        before
        and before == [(e.created_at, e.features) for e in poisoned if e.created_at < cut][: len(before)]
    )
    assert {e.outcome.label for e in clean if e.outcome} <= {1, -1, 0}


@pytest.mark.parametrize("n", [1000])
def test_evaluation_is_deterministic(n: int) -> None:
    a, b = evaluate(synthetic(n, True), seed=5), evaluate(synthetic(n, True), seed=5)
    assert a[0] == b[0] and (a[1] == b[1]).all()
