"""L3 meta-labeling (spec 14.5): learn which of a strategy's signals to skip.

For a strategy with at least 300 signals with outcomes, a gradient-boosted tree classifier (scikit-learn's
HistGradientBoostingClassifier, the LightGBM algorithm; decisions.md) predicts the probability that a signal
reaches its target before its stop, from the market snapshot at the signal. It is judged only out of sample,
by purged and embargoed time-ordered cross-validation: training samples whose outcome window overlaps the
test fold are dropped, and so are the next 1% of samples after each test fold.

The model can only skip (size factor 0) or keep (1) a signal; it never adds size. A signal is kept when the
model's probability is at least the training set's base rate of winners: no threshold is searched, so there
is nothing extra to count as a trial. A model version passes only if, out of sample, the filtered strategy
has a higher expectancy in R per signal (skipped signals count 0) and a higher deflated Sharpe than the
unfiltered one; then it still has to win a shadow A/B before it touches money (spec; not in this slice).
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]

from autotrader.learning.journal import JournalEntry
from autotrader.validation.dsr import deflated_sharpe, sharpe

MIN_SIGNALS = 300
FOLDS = 5
EMBARGO = 0.01
PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}


def features_of(e: JournalEntry) -> dict[str, float]:
    row = e.features.row()
    row["side_buy"] = 1.0 if e.side == "buy" else 0.0
    row["win_rate_20"] = math.nan if e.strategy_win_rate_20 is None else e.strategy_win_rate_20
    return row


@dataclass(frozen=True)
class Dataset:
    names: list[str]
    x: np.ndarray  # (n, features), NaN = unknown
    y: np.ndarray  # 1 = target reached first
    r: np.ndarray  # R per signal
    start: np.ndarray  # signal time, ns
    end: np.ndarray  # outcome time, ns


def dataset(entries: Sequence[JournalEntry]) -> Dataset:
    done = sorted((e for e in entries if e.outcome is not None), key=lambda e: e.created_at)
    rows = [features_of(e) for e in done]
    names = sorted(rows[0]) if rows else []
    x = np.array([[r[k] for k in names] for r in rows], dtype=np.float64).reshape(len(rows), len(names))
    known = (
        ~np.isnan(x).all(axis=0) if rows else np.ones(len(names), dtype=bool)
    )  # never known: no information
    return Dataset(
        names=[n for n, k in zip(names, known, strict=True) if k],
        x=x[:, known],
        y=np.array([1 if e.outcome and e.outcome.label == 1 else 0 for e in done], dtype=np.int64),
        r=np.array([e.outcome.r for e in done if e.outcome], dtype=np.float64),
        start=np.array([int(e.created_at.timestamp() * 1e9) for e in done], dtype=np.int64),
        end=np.array(
            [int(e.outcome.resolved_at.timestamp() * 1e9) for e in done if e.outcome], dtype=np.int64
        ),
    )


def purged_folds(
    start: np.ndarray, end: np.ndarray, k: int = FOLDS, embargo: float = EMBARGO
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered folds; each training set drops samples whose [start, end] overlaps the test fold's span,
    and the `embargo` share of samples right after the fold (their features may still carry its signal)."""
    n = start.size
    bounds = np.linspace(0, n, k + 1).astype(int)
    gap = math.ceil(embargo * n)
    out = []
    idx = np.arange(n)
    for a, b in pairwise(bounds):
        if b <= a:
            continue
        test = idx[a:b]
        t0, t1 = start[a], end[a:b].max()
        overlaps = (end >= t0) & (start <= t1)
        embargoed = (idx >= b) & (idx < b + gap)
        train = idx[~overlaps & ~embargoed & ((idx < a) | (idx >= b))]
        out.append((train, test))
    return out


@dataclass(frozen=True)
class MetaReport:
    signals: int
    folds: int
    base_expectancy: float  # mean R per signal, unfiltered, over the out-of-sample folds
    filtered_expectancy: float  # mean R per signal with skipped signals at 0
    kept_share: float
    kept_win_rate: float | None
    base_win_rate: float
    base_dsr: float
    filtered_dsr: float
    passed: bool
    reasons: list[str]  # why it failed the gate
    warnings: list[str]  # true things a pass does not say (a filtered strategy that still loses)


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, **PARAMS)


def evaluate(ds: Dataset, *, seed: int = 0, k: int = FOLDS) -> tuple[MetaReport, np.ndarray]:
    """Out-of-sample decisions for every signal a fold tested, and the gate. Returns (report, keep mask)."""
    n = ds.y.size
    if n < MIN_SIGNALS:
        r = MetaReport(
            n,
            0,
            0.0,
            0.0,
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            False,
            [f"{n} signals; meta models need {MIN_SIGNALS}"],
            [],
        )
        return r, np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    tested = np.zeros(n, dtype=bool)
    for train, test in purged_folds(ds.start, ds.end, k):
        if train.size < 50 or np.unique(ds.y[train]).size < 2:
            continue  # the first fold has nothing before it; a fold cannot learn from one class
        m = _model(seed).fit(ds.x[train], ds.y[train])
        base_rate = float(ds.y[train].mean())
        keep[test] = m.predict_proba(ds.x[test])[:, 1] >= base_rate
        tested[test] = True
    r_base = ds.r[tested]
    r_filt = np.where(keep[tested], ds.r[tested], 0.0)
    kept = keep[tested]
    trials = [sharpe(r_base), sharpe(r_filt)]  # both variants were tried: each is deflated against both
    dsr_b, dsr_f = deflated_sharpe(r_base, trials).probability, deflated_sharpe(r_filt, trials).probability
    why = []
    if not r_filt.mean() > r_base.mean():
        why.append(f"filtered expectancy {r_filt.mean():+.3f}R is not above unfiltered {r_base.mean():+.3f}R")
    if not dsr_f > dsr_b:
        why.append(f"filtered deflated Sharpe {dsr_f:.3f} is not above unfiltered {dsr_b:.3f}")
    rep = MetaReport(
        signals=n,
        folds=k,
        base_expectancy=float(r_base.mean()),
        filtered_expectancy=float(r_filt.mean()),
        kept_share=float(kept.mean()),
        kept_win_rate=float((ds.r[tested][kept] > 0).mean()) if kept.any() else None,
        base_win_rate=float((r_base > 0).mean()),
        base_dsr=dsr_b,
        filtered_dsr=dsr_f,
        passed=not why,
        reasons=why,
        warnings=[f"still loses {r_filt.mean():+.3f}R per signal after the filter: better is not profitable"]
        if r_filt.mean() <= 0
        else [],
    )
    return rep, keep


@dataclass(frozen=True)
class MetaModel:
    """A trained model version: what it learned from and how it did, with the file it is kept in."""

    model_id: str
    strategy_id: str
    strategy_version: str
    window_start: str
    window_end: str
    features: list[str]
    params: dict[str, Any]
    seed: int
    report: dict[str, Any]
    file: str
    sha256: str
    created_at: str
    status: str  # "passed_oos" (may enter a shadow A/B) or "refused"


def train_and_register(
    entries: Sequence[JournalEntry], models_dir: Path, *, seed: int = 0
) -> tuple[MetaModel | None, MetaReport]:
    """Evaluate out of sample; fit the final model on everything; store it and its record (spec 14.5)."""
    ds = dataset(entries)
    rep, _ = evaluate(ds, seed=seed)
    if ds.y.size < MIN_SIGNALS:
        return None, rep
    model = _model(seed).fit(ds.x, ds.y)
    blob = pickle.dumps({"model": model, "features": ds.names, "base_rate": float(ds.y.mean())})
    digest = hashlib.sha256(blob).hexdigest()
    first = entries[0]
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"meta_{first.strategy_id}_{first.strategy_version}_{digest[:12]}.pkl"
    path.write_bytes(blob)
    rec = MetaModel(
        model_id=digest[:16],
        strategy_id=first.strategy_id,
        strategy_version=first.strategy_version,
        window_start=datetime.fromtimestamp(int(ds.start[0]) / 1e9, UTC).isoformat(),
        window_end=datetime.fromtimestamp(int(ds.end[-1]) / 1e9, UTC).isoformat(),
        features=ds.names,
        params=PARAMS,
        seed=seed,
        report=asdict(rep),
        file=path.name,
        sha256=digest,
        created_at=datetime.now(UTC).isoformat(),
        status="passed_oos" if rep.passed else "refused",
    )
    with (models_dir / "registry.jsonl").open("a") as f:  # spec: model_registry (append-only)
        f.write(json.dumps(asdict(rec)) + "\n")
    return rec, rep
