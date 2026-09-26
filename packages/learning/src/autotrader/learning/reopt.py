"""L2 re-optimization (spec 14.4): new parameters for an existing strategy from its most recent data.

Fit on the recent window (default 3 years, ending where the holdout starts, so the holdout stays unseen),
every variant recorded as a trial (the deflated Sharpe of the family pays for each one). A challenger exists
only if some tunable parameter moved by more than the stability band, and each parameter moves at most 25% a
step, so a strategy cannot lurch. The challenger is a new version of the same code with those parameters; it
must pass full validation as its own version and then win in shadow (learning.challenger) before it swaps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from autotrader.core.hashing import hash_obj
from autotrader.core.series import from_ns
from autotrader.engine.backtest import BacktestConfig, run_backtest
from autotrader.strategies_api.base import Strategy
from autotrader.strategies_api.manifest import ParamSpec, ParamValue, StrategyManifest
from autotrader.validation.dsr import sharpe
from autotrader.validation.inputs import EngineInputs
from autotrader.validation.runner import daily_r_returns, objective, param_candidates
from autotrader.validation.store import Trial, TrialRegistry

STEP_CAP = 0.25  # spec 14.4: parameter changes capped at 25% per step


def _scale(spec: ParamSpec, value: float) -> float:
    """What a 100% move means for a parameter: its value, or 10% of its range when the value is 0."""
    if value != 0:
        return abs(value)
    return 0.1 * (float(spec.max or 0.0) - float(spec.min or 0.0)) or 1.0


def step_params(
    manifest: StrategyManifest,
    champion: Mapping[str, ParamValue],
    target: Mapping[str, ParamValue],
    *,
    band: float,
    cap: float = STEP_CAP,
) -> dict[str, ParamValue] | None:
    """The challenger's parameters, or None when every tunable parameter is within the stability band."""
    tunable = {k: p for k, p in manifest.params.items() if p.tunable}
    moved = [
        k
        for k, p in tunable.items()
        if abs(float(target[k]) - float(champion[k])) > band * _scale(p, float(champion[k]))
    ]
    if not moved:
        return None
    out = dict(champion)
    for k in moved:
        p = tunable[k]
        c, t = float(champion[k]), float(target[k])
        limit = cap * _scale(p, c)
        v = c + max(-limit, min(limit, t - c))
        v = min(max(v, float(p.min if p.min is not None else v)), float(p.max if p.max is not None else v))
        out[k] = round(v) if isinstance(p.value, int) and not isinstance(p.value, bool) else round(v, 6)
    return out if out != dict(champion) else None


def next_version(base: str, taken: Iterable[str]) -> str:
    """The next free patch number of the champion's major.minor, e.g. 1.0.2 -> 1.0.3 (or later if taken)."""
    major, minor, patch = (int(x) for x in base.split("."))
    used = {int(v.split(".")[2]) for v in taken if v.split(".")[:2] == [str(major), str(minor)]}
    n = patch + 1
    while n in used:
        n += 1
    return f"{major}.{minor}.{n}"


def challenger_class(cls: type[Strategy], version: str, params: Mapping[str, ParamValue]) -> type[Strategy]:
    """The same code as a new version whose default parameters are the challenger's."""
    m = cls.manifest
    specs = {k: p.model_copy(update={"value": params.get(k, p.value)}) for k, p in m.params.items()}
    manifest = m.model_copy(update={"version": version, "params": specs, "origin": "learning_reopt"})
    return type(cls.__name__, (cls,), {"manifest": manifest})


@dataclass(frozen=True)
class Fit:
    params: dict[str, ParamValue]
    score: float  # t-statistic of mean R (validation.runner.objective)
    trades: int
    tried: int


def fit_recent(
    cls: type[Strategy],
    inp: EngineInputs,
    *,
    start_ns: int,
    end_ns: int,
    budget: int,
    min_trades: int,
    registry: TrialRegistry,
    code_hash: str,
    seed: int = 0,
    risk_fraction: float = 0.005,
) -> Fit:
    """The best of `budget` parameter sets on [start, end); every set tried is recorded as a trial."""
    m = cls.manifest
    bt = BacktestConfig(risk_fraction=risk_fraction)
    best = Fit(dict(m.param_values()), -math.inf, 0, 0)
    tried = 0
    for params in param_candidates(m, budget, seed):
        res = run_backtest(cls, inp.m1, inp.series, inp.instruments, inp.cost_model, params=params, config=bt)
        trades = [t for t in res.trades if start_ns <= t.entry_time_ns < end_ns]
        r = np.array([t.r_multiple for t in trades], dtype=np.float64)
        score = objective(r, min_trades)
        tried += 1
        registry.record(
            Trial(
                family=m.family,
                strategy_id=m.id,
                version=m.version,
                params_hash=hash_obj(dict(params)),
                kind="wf_train",
                data_version=hash_obj(inp.data_versions),
                window_start=from_ns(start_ns).isoformat(),
                window_end=from_ns(end_ns).isoformat(),
                sharpe=sharpe(
                    daily_r_returns(trades, start_ns, end_ns, risk_fraction)
                ),  # daily, as trials are
                trades=len(trades),
                code_hash=code_hash,
                note="reopt",
            )
        )
        if score > best.score:
            best = Fit(dict(params), score, len(trades), 0)
    return Fit(best.params, best.score, best.trades, tried)
