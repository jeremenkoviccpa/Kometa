"""Probabilistic and Deflated Sharpe Ratio (spec section 9; Bailey and Lopez de Prado).

All Sharpe ratios here are per period (daily), NOT annualized, and trial
Sharpes must use the same periodicity as the tested strategy.

PSR(SR*) = Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2))
SR*      = sqrt(Var(SR_trials)) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))
with kurt the non-excess kurtosis and g the Euler-Mascheroni constant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def sharpe(returns: Sequence[float] | np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return 0.0 if sd == 0.0 else float(r.mean()) / sd


def moments(returns: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """(skewness, non-excess kurtosis), population estimators."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 3:
        return 0.0, 3.0
    d = r - r.mean()
    m2 = float((d**2).mean())
    if m2 == 0.0:
        return 0.0, 3.0
    return float((d**3).mean()) / m2**1.5, float((d**4).mean()) / m2**2


def psr(sr_hat: float, sr_star: float, t: int, skew: float, kurt: float) -> float:
    if t < 2:
        return 0.0
    var_term = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if var_term <= 0:
        return 0.0
    return _N.cdf((sr_hat - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term))


def expected_max_sharpe(trial_sharpe_var: float, n_trials: int) -> float:
    """SR* benchmark: the Sharpe the best of N unskilled trials is expected to reach."""
    if n_trials < 2 or trial_sharpe_var <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_sharpe_var) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


@dataclass(frozen=True)
class DSRResult:
    sharpe: float
    sr_star: float
    n_trials: int
    t: int
    skew: float
    kurt: float
    probability: float


def deflated_sharpe(
    returns: Sequence[float] | np.ndarray, trial_sharpes: Sequence[float] | np.ndarray
) -> DSRResult:
    r = np.asarray(returns, dtype=np.float64)
    sr = sharpe(r)
    skew, kurt = moments(r)
    ts = np.asarray(trial_sharpes, dtype=np.float64)
    var = float(ts.var(ddof=1)) if ts.size > 1 else 0.0
    sr_star = expected_max_sharpe(var, int(ts.size))
    return DSRResult(
        sr, sr_star, int(ts.size), int(r.size), skew, kurt, psr(sr, sr_star, int(r.size), skew, kurt)
    )
