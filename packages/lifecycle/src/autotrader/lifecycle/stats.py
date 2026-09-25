"""Stage metrics and the backtest bands they are compared with (spec section 10).

Bands are percentile intervals of a bootstrap of the backtest sample, drawn at the size of the live or
shadow sample, so a small sample gets a wide band. Seeded: the same inputs always give the same band.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np

from autotrader.core.models import Trade

BOOTSTRAP_RUNS = 4000


def bootstrap_mean_band(sample: Sequence[float], n: int, level: float, seed: int = 7) -> tuple[float, float]:
    """Central `level` interval of the mean of n i.i.d. draws from `sample`."""
    if n <= 0 or not sample:
        raise ValueError("need a sample and n > 0")
    rng = np.random.default_rng(seed)
    arr = np.asarray(sample, dtype=np.float64)
    means = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_RUNS, n))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def inside(x: float, band: tuple[float, float]) -> bool:
    return band[0] <= x <= band[1]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def profit_factor(rs: Sequence[float]) -> float:
    gains = sum(r for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return gains / losses


def max_drawdown_r(rs: Sequence[float]) -> float:
    """Largest peak-to-trough fall of cumulative R (0 if it never falls)."""
    peak = cum = worst = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        worst = max(worst, peak - cum)
    return worst


def daily_r(trades: Sequence[Trade]) -> list[float]:
    """R per calendar weekday from first to last exit, zero days included."""
    if not trades:
        return []
    by_day: dict[date, float] = {}
    for t in trades:
        d = t.exit_time.date()
        by_day[d] = by_day.get(d, 0.0) + t.r_multiple
    first, last = min(by_day), max(by_day)
    out = []
    for k in range((last - first).days + 1):
        d = date.fromordinal(first.toordinal() + k)
        if d.weekday() < 5:
            out.append(by_day.get(d, 0.0))
    return out


def annualized_sharpe(daily: Sequence[float]) -> float:
    if len(daily) < 2:
        return 0.0
    arr = np.asarray(daily, dtype=np.float64)
    sd = float(arr.std(ddof=1))
    return 0.0 if sd == 0 else float(arr.mean() / sd * math.sqrt(252))
