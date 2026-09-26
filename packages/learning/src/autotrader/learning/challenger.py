"""Champion vs challenger (spec 14.8): when a challenger in shadow may replace its champion; rollback.

A challenger replaces the champion only if all hold: both have at least 4 weeks and 40 signals in shadow;
the challenger's average R per signal is higher and a one-sided bootstrap of the difference gives p below
0.10 / k (k challengers of the strategy at once: Bonferroni); its drawdown is not worse than 1.2x the
champion's; the strategy has not swapped in the last month. (Passing full validation as its own version is
checked before a challenger ever reaches shadow.) Every test, passed or failed, is recorded as a trial.
The old champion stays in shadow 4 weeks; a demotion of the new one in that time rolls the swap back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

MIN_WEEKS = 4
MIN_SIGNALS = 40
ALPHA = 0.10
DD_RATIO = 1.2
SWAP_GAP = timedelta(days=30)
ROLLBACK_WINDOW = timedelta(weeks=4)
BOOTSTRAP = 10_000


@dataclass(frozen=True)
class Record:
    """One version's shadow signals with an outcome, in time order."""

    r: Sequence[float]
    first_at: datetime
    last_at: datetime


@dataclass(frozen=True)
class SwapVerdict:
    swap: bool
    reasons: list[str] = field(default_factory=list)  # why not (empty when it swaps)
    diff: float | None = None  # challenger avg R minus champion avg R
    p_value: float | None = None
    champion_dd: float | None = None  # in R
    challenger_dd: float | None = None


def max_drawdown_r(r: Sequence[float]) -> float:
    eq = np.cumsum(np.asarray(r, dtype=np.float64))
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.max(peak - eq))


def bootstrap_p(
    champion: Sequence[float], challenger: Sequence[float], *, seed: int, n: int = BOOTSTRAP
) -> float:
    """One-sided: the share of resampled differences (challenger - champion) that are <= 0."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(champion, dtype=np.float64), np.asarray(challenger, dtype=np.float64)
    ma = a[rng.integers(0, a.size, size=(n, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(n, b.size))].mean(axis=1)
    return float(np.mean(mb - ma <= 0))


def swap_test(
    champion: Record,
    challenger: Record,
    *,
    now: datetime,
    k: int = 1,
    last_swap: datetime | None = None,
    seed: int = 0,
) -> SwapVerdict:
    why: list[str] = []
    for name, rec in (("champion", champion), ("challenger", challenger)):
        weeks = (rec.last_at - rec.first_at) / timedelta(weeks=1) if rec.r else 0.0
        if weeks < MIN_WEEKS or len(rec.r) < MIN_SIGNALS:
            why.append(
                f"{name}: {weeks:.1f} weeks and {len(rec.r)} signals (needs {MIN_WEEKS} and {MIN_SIGNALS})"
            )
    if last_swap is not None and now - last_swap < SWAP_GAP:
        why.append(f"swapped {(now - last_swap).days} days ago (one swap a month at most)")
    if why and (not champion.r or not challenger.r):
        return SwapVerdict(False, why)
    diff = float(np.mean(challenger.r) - np.mean(champion.r)) if champion.r and challenger.r else None
    p = bootstrap_p(champion.r, challenger.r, seed=seed) if champion.r and challenger.r else None
    dd_a, dd_b = max_drawdown_r(champion.r), max_drawdown_r(challenger.r)
    if diff is not None and diff <= 0:
        why.append(f"challenger is not better ({diff:+.3f}R per signal)")
    threshold = ALPHA / max(k, 1)
    if p is not None and p >= threshold:
        why.append(
            f"not significant: bootstrap p {p:.3f} >= {threshold:.3f}"
            + (f" ({k} challengers)" if k > 1 else "")
        )
    if dd_b > DD_RATIO * dd_a:
        why.append(f"drawdown {dd_b:.1f}R > {DD_RATIO}x the champion's {dd_a:.1f}R")
    return SwapVerdict(not why, why, diff, p, dd_a, dd_b)


def rollback_due(swapped_at: datetime, demoted_at: datetime | None) -> bool:
    """The new champion was demoted within 4 weeks of the swap: put the old one back."""
    return demoted_at is not None and swapped_at <= demoted_at <= swapped_at + ROLLBACK_WINDOW
