"""Weights (spec section 13), pure functions.

1. w_raw = n/(n+k) * sharpe_live + k/(n+k) * sharpe_backtest   (n = live trades)
2. w = max(w_raw, 0)
3. versions whose daily returns correlate above the threshold form a cluster that shares one slot:
   the cluster weighs as its strongest member
4. slot shares are normalized, then capped at max_share_per_version; the excess goes to uncapped slots
   (water filling). If every slot is capped the rest of the budget stays unallocated.
5. a slot's share is split among its members in proportion to their weights, so no version and no
   cluster ever exceeds the cap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from autotrader.core.models import Frozen, Stage

Key = tuple[str, str]


class VersionPerformance(Frozen):
    strategy_id: str
    version: str
    stage: Stage
    live_trades: int
    sharpe_live: float
    sharpe_backtest: float
    daily_r: dict[date, float]  # R per day of the version's live trading

    @property
    def key(self) -> Key:
        return (self.strategy_id, self.version)


def shrunk_weight(n: int, sharpe_live: float, sharpe_backtest: float, k: float) -> float:
    return (n / (n + k)) * sharpe_live + (k / (n + k)) * sharpe_backtest


def correlation(a: Mapping[date, float], b: Mapping[date, float], min_overlap: int) -> float:
    days = sorted(set(a) | set(b))
    if len(days) < min_overlap:
        return 0.0
    x = np.asarray([a.get(d, 0.0) for d in days])
    y = np.asarray([b.get(d, 0.0) for d in days])
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def clusters(perf: Sequence[VersionPerformance], threshold: float, min_overlap: int) -> list[list[Key]]:
    """Single linkage: A~B and B~C put A, B, C in one cluster."""
    parent = {p.key: p.key for p in perf}

    def find(k: Key) -> Key:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(perf):
        for b in perf[i + 1 :]:
            if correlation(a.daily_r, b.daily_r, min_overlap) > threshold:
                parent[find(a.key)] = find(b.key)
    groups: dict[Key, list[Key]] = {}
    for p in perf:
        groups.setdefault(find(p.key), []).append(p.key)
    return sorted(sorted(g) for g in groups.values())


def capped_shares(weights: Mapping[Key, float], cap: float) -> dict[Key, float]:
    """Normalize to shares summing to at most 1 with no share above `cap`."""
    live = {k: w for k, w in weights.items() if w > 0}
    shares = dict.fromkeys(weights, 0.0)
    fixed: dict[Key, float] = {}
    while live:
        budget = 1.0 - sum(fixed.values())
        total = sum(live.values())
        trial = {k: budget * w / total for k, w in live.items()}
        over = {k for k, s in trial.items() if s > cap}
        if not over:
            shares.update(trial)
            break
        for k in over:
            fixed[k] = cap
            del live[k]
    shares.update(fixed)
    return shares


def allocate_shares(
    perf: Sequence[VersionPerformance], *, k: float, threshold: float, min_overlap: int, cap: float
) -> tuple[dict[Key, float], list[list[Key]]]:
    w = {p.key: max(0.0, shrunk_weight(p.live_trades, p.sharpe_live, p.sharpe_backtest, k)) for p in perf}
    groups = clusters(perf, threshold, min_overlap)
    # the cap applies to the slot: a cluster together never gets more than one version could
    slot_w = {g[0]: max(w[m] for m in g) for g in groups}
    slot_share = capped_shares(slot_w, cap)
    shares: dict[Key, float] = {}
    for g in groups:
        total = sum(w[m] for m in g)
        for m in g:
            shares[m] = slot_share[g[0]] * w[m] / total if total > 0 else 0.0
    return shares, groups
