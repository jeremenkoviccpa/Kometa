"""Checks on what a strategy returns, shared by the backtest loop and the live runner.

A strategy that breaks its contract stops (fail closed); both modes apply the same rules so a
strategy cannot behave differently live than it did in validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from autotrader.core.models import Signal
from autotrader.strategies_api.base import Request
from autotrader.strategies_api.manifest import StrategyManifest


class StrategyError(Exception):
    """The strategy broke its contract. Fail closed: the run stops."""


def check_requests(
    reqs: object, manifest: StrategyManifest, now: datetime, seen_ids: set[str]
) -> Sequence[Request]:
    if not isinstance(reqs, list):
        raise StrategyError(f"{manifest.id} must return a list, got {type(reqs).__name__}")
    for r in reqs:
        if isinstance(r, Signal):
            if (r.strategy_id, r.strategy_version) != (manifest.id, manifest.version):
                raise StrategyError("signal carries another strategy's id or version")
            if r.created_at != now:
                raise StrategyError("signal created_at must equal market.now (use ctx.signal)")
            if r.symbol not in manifest.symbols:
                raise StrategyError(f"signal for unsubscribed symbol {r.symbol}")
            sid = str(r.signal_id)
            if sid in seen_ids:
                raise StrategyError(f"duplicate signal id {sid}")
            seen_ids.add(sid)
    return reqs
