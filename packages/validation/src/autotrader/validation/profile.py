"""Validation report -> BacktestProfile: what the lifecycle evaluator compares shadow and live with."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from autotrader.core.profile import BacktestProfile
from autotrader.core.series import from_ns
from autotrader.engine.simbroker import TradeRecord
from autotrader.validation.runner import ValidationReport


def weekly_counts(entry_times: Sequence[datetime], start: datetime, end: datetime) -> tuple[int, ...]:
    """Entries per 7-day week from `start` to `end`, zero weeks included (partial last week dropped)."""
    weeks = max(1, int((end - start) / timedelta(weeks=1)))
    counts = [0] * weeks
    for t in entry_times:
        k = int((t - start) / timedelta(weeks=1))
        if 0 <= k < weeks:
            counts[k] += 1
    return tuple(counts)


def profile_from_trades(
    strategy_id: str,
    version: str,
    trades: Sequence[TradeRecord],
    window: tuple[datetime, datetime],
    *,
    mc_dd_p95: float,
    risk_fraction: float,
    model_slippage: Mapping[str, float],
    source: str,
    synthetic: bool,
) -> BacktestProfile:
    return BacktestProfile(
        strategy_id=strategy_id,
        strategy_version=version,
        trade_r=tuple(t.r_multiple for t in trades),
        weekly_entries=weekly_counts([from_ns(t.entry_time_ns) for t in trades], *window),
        # Monte Carlo drawdown is a fraction of equity at `risk_fraction` per trade; in R that is dd / rf
        mc_dd_p95_r=mc_dd_p95 / risk_fraction,
        model_slippage=dict(model_slippage),
        source=source,
        synthetic=synthetic,
    )


def profile_from_report(r: ValidationReport, model_slippage: Mapping[str, float]) -> BacktestProfile:
    if r.monte_carlo is None or not r.oos_trades:
        raise ValueError("report has no out-of-sample trades or Monte Carlo result; no profile")
    return profile_from_trades(
        r.strategy_id,
        r.version,
        r.oos_trades,
        r.research_window,
        mc_dd_p95=r.monte_carlo.dd_p95,
        risk_fraction=r.risk_fraction,
        model_slippage=model_slippage,
        source=f"validation {r.code_hash[:12]} data {','.join(sorted(r.data_versions.values()))}",
        synthetic=r.synthetic,
    )
