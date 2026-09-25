"""The allocator (spec section 13): per-version risk fractions and proposed lot sizes.

It is self-learning loop L5: the weights move with live results, weekly. Between rebalances the
allocation is frozen, except that a stage cap can only lower a fraction immediately (a demoted
version never keeps its old budget). Micro versions always get the micro fraction. The allocator
proposes; the risk gate sizes again independently and can only reduce.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

from pydantic import Field

from autotrader.allocator.config import AllocatorConfig
from autotrader.allocator.weights import VersionPerformance, allocate_shares
from autotrader.core.broker import intent_id_for
from autotrader.core.fileio import atomic_write_text
from autotrader.core.models import Frozen, Instrument, OrderIntent, Signal, Stage, UtcDatetime


def key_str(strategy_id: str, version: str) -> str:
    return f"{strategy_id}@{version}"


class Allocation(Frozen):
    at: UtcDatetime
    shares: dict[str, float]  # of the total risk budget
    risk_fraction: dict[str, float]  # per trade, before the live stage cap is re-applied
    clusters: list[list[str]] = Field(default_factory=list)
    inputs: dict[str, dict[str, float]] = Field(default_factory=dict)


class Allocator:
    def __init__(self, cfg: AllocatorConfig, state_path: Path | None = None) -> None:
        self.cfg = cfg
        self.path = state_path
        self.current: Allocation | None = None
        if state_path is not None and state_path.exists():
            self.current = Allocation.model_validate_json(state_path.read_text())

    def due(self, now: datetime) -> bool:
        return self.current is None or now - self.current.at >= timedelta(days=self.cfg.rebalance_days)

    def rebalance(
        self, perf: Sequence[VersionPerformance], now: datetime, *, force: bool = False
    ) -> Allocation:
        if not force and not self.due(now) and self.current is not None:
            return self.current
        weighted = [p for p in perf if p.stage in (Stage.LIVE, Stage.SCALED)]
        shares, groups = allocate_shares(
            weighted,
            k=self.cfg.shrinkage_k,
            threshold=self.cfg.correlation_cluster_threshold,
            min_overlap=self.cfg.min_overlap_days,
            cap=self.cfg.max_share_per_version,
        )
        stage = {p.key: p.stage for p in weighted}
        rf = {
            key_str(*k): min(s * self.cfg.total_risk_budget, self.cfg.stage_limit(stage[k]))
            for k, s in shares.items()
        }
        self.current = Allocation(
            at=now,
            shares={key_str(*k): s for k, s in shares.items()},
            risk_fraction=rf,
            clusters=[[key_str(*k) for k in g] for g in groups],
            inputs={
                key_str(*p.key): {
                    "n": p.live_trades,
                    "sharpe_live": p.sharpe_live,
                    "sharpe_bt": p.sharpe_backtest,
                }
                for p in weighted
            },
        )
        if self.path is not None:
            atomic_write_text(self.path, self.current.model_dump_json())
        return self.current

    def risk_fraction(self, strategy_id: str, version: str, stage: Stage) -> float:
        if stage in (Stage.MICRO, Stage.DEMO_ONLY):
            return self.cfg.stage_limit(Stage.MICRO)
        if stage not in (Stage.LIVE, Stage.SCALED) or self.current is None:
            return 0.0
        rf = self.current.risk_fraction.get(key_str(strategy_id, version), 0.0)
        return min(rf, self.cfg.stage_limit(stage))

    def propose(
        self,
        signal: Signal,
        stage: Stage,
        *,
        equity: Decimal,
        entry: Decimal,
        instrument: Instrument,
        to_account: Decimal,
        account_id: str,
    ) -> OrderIntent | None:
        """Lots for this signal from the version's risk fraction; None when it gets no budget."""
        rf = self.risk_fraction(signal.strategy_id, signal.strategy_version, stage)
        dist = abs(entry - Decimal(str(signal.stop_price)))
        if rf <= 0 or dist <= 0 or equity <= 0:
            return None
        raw = equity * Decimal(str(rf)) / (dist * instrument.contract_size * to_account)
        lots = (raw / instrument.lot_step).to_integral_value(rounding=ROUND_FLOOR) * instrument.lot_step
        lots = min(lots, instrument.max_lot)
        if stage == Stage.DEMO_ONLY:
            lots = instrument.min_lot  # paper trading of plumbing strategies: always the minimum size
        if lots < instrument.min_lot:
            return None
        return OrderIntent(
            intent_id=intent_id_for(signal.signal_id),  # one intent per signal, ever
            signal=signal,
            proposed_lots=lots,
            risk_fraction=rf,
            account_id=account_id,
        )


def dump(a: Allocation) -> str:
    return json.dumps(a.model_dump(mode="json"), indent=2, sort_keys=True)
