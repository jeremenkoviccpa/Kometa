"""Order gate hook for the in-process pipeline, plus the backtest-only guard.

In live modes the only gate is the risk-gate service (spec section 11). In
backtests the engine runs in process and needs sizing plus the basic stop
checks; `BacktestGuard` provides them and refuses to be used in any other
mode. Phase 4 plugs the real RiskGate in behind the same `OrderGate` protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

from autotrader.core.models import Signal
from autotrader.engine.costs import InstrumentCosts

Mode = Literal["backtest", "shadow", "paper", "live"]


@dataclass(frozen=True)
class GateRequest:
    signal: Signal
    ref_entry: float  # expected entry price (ask for buys, bid for sells, or the order price)
    spread: float
    equity: float
    open_risk: float  # money at risk across open positions, account currency
    costs: InstrumentCosts
    to_account: float


@dataclass(frozen=True)
class GateResult:
    lots: float  # 0 means rejected
    reasons: tuple[str, ...] = ()


class OrderGate(Protocol):
    def decide(self, req: GateRequest) -> GateResult: ...


def lots_for_risk(
    equity: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    costs: InstrumentCosts,
    to_account: float,
) -> float:
    """Spec section 11 sizing. Floors to lot_step; returns 0.0 below min_lot (never rounds up)."""
    distance = abs(entry - stop)
    if distance <= 0 or equity <= 0 or risk_fraction <= 0:
        return 0.0
    raw = equity * risk_fraction / (distance * costs.contract_size * to_account)
    steps = math.floor(raw / costs.lot_step + 1e-9)
    lots = round(steps * costs.lot_step, 10)
    lots = min(lots, costs.max_lot)
    return 0.0 if lots < costs.min_lot else lots


@dataclass
class BacktestGuard:
    """Backtest-only sizing and sanity checks. Never used outside mode='backtest'."""

    mode: Mode
    risk_fraction: float = 0.005
    min_stop_spreads: float = 1.5
    open_risk_total_max: float = 0.03
    reasons_seen: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode != "backtest":
            raise RuntimeError(
                "BacktestGuard may only be used in backtest mode; live paths use the risk gate"
            )

    def _reject(self, why: str) -> GateResult:
        self.reasons_seen[why] = self.reasons_seen.get(why, 0) + 1
        return GateResult(0.0, (why,))

    def decide(self, req: GateRequest) -> GateResult:
        s = req.signal
        if (s.side == "buy" and s.stop_price >= req.ref_entry) or (
            s.side == "sell" and s.stop_price <= req.ref_entry
        ):
            return self._reject("stop on wrong side of entry")
        if abs(req.ref_entry - s.stop_price) < self.min_stop_spreads * req.spread:
            return self._reject("stop closer than 1.5 x spread")
        lots = lots_for_risk(
            req.equity, self.risk_fraction, req.ref_entry, s.stop_price, req.costs, req.to_account
        )
        if lots == 0.0:
            return self._reject("size below min lot")
        new_risk = abs(req.ref_entry - s.stop_price) * req.costs.contract_size * lots * req.to_account
        if req.open_risk + new_risk > self.open_risk_total_max * req.equity:
            return self._reject("total open risk limit")
        return GateResult(lots)
