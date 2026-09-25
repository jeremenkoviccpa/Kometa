"""What validation expects of a strategy version: the reference the lifecycle evaluator compares
shadow and live behaviour against (spec section 10). Produced by `validation`, read by `lifecycle`."""

from __future__ import annotations

from pydantic import Field

from autotrader.core.models import Frozen


class BacktestProfile(Frozen):
    strategy_id: str
    strategy_version: str
    trade_r: tuple[float, ...] = Field(min_length=1)  # out-of-sample trade R multiples
    weekly_entries: tuple[int, ...] = Field(min_length=1)  # entries per calendar week, zero weeks included
    mc_dd_p95_r: float = Field(gt=0)  # 95th percentile Monte Carlo max drawdown, in R
    model_slippage: dict[str, float] = Field(default_factory=dict)  # expected adverse slippage per fill
    source: str  # validation report / data version it came from
    synthetic: bool

    @property
    def avg_r(self) -> float:
        return sum(self.trade_r) / len(self.trade_r)

    @property
    def win_rate(self) -> float:
        return sum(r > 0 for r in self.trade_r) / len(self.trade_r)
