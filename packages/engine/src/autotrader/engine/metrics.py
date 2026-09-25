"""Basic performance metrics from trades and the daily equity curve."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from autotrader.engine.simbroker import TradeRecord


@dataclass(frozen=True)
class Metrics:
    trades: int
    win_rate: float
    avg_r: float
    total_r: float
    profit_factor: float  # in R; inf if no losers
    max_drawdown: float  # fraction of peak equity
    sharpe_daily_ann: float  # sqrt(252) * mean/std of daily returns
    net_pnl: float
    commission: float
    swap: float
    spread_cost: float
    slippage_cost: float


def daily_returns(equity: Sequence[float]) -> np.ndarray:
    e = np.asarray(equity, dtype=np.float64)
    if e.size < 2:
        return np.empty(0)
    return e[1:] / e[:-1] - 1.0


def max_drawdown(equity: Sequence[float]) -> float:
    e = np.asarray(equity, dtype=np.float64)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.max((peak - e) / peak))


def compute(trades: Sequence[TradeRecord], equity: Sequence[float]) -> Metrics:
    r = np.asarray([t.r_multiple for t in trades], dtype=np.float64)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    rets = daily_returns(equity)
    sd = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
    return Metrics(
        trades=int(r.size),
        win_rate=float((r > 0).mean()) if r.size else 0.0,
        avg_r=float(r.mean()) if r.size else 0.0,
        total_r=float(r.sum()),
        profit_factor=float(wins / losses) if losses > 0 else (math.inf if wins > 0 else 0.0),
        max_drawdown=max_drawdown(equity),
        sharpe_daily_ann=float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
        net_pnl=float(sum(t.pnl_net for t in trades)),
        commission=float(sum(t.commission for t in trades)),
        swap=float(sum(t.swap for t in trades)),
        spread_cost=float(sum(t.spread_cost for t in trades)),
        slippage_cost=float(sum(t.slippage_cost for t in trades)),
    )
