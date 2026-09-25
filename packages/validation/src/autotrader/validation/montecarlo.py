"""Monte Carlo drawdown via circular block bootstrap of trade R multiples (spec section 9, v1.1).

Blocks of consecutive trades are resampled so streaks and volatility clusters
survive; an i.i.d. bootstrap understates drawdown tails. Trade count is
preserved. Equity compounds at a fixed risk fraction per trade.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    runs: int
    block_length: int
    risk_fraction: float
    dd_p50: float
    dd_p95: float
    dd_p99: float
    final_return_p5: float
    final_return_p50: float
    drawdowns: np.ndarray  # per run, for the report histogram


def auto_block_length(n: int) -> int:
    cube_root: float = float(n) ** (1.0 / 3.0)
    return max(1, round(cube_root))


def block_bootstrap_indices(n: int, runs: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(runs, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(runs, n_blocks * block)[:, :n]


def max_drawdowns(equity: np.ndarray) -> np.ndarray:
    """Row-wise max drawdown fraction; equity rows start after the initial 1.0."""
    full = np.concatenate([np.ones((equity.shape[0], 1)), equity], axis=1)
    peak = np.maximum.accumulate(full, axis=1)
    return np.asarray(((peak - full) / peak).max(axis=1))


def monte_carlo(
    r_multiples: Sequence[float] | np.ndarray,
    *,
    runs: int = 5000,
    risk_fraction: float = 0.005,
    block_length: int | None = None,
    method: str = "block_bootstrap_trades",
    seed: int = 0,
) -> MonteCarloResult:
    r = np.asarray(r_multiples, dtype=np.float64)
    if r.size == 0:
        raise ValueError("no trades")
    rng = np.random.default_rng(seed)
    if method == "block_bootstrap_trades":
        block = block_length or auto_block_length(r.size)
    elif method == "bootstrap_trades":
        block = 1
    else:
        raise ValueError(f"unknown Monte Carlo method {method!r}")
    idx = block_bootstrap_indices(r.size, runs, block, rng)
    growth = np.maximum(1.0 + risk_fraction * r[idx], 0.0)  # cannot lose more than everything
    equity = np.cumprod(growth, axis=1)
    dd = max_drawdowns(equity)
    final = equity[:, -1] - 1.0
    return MonteCarloResult(
        runs=runs,
        block_length=block,
        risk_fraction=risk_fraction,
        dd_p50=float(np.quantile(dd, 0.50)),
        dd_p95=float(np.quantile(dd, 0.95)),
        dd_p99=float(np.quantile(dd, 0.99)),
        final_return_p5=float(np.quantile(final, 0.05)),
        final_return_p50=float(np.quantile(final, 0.50)),
        drawdowns=dd,
    )
