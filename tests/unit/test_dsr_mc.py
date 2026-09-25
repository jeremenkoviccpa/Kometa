"""DSR against hand-computed values; Monte Carlo properties."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autotrader.validation.dsr import deflated_sharpe, expected_max_sharpe, moments, psr, sharpe
from autotrader.validation.montecarlo import block_bootstrap_indices, max_drawdowns, monte_carlo


def test_psr_hand_computed() -> None:
    # SR=0.1, T=253, normal returns: z = 0.1*sqrt(252)/sqrt(1 + 0.5*0.01) = 1.587451/1.002497 = 1.583497
    assert psr(0.1, 0.0, 253, 0.0, 3.0) == pytest.approx(0.94334, abs=2e-5)
    # negative skew and fat tails reduce confidence
    assert psr(0.1, 0.0, 253, -1.0, 6.0) < psr(0.1, 0.0, 253, 0.0, 3.0)
    assert psr(0.1, 0.1, 253, 0.0, 3.0) == pytest.approx(0.5)


def test_expected_max_sharpe_hand_computed() -> None:
    # N=100, sd=0.05: Phi^-1(0.99)=2.326348, Phi^-1(1-1/(100e))=2.680210
    # SR* = 0.05*((1-0.577216)*2.326348 + 0.577216*2.680210) = 0.05*2.530601 = 0.126530
    assert expected_max_sharpe(0.0025, 100) == pytest.approx(0.126530, abs=1e-6)
    assert expected_max_sharpe(0.0025, 1000) > expected_max_sharpe(0.0025, 100)
    assert expected_max_sharpe(0.0025, 1) == 0.0


def test_moments_and_sharpe() -> None:
    r = np.array([1.0, -1.0, 1.0, -1.0])
    assert sharpe(r) == pytest.approx(0.0)
    skew, kurt = moments(r)
    assert skew == pytest.approx(0.0)
    assert kurt == pytest.approx(1.0)


def test_deflated_sharpe_gets_harder_with_more_trials() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)  # daily SR ~ 0.1
    few = deflated_sharpe(r, rng.normal(0, 0.03, 5))
    many = deflated_sharpe(r, rng.normal(0, 0.03, 2000))
    assert many.sr_star > few.sr_star
    assert many.probability < few.probability
    assert 0.0 <= many.probability <= 1.0
    assert math.isclose(few.sharpe, sharpe(r))


def test_block_indices_are_contiguous_blocks() -> None:
    idx = block_bootstrap_indices(10, 3, 4, np.random.default_rng(1))
    assert idx.shape == (3, 10)
    first_block = idx[0, :4]
    assert all((first_block[i + 1] - first_block[i]) % 10 == 1 for i in range(3))


def test_max_drawdowns() -> None:
    eq = np.array([[1.1, 0.99, 1.2], [0.5, 0.75, 1.0]])
    np.testing.assert_allclose(max_drawdowns(eq), [0.1, 0.5])


def test_block_bootstrap_widens_tails_for_clustered_losses() -> None:
    # losses clustered in streaks: iid resampling breaks the streaks, blocks keep them
    r = np.array(([-1.0] * 8 + [1.5] * 8) * 20)
    iid = monte_carlo(r, runs=3000, method="bootstrap_trades", risk_fraction=0.01, seed=1)
    blk = monte_carlo(r, runs=3000, block_length=8, risk_fraction=0.01, seed=1)
    assert blk.dd_p95 > iid.dd_p95
    assert blk.dd_p50 <= blk.dd_p95 <= blk.dd_p99


def test_monte_carlo_deterministic_with_seed() -> None:
    r = np.random.default_rng(3).normal(0.2, 1.0, 300)
    a = monte_carlo(r, runs=500, seed=7)
    b = monte_carlo(r, runs=500, seed=7)
    assert a.dd_p95 == b.dd_p95
    with pytest.raises(ValueError, match="no trades"):
        monte_carlo([], runs=10)
