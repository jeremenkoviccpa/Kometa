"""Typed view of config/validation.yaml (a fenced file: learning and research never write it)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from autotrader.core.configs import read_config
from autotrader.core.hashing import sha256_hex


class _M(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WalkForwardCfg(_M):
    train_years: float = Field(default=3, gt=0)
    test_months: float = Field(default=6, gt=0)
    step_months: float = Field(default=6, gt=0)
    search_budget: int = Field(default=20, ge=1)
    min_train_trades: int = Field(default=10, ge=1)


class HoldoutCfg(_M):
    months: float = Field(default=12, gt=0)
    max_attempts_per_version: int = 1
    max_attempts_per_family_per_epoch: int = 5


class MonteCarloCfg(_M):
    runs: int = 5000
    method: Literal["block_bootstrap_trades", "bootstrap_trades"] = "block_bootstrap_trades"
    block_length: int | Literal["auto"] = "auto"


class StabilityCfg(_M):
    param_shift: float = 0.20


class CrossMarketCfg(_M):
    min_pairs_passing: int = 3
    of_pairs: int = 5


class Thresholds(_M):
    min_oos_trades: int = 200
    min_profit_factor: float = 1.3
    min_deflated_sharpe_prob: float = 0.95
    max_mc_dd_p95_at_half_percent_risk: float = 0.12
    min_stability_profit_factor: float = 1.1
    min_holdout_profit_factor: float = 1.1
    suspicious_monthly_return: float = 0.10


class ValidationConfig(_M):
    walk_forward: WalkForwardCfg = WalkForwardCfg()
    holdout: HoldoutCfg = HoldoutCfg()
    monte_carlo: MonteCarloCfg = MonteCarloCfg()
    stability: StabilityCfg = StabilityCfg()
    cross_market: CrossMarketCfg = CrossMarketCfg()
    thresholds: Thresholds = Thresholds()

    @staticmethod
    def load(path: Path) -> tuple[ValidationConfig, str]:
        data = read_config(path)
        return ValidationConfig.model_validate(yaml.safe_load(data)), sha256_hex(data)
