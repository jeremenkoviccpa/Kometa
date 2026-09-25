"""Allocator settings (config/allocator.yaml) plus the stage risk limits from config/promotion.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from autotrader.core.configs import read_config
from autotrader.core.models import Stage


class AllocatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shrinkage_k: float = Field(gt=0)
    correlation_cluster_threshold: float = Field(gt=0, le=1)
    min_overlap_days: int = Field(ge=2)
    max_share_per_version: float = Field(gt=0, le=1)
    total_risk_budget: float = Field(gt=0, le=0.05)
    rebalance_days: float = Field(gt=0)
    stage_limits: dict[Stage, float]  # per-trade risk fraction cap per stage

    def stage_limit(self, stage: Stage) -> float:
        return self.stage_limits.get(stage, 0.0)


def load_allocator_config(allocator: Path, promotion: Path) -> AllocatorConfig:
    raw = yaml.safe_load(read_config(allocator)) or {}
    promo = yaml.safe_load(read_config(promotion))
    raw["stage_limits"] = {
        Stage.MICRO: float(promo["micro"]["risk_per_trade"]),
        Stage.LIVE: float(promo["live"]["risk_per_trade"]),
        Stage.SCALED: float(promo["scaled"]["risk_per_trade_max"]),
    }
    return AllocatorConfig.model_validate(raw)
