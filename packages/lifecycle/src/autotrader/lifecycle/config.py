"""Promotion ladder thresholds from config/promotion.yaml (spec section 10). Learning cannot edit it."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from autotrader.core.configs import read_config


class _M(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ShadowExit(_M):
    min_weeks: float = Field(gt=0)
    min_signals: int = Field(gt=0)
    band: float = Field(gt=0, lt=1)


class MicroExit(_M):
    min_trades: int = Field(gt=0)
    max_slippage_vs_model: float = Field(gt=0)
    band: float = Field(gt=0, lt=1)


class LiveExit(_M):
    min_trades: int = Field(gt=0)
    min_rolling_sharpe: float


class ShadowStage(_M):
    risk_per_trade: float = Field(ge=0, le=0)  # shadow never risks money
    exit: ShadowExit


class MicroStage(_M):
    risk_per_trade: float = Field(gt=0)
    exit: MicroExit


class LiveStage(_M):
    risk_per_trade: float = Field(gt=0)
    exit: LiveExit


class ScaledStage(_M):
    risk_per_trade_max: float = Field(gt=0)


class Demotion(_M):
    dd_vs_mc_p95: float = Field(gt=0)
    rolling_pf_window: int = Field(gt=0)
    rolling_pf_min: float = Field(gt=0)
    slippage_vs_model_max: float = Field(gt=0)
    slippage_window: int = Field(gt=0)
    drift_window: int = Field(gt=0)
    drift_interval: float = Field(gt=0, lt=1)
    retire_after_demotions: int = Field(ge=1)
    retire_window_months: int = Field(ge=1)


class GlobalLimits(_M):
    max_promotions_to_live_per_week: int = Field(ge=0)


class PromotionConfig(_M):
    shadow: ShadowStage
    micro: MicroStage
    live: LiveStage
    scaled: ScaledStage
    demotion: Demotion
    global_: GlobalLimits = Field(alias="global")


def load_promotion_config(path: Path) -> PromotionConfig:
    return PromotionConfig.model_validate(yaml.safe_load(read_config(path)))
