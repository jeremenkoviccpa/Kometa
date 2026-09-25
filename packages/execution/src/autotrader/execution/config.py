"""Execution tunables from config/execution.yaml."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from autotrader.core.configs import read_config


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reconcile_interval_seconds: float = Field(default=60, gt=0)
    heartbeat_interval_seconds: float = Field(default=30, gt=0)
    watchdog_timeout_seconds: float = Field(default=120, gt=0)
    watched_services: tuple[str, ...] = ("engine", "risk-gate")
    stop_confirm_attempts: int = Field(default=2, ge=1)
    max_quote_age_seconds: float = Field(default=10, gt=0)
    send_grace_seconds: float = Field(default=120, gt=0)
    balance_tolerance: Decimal = Field(default=Decimal("0.01"), ge=0)
    deal_lookback_seconds: float = Field(default=300, ge=0)


def load_execution_config(path: Path) -> ExecutionConfig:
    raw = yaml.safe_load(read_config(path)) or {}
    # YAML floats go through str before Decimal (Phase 1 lesson)
    if "balance_tolerance" in raw:
        raw["balance_tolerance"] = Decimal(str(raw["balance_tolerance"]))
    return ExecutionConfig.model_validate(raw)
