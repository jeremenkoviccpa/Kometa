"""Strategy manifest (`strategy.yaml`, spec section 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autotrader.core.models import Timeframe

MAX_TUNABLE_PARAMS = 6

ParamValue = float | int | str | bool


class ParamSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: ParamValue
    min: float | None = None
    max: float | None = None
    tunable: bool = False

    @model_validator(mode="after")
    def _check(self) -> ParamSpec:
        if self.tunable:
            if isinstance(self.value, (str, bool)):
                raise ValueError("only numeric params can be tunable")
            if self.min is None or self.max is None:
                raise ValueError("tunable params need min and max")
        if self.min is not None and self.max is not None:
            if self.min > self.max:
                raise ValueError("min > max")
            if not isinstance(self.value, (str, bool)) and not (self.min <= self.value <= self.max):
                raise ValueError(f"value {self.value} outside [{self.min}, {self.max}]")
        return self


class Expected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trades_per_month: float = Field(gt=0)
    win_rate: float = Field(ge=0, le=1)
    avg_r: float


class StrategyManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    origin: Literal["trader", "research_agent", "owner", "learning_reopt"]
    family: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    symbols: tuple[str, ...] = Field(min_length=1)
    timeframes: tuple[Timeframe, ...] = Field(min_length=1)
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    expected: Expected
    intake_ref: str | None = None
    description: str = ""
    demo_only: bool = False

    @model_validator(mode="after")
    def _check(self) -> StrategyManifest:
        tunable = [k for k, p in self.params.items() if p.tunable]
        if len(tunable) > MAX_TUNABLE_PARAMS:
            raise ValueError(f"{len(tunable)} tunable params; max is {MAX_TUNABLE_PARAMS}")
        if len(set(self.symbols)) != len(self.symbols) or len(set(self.timeframes)) != len(self.timeframes):
            raise ValueError("duplicate symbols or timeframes")
        return self

    def param_values(self, overrides: dict[str, ParamValue] | None = None) -> dict[str, ParamValue]:
        values = {k: p.value for k, p in self.params.items()}
        for k, v in (overrides or {}).items():
            spec = self.params.get(k)
            if spec is None:
                raise KeyError(f"unknown param {k!r}")
            lo, hi = spec.min, spec.max
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            if numeric and lo is not None and hi is not None and not lo <= float(v) <= hi:
                raise ValueError(f"param {k}={v} outside [{lo}, {hi}]")
            values[k] = v
        return values

    @staticmethod
    def load(path: Path) -> StrategyManifest:
        return StrategyManifest.model_validate(yaml.safe_load(path.read_text()))
