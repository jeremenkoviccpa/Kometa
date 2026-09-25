"""Signed risk limits (spec section 11).

The gate refuses to start unless `risk.yaml.sig` is a valid Ed25519
signature of the exact bytes of `risk.yaml` by the owner key baked into the
image. The owner signs on their own machine with `at risk sign`.
"""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, field_validator

from autotrader.core.configs import read_config
from autotrader.core.hashing import sha256_hex
from autotrader.core.signing import verify_bytes

_DAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


class ConfigSignatureError(Exception):
    pass


class WeeklyCutoff(BaseModel):
    model_config = ConfigDict(frozen=True)

    weekday: int
    at: time
    tz: str


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    account_currency: str = Field(min_length=3, max_length=3)
    risk_per_trade_default: Decimal = Field(gt=0)
    risk_per_trade_max: Decimal = Field(gt=0, le=Decimal("0.02"))
    open_risk_per_strategy_max: Decimal = Field(gt=0)
    open_risk_total_max: Decimal = Field(gt=0, le=Decimal("0.10"))
    same_currency_same_direction_max_positions: int = Field(ge=1)
    leverage_notional_max: Decimal = Field(gt=0)
    per_symbol_max_lots: dict[str, Decimal]
    daily_loss_halt: Decimal = Field(gt=0)
    weekly_loss_halt: Decimal = Field(gt=0)
    peak_drawdown_full_halt: Decimal = Field(gt=0)
    news_blackout_minutes: int = Field(ge=0)
    no_new_entries_after: WeeklyCutoff
    min_free_margin_ratio: Decimal = Field(ge=1)
    require_stop: Literal[True] = True  # cannot be switched off, even by a signed config
    decision_ttl_seconds: int = Field(default=5, ge=1, le=60)
    min_stop_spreads: Decimal = Decimal("1.5")
    stage_risk_limits: dict[str, Decimal] = Field(
        default_factory=lambda: {
            "micro": Decimal("0.001"),
            "live": Decimal("0.005"),
            "scaled": Decimal("0.01"),
        }
    )

    @field_validator("no_new_entries_after", mode="before")
    @classmethod
    def _parse_cutoff(cls, v: object) -> object:
        if isinstance(v, str):
            day, hhmm, tz = v.split()
            h, m = hhmm.split(":")
            return {"weekday": _DAYS[day.upper()], "at": time(int(h), int(m)), "tz": tz}
        return v

    def max_lots(self, symbol: str) -> Decimal:
        return self.per_symbol_max_lots.get(symbol, self.per_symbol_max_lots["default"])


def load_signed(config: Path, signature: Path, owner_key: Ed25519PublicKey) -> tuple[RiskLimits, str]:
    """Verify then parse. Returns (limits, sha256 of the file). Any problem raises ConfigSignatureError."""
    if not signature.exists():
        raise ConfigSignatureError(f"missing signature {signature}")
    data = read_config(config)
    if not verify_bytes(owner_key, data, signature.read_text().strip()):
        raise ConfigSignatureError(f"invalid signature for {config}")
    limits = RiskLimits.model_validate(yaml.safe_load(data))
    if "default" not in limits.per_symbol_max_lots:
        raise ConfigSignatureError("per_symbol_max_lots needs a default")
    return limits, sha256_hex(data)
