"""Persistent risk state: halt state, loss references, decision sequence, used resume nonces.

Halts survive restarts: the state file is written atomically on every change,
and an unreadable or tampered file loads as FULL_HALT (fail closed).
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from autotrader.core.fileio import atomic_write_text
from autotrader.core.hashing import canonical_json, sha256_hex
from autotrader.core.models import HaltState, UtcDatetime

SEVERITY = {
    HaltState.NORMAL: 0,
    HaltState.RECON_HALT: 1,
    HaltState.DAILY_HALT: 2,
    HaltState.WEEKLY_HALT: 3,
    HaltState.FULL_HALT: 4,
}


class RiskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    halt: HaltState = HaltState.NORMAL
    halt_reason: str = ""
    halt_since: UtcDatetime | None = None
    day_ref_equity: Decimal | None = None
    week_ref_equity: Decimal | None = None
    peak_eod_equity: Decimal | None = None
    sequence: int = 0
    used_nonces: list[str] = Field(default_factory=list)


class StateStore:
    """JSON file with an embedded content hash; atomic replace on save."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> RiskState:
        if not self.path.exists():
            return RiskState()
        try:
            raw = json.loads(self.path.read_text())
            body, digest = raw["state"], raw["sha256"]
            if sha256_hex(canonical_json(body)) != digest:
                raise ValueError("state hash mismatch")
            return RiskState.model_validate(body)
        except (ValueError, KeyError, TypeError) as e:
            return RiskState(halt=HaltState.FULL_HALT, halt_reason=f"risk state unreadable: {e}")

    def save(self, state: RiskState) -> None:
        body = state.model_dump(mode="json")
        doc = {"state": body, "sha256": sha256_hex(canonical_json(body))}
        atomic_write_text(self.path, json.dumps(doc, sort_keys=True))


def escalate(state: RiskState, new: HaltState, reason: str, now: datetime) -> bool:
    """Move to a more severe halt. Returns True if the state changed."""
    if SEVERITY[new] <= SEVERITY[state.halt]:
        return False
    state.halt, state.halt_reason, state.halt_since = new, reason, now
    return True
