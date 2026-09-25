"""Trial registry and holdout attempts (spec section 9): append-only, tamper-evident.

Records go to a `core.ledger` Ledger (hash-chained JSON Lines in dev and CI).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from autotrader.core.ledger import GENESIS, JsonlLedger, Ledger, LedgerCorruptError

TrialKind = Literal[
    "wf_train", "wf_test", "full", "stability", "cross_market", "holdout", "challenger", "research"
]


@dataclass(frozen=True)
class Trial:
    family: str
    strategy_id: str
    version: str
    params_hash: str
    kind: TrialKind
    data_version: str
    window_start: str
    window_end: str
    sharpe: float  # daily, not annualized
    trades: int
    passed: bool | None = None
    code_hash: str = ""
    note: str = ""
    tenant_id: str = "default"


@dataclass(frozen=True)
class HoldoutAttempt:
    family: str
    strategy_id: str
    version: str
    epoch: str
    at: str


class TrialRegistry:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def record(self, trial: Trial) -> None:
        self.ledger.append("trial", asdict(trial))

    def trials(self, family: str | None = None) -> list[Trial]:
        out = [Trial(**r["payload"]) for r in self.ledger.records("trial")]
        return [t for t in out if family is None or t.family == family]

    def family_sharpes(self, family: str) -> list[float]:
        return [t.sharpe for t in self.trials(family)]


class HoldoutRefusedError(Exception):
    pass


class HoldoutLock:
    """Opens the locked holdout at most once per version and N times per family per epoch."""

    def __init__(self, ledger: Ledger, epoch: str, max_per_family: int) -> None:
        self.ledger = ledger
        self.epoch = epoch
        self.max_per_family = max_per_family

    def attempts(self) -> list[HoldoutAttempt]:
        return [HoldoutAttempt(**r["payload"]) for r in self.ledger.records("holdout_attempt")]

    def open(self, family: str, strategy_id: str, version: str) -> HoldoutAttempt:
        past = self.attempts()
        if any(a.strategy_id == strategy_id and a.version == version for a in past):
            raise HoldoutRefusedError(f"{strategy_id} {version} already used its holdout attempt")
        used = sum(1 for a in past if a.family == family and a.epoch == self.epoch)
        if used >= self.max_per_family:
            raise HoldoutRefusedError(
                f"family {family} used {used}/{self.max_per_family} holdout attempts this epoch"
            )
        # recorded BEFORE the holdout is evaluated: a crash still consumes the attempt
        att = HoldoutAttempt(family, strategy_id, version, self.epoch, datetime.now(UTC).isoformat())
        self.ledger.append("holdout_attempt", asdict(att))
        return att


__all__ = [
    "GENESIS",
    "HoldoutAttempt",
    "HoldoutLock",
    "JsonlLedger",
    "Ledger",
    "LedgerCorruptError",
    "Trial",
    "TrialRegistry",
]
