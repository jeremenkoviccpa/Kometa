"""L8 cost-model calibration (spec 14.11): backtests are only as honest as their costs.

A calibration measures, from Kometa's own data, the spread by hour of week (the quotes it receives) and the
slippage per market as a multiple of the spread at the fill (the execution-quality log: requested vs filled,
adverse positive), the two numbers the backtest cost model uses. Updates are asymmetric:
- a version with higher costs in every market it covers activates at once;
- a cheaper one waits: it needs at least 200 fills in each market it makes cheaper and four weekly
  calibrations in a row that are all cheaper, and the owner is told every week while it waits.
Versions are recorded with their evidence and a hash (cost_registry.jsonl); the active one can drive
validation. SPEC-QUESTION: re-running every micro+ version under a newly active model is announced (an alert
lists them) until the services run as a deployment (open question 39).
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from autotrader.core.broker import ExecutionQuality
from autotrader.engine.costs import DEFAULT_SLIPPAGE_MULT, hour_of_week

MIN_FILLS = 200
WEEKS_OF_EVIDENCE = 4
BUCKET = 300  # spread samples kept per market and hour of week


class SpreadSamples:
    """Recent spreads per market and hour of week (the live quotes)."""

    def __init__(self) -> None:
        self.buckets: dict[str, list[deque[float]]] = {}

    def add(self, symbol: str, t_ns: int, spread: float) -> None:
        b = self.buckets.setdefault(symbol, [deque(maxlen=BUCKET) for _ in range(168)])
        b[hour_of_week(t_ns)].append(spread)

    def medians(self) -> dict[str, list[float | None]]:
        return {s: [median(d) if d else None for d in b] for s, b in self.buckets.items()}

    def count(self, symbol: str) -> int:
        return sum(len(d) for d in self.buckets.get(symbol, []))


def slippage_mults(fills: Iterable[ExecutionQuality]) -> tuple[dict[str, float], dict[str, int]]:
    """Mean slippage / spread at the fill per market, and the number of fills behind it."""
    by: dict[str, list[float]] = {}
    for q in fills:
        if q.slippage is None or q.spread_at_fill is None or q.spread_at_fill <= 0:
            continue
        by.setdefault(q.symbol, []).append(float(q.slippage / q.spread_at_fill))
    return {s: sum(v) / len(v) for s, v in by.items()}, {s: len(v) for s, v in by.items()}


@dataclass(frozen=True)
class CostVersion:
    version_id: str
    created_at: str
    spread_by_hour: dict[str, list[float | None]]
    slippage_mult: dict[str, float]
    evidence: dict[str, dict[str, int]]  # symbol -> quotes, fills
    status: str = "candidate"  # active | pending (cheaper, waiting for evidence) | superseded
    reason: str = ""
    cheaper_weeks: int = 0

    def cost(self, symbol: str) -> float | None:
        """One number per market to compare versions: mean spread over the hours seen x (1 + slippage)."""
        hours = [x for x in self.spread_by_hour.get(symbol, []) if x is not None]
        if not hours:
            return None
        return sum(hours) / len(hours) * (1.0 + self.slippage_mult.get(symbol, DEFAULT_SLIPPAGE_MULT))


def calibrate(samples: SpreadSamples, fills: Sequence[ExecutionQuality], now: datetime) -> CostVersion:
    mults, n_fills = slippage_mults(fills)
    spreads = samples.medians()
    body = {"spread_by_hour": spreads, "slippage_mult": mults}
    vid = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    evidence = {
        s: {"quotes": samples.count(s), "fills": n_fills.get(s, 0)} for s in sorted({*spreads, *mults})
    }
    return CostVersion(vid, now.isoformat(), spreads, mults, evidence)


@dataclass
class Decision:
    version: CostVersion
    activate: bool
    alert: str | None
    revalidate: list[str] = field(default_factory=list)


def decide(new: CostVersion, active: CostVersion | None, history: Sequence[CostVersion] = ()) -> Decision:
    """The asymmetric rule. `history`: earlier calibrations, newest last."""
    if active is None:
        v = CostVersion(**{**asdict(new), "status": "active", "reason": "first calibration"})
        return Decision(v, True, None)
    cheaper = [
        s
        for s in new.spread_by_hour
        if (a := active.cost(s)) is not None and (b := new.cost(s)) is not None and b < a
    ]
    if not cheaper:
        v = CostVersion(
            **{**asdict(new), "status": "active", "reason": "costs higher or equal: activates at once"}
        )
        return Decision(v, True, None)
    streak = 1
    for old in reversed(history):
        if old.status == "pending":
            streak += 1
        else:
            break
    enough = all(new.evidence.get(s, {}).get("fills", 0) >= MIN_FILLS for s in cheaper)
    if enough and streak >= WEEKS_OF_EVIDENCE:
        v = CostVersion(
            **{
                **asdict(new),
                "status": "active",
                "reason": f"cheaper in {', '.join(cheaper)} for {streak} weeks",
            }
        )
        return Decision(v, True, f"cost model {new.version_id} activated: cheaper in {', '.join(cheaper)}")
    why = (
        f"cheaper in {', '.join(cheaper)}: needs {MIN_FILLS} fills each and {WEEKS_OF_EVIDENCE} weeks "
        f"(week {streak})"
    )
    v = CostVersion(**{**asdict(new), "status": "pending", "reason": why, "cheaper_weeks": streak})
    return Decision(v, False, f"cost model {new.version_id} waiting: {why}")


class CostRegistry:
    """Append-only record of calibrations; the newest active one is in force."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def versions(self) -> list[CostVersion]:
        if not self.path.exists():
            return []
        return [CostVersion(**json.loads(x)) for x in self.path.read_text().splitlines() if x.strip()]

    def active(self) -> CostVersion | None:
        act = [v for v in self.versions() if v.status == "active"]
        return act[-1] if act else None

    def add(self, v: CostVersion) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(v)) + "\n")


def overrides(v: CostVersion) -> tuple[Mapping[str, list[float | None]], Mapping[str, float]]:
    """What validation takes from a version: spread by hour (None = keep the data's) and slippage."""
    return v.spread_by_hour, v.slippage_mult


def view(reg: CostRegistry) -> dict[str, Any]:
    vs = reg.versions()
    act = reg.active()
    return {
        "active": None
        if act is None
        else {
            k: getattr(act, k) for k in ("version_id", "created_at", "reason", "slippage_mult", "evidence")
        },
        "latest": None
        if not vs
        else {k: getattr(vs[-1], k) for k in ("version_id", "created_at", "status", "reason", "evidence")},
        "versions": len(vs),
    }
