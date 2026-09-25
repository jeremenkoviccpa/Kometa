"""What Claude may answer, and the hard rules an answer must pass in code before anything is sent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CHECKLIST = (
    "htf_bias",
    "poi",
    "liquidity_sweep",
    "choch_mss",
    "displacement",
    "entry_confirmation",
    "logical_stop",
    "rr_at_least_3",
)


class AiDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: Literal["take", "skip"]
    side: Literal["buy", "sell"] | None = None
    stop: float | None = None
    target: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=6000)
    checklist: dict[str, bool] = Field(default_factory=dict)


# The tool Claude must answer with (a JSON schema without references, as the API expects).
DECIDE_TOOL: dict[str, Any] = {
    "name": "decide",
    "description": "Your trading decision for this moment. Call it exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["take", "skip"]},
            "side": {"type": "string", "enum": ["buy", "sell"], "description": "Required when taking."},
            "stop": {"type": "number", "description": "Stop-loss price at the logical invalidation."},
            "target": {"type": "number", "description": "Take-profit price at real liquidity."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string", "description": "Short: bias, POI, sweep, CHOCH, target, RR."},
            "checklist": {
                "type": "object",
                "properties": {k: {"type": "boolean"} for k in CHECKLIST},
                "required": list(CHECKLIST),
            },
        },
        "required": ["action", "confidence", "reasoning", "checklist"],
    },
}


@dataclass(frozen=True)
class Accepted:
    side: Literal["buy", "sell"]
    entry: float
    stop: float
    target: float
    rr: float


def check(
    d: AiDecision,
    bid: float,
    ask: float,
    *,
    min_rr: float,
    max_stop: float,
    side: Literal["buy", "sell"] | None = None,
    sweep: float | None = None,
) -> Accepted | str:
    """The method's hard rules at the price the order would really get. Returns why not, if not.

    `side`/`sweep` (judge track): Claude may not turn a setup around, and the stop stays beyond the sweep."""
    if d.action != "take":
        return "skip"
    if d.side is None or d.stop is None or d.target is None:
        return "a trade needs side, stop and target"
    missing = [k for k in CHECKLIST if not d.checklist.get(k, False)]
    if missing:
        return f"Claude's own checklist is incomplete ({', '.join(missing)}): NO TRADE"
    if side is not None and d.side != side:
        return f"the setup is a {side}; Claude may skip it, not turn it around"
    buy = d.side == "buy"
    entry = ask if buy else bid
    risk = entry - d.stop if buy else d.stop - entry
    reward = d.target - entry if buy else entry - d.target
    if not risk > 0:
        return "the stop is on the wrong side of the price"
    if not reward > 0:
        return "the target is on the wrong side of the price"
    if sweep is not None and (d.stop > sweep if buy else d.stop < sweep):
        return "the stop must stay beyond the liquidity sweep"
    if risk > max_stop:
        return f"stop too wide ({risk:.2f} > {max_stop:.2f}): NO TRADE"
    rr = reward / risk
    if rr < min_rr:
        return f"RR {rr:.2f} at the current price is below {min_rr:g}: NO TRADE"
    return Accepted(d.side, entry, d.stop, d.target, rr)
