"""Learning safety switches (spec 14.9).

- The owner's freeze (a file, the hub's Pause learning button) stops every learning loop without touching
  trading: no challenger swaps, no lessons, the offline `at learn` commands refuse.
- A total equity drawdown above 8% pauses loops L1-L4 (discovery, re-optimization, meta-labeling, regime)
  until equity is back within 4% of its peak: learning during a drawdown tends to chase noise.

The drawdown pause is persisted, with its reason, before it takes effect (a restart must remember why).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PAUSE_DD, RESUME_DD = 0.08, 0.04
DRAWDOWN_LOOPS = frozenset({"L1", "L2", "L3", "L4"})


@dataclass
class LearningGuard:
    freeze_path: Path
    state_path: Path | None = None
    pause_dd: float = PAUSE_DD
    resume_dd: float = RESUME_DD
    dd_paused_since: str | None = None
    dd_reason: str = ""

    def __post_init__(self) -> None:
        if self.state_path is not None and self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text())
                self.dd_paused_since, self.dd_reason = (
                    raw.get("dd_paused_since"),
                    str(raw.get("dd_reason", "")),
                )
            except (OSError, ValueError):
                self.dd_paused_since, self.dd_reason = "unknown", "state file unreadable: paused to be safe"

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"dd_paused_since": self.dd_paused_since, "dd_reason": self.dd_reason}))
        tmp.replace(self.state_path)

    def update(self, drawdown: float, now: datetime) -> None:
        """Feed the account's drawdown from its peak (0.05 = 5%)."""
        if self.dd_paused_since is None and drawdown > self.pause_dd:
            self.dd_reason = f"equity {drawdown:.1%} below its peak (pause above {self.pause_dd:.0%})"
            self.dd_paused_since = now.isoformat()
            self._save()  # persisted before the pause is in effect
        elif self.dd_paused_since is not None and drawdown <= self.resume_dd:
            self.dd_paused_since, self.dd_reason = None, ""
            self._save()

    def frozen_by_owner(self) -> bool:
        return self.freeze_path.exists()

    def paused(self, loop: str) -> str | None:
        """Why `loop` (e.g. "L2") may not act now, or None."""
        if self.frozen_by_owner():
            return "learning frozen by the owner"
        if loop in DRAWDOWN_LOOPS and self.dd_paused_since is not None:
            return f"paused in a drawdown: {self.dd_reason}"
        return None

    def view(self) -> dict[str, object]:
        return {
            "frozen_by_owner": self.frozen_by_owner(),
            "drawdown_paused_since": self.dd_paused_since,
            "drawdown_reason": self.dd_reason,
            "pause_above": self.pause_dd,
            "resume_within": self.resume_dd,
        }
