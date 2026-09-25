"""Audit log (spec section 16): append-only, hash-chained; every decision-relevant event is recorded.

Dev and CI use the JSON Lines ledger (`core.ledger`); in Postgres the same records go to `audit_log`,
where a trigger rejects UPDATE and DELETE (migration 0004). `at audit verify` walks the chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from autotrader.core import bus as streams
from autotrader.core.alerts import Alert, AlertSink, Severity
from autotrader.core.bus import Handler
from autotrader.core.events import ConfigChanged
from autotrader.core.hashing import canonical_json
from autotrader.core.ledger import Ledger, LedgerCorruptError

# stream -> the service that writes to it (the audit actor)
ACTORS = {
    streams.SIGNALS: "engine",
    streams.INTENTS: "allocator",
    streams.DECISIONS: "risk-gate",
    streams.HALTS: "risk-gate",
    streams.TRADES: "execution",
    streams.ORDERS: "execution",
    streams.STAGES: "lifecycle",
    streams.CONTROL: "lifecycle",
    streams.REQUESTS: "engine",
    streams.ALERTS: "monitor",
    streams.CONFIG: "config",  # the reading service, from the message
}
# high-volume streams that are state, not decisions: not audited
NOT_AUDITED = {streams.QUOTES, streams.ACCOUNT, streams.HEARTBEATS}


class AuditLog:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def record(self, event_type: str, actor: str, payload: Mapping[str, Any]) -> str:
        return self.ledger.append(event_type, {"actor": actor, "data": dict(payload)})

    def tail(self, n: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.ledger.records(event_type))
        return rows[-n:]

    def verify(self) -> int:
        verify = getattr(self.ledger, "verify", None)
        if verify is None:
            raise NotImplementedError("this ledger cannot verify its chain")
        n: int = verify()
        return n


def check_chain(log: AuditLog, alerts: AlertSink, now: datetime) -> int | None:
    """Scheduled job: verify the chain; a break is a critical alert (spec section 16)."""
    try:
        return log.verify()
    except LedgerCorruptError as e:
        alerts.send(Alert(severity=Severity.CRITICAL, kind="audit_chain_break", message=str(e), at=now))
        return None


class AuditService:
    """Consumes every audited stream and writes each message to the audit log."""

    name = "audit"

    def __init__(self, log: AuditLog) -> None:
        self.log = log

    def handlers(self) -> Mapping[str, Handler]:
        return {s: self._writer(s) for s in ACTORS}

    def _writer(self, stream: str) -> Handler:
        async def write(msg: Any) -> None:
            payload = msg.model_dump(mode="json")
            canonical_json(payload)  # fail loudly on anything not serializable
            actor = msg.service if isinstance(msg, ConfigChanged) else ACTORS[stream]
            self.log.record(msg.kind, actor, payload)

        return write
