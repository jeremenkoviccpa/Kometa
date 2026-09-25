"""Alerts (spec section 16). Producers depend on this interface; `monitor` delivers them (Telegram, email)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pydantic import Field

from autotrader.core.models import Frozen, UtcDatetime


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Alert(Frozen):
    severity: Severity
    kind: str  # stable machine name, e.g. "missing_stop", "recon_mismatch", "heartbeat_lost"
    message: str
    at: UtcDatetime
    details: dict[str, str] = Field(default_factory=dict)


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


@dataclass
class MemoryAlertSink:
    """Collects alerts in memory. Tests, and a buffer until the monitor service exists."""

    alerts: list[Alert] = field(default_factory=list)

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def kinds(self, severity: Severity | None = None) -> list[str]:
        return [a.kind for a in self.alerts if severity is None or a.severity == severity]


@dataclass
class QueueAlertSink:
    """Buffers alerts for a service that forwards them to the monitor over the bus (flushed each cycle)."""

    queue: list[Alert] = field(default_factory=list)

    def send(self, alert: Alert) -> None:
        self.queue.append(alert)

    def drain(self) -> list[Alert]:
        out, self.queue = self.queue, []
        return out


@dataclass
class FanOutAlertSink:
    """Sends every alert to several sinks (e.g. the in-process monitor and a local buffer)."""

    sinks: list[AlertSink] = field(default_factory=list)

    def send(self, alert: Alert) -> None:
        for s in self.sinks:
            s.send(alert)
