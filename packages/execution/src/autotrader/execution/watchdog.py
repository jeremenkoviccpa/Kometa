"""Watchdog (spec section 12).

Every service sends a heartbeat every 30 seconds. If a watched service (engine, risk gate) is silent
for 2 minutes, the watchdog cancels every pending entry order the system placed, directly through the
adapter, and alerts. Open positions stay protected by their broker-side stops. A service that never
sent a heartbeat counts as silent from the moment the watchdog started.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from autotrader.core.alerts import Severity
from autotrader.core.timeutil import ensure_utc
from autotrader.execution.adapter import BrokerUnavailableError
from autotrader.execution.order_manager import OrderManager


class Watchdog:
    def __init__(self, om: OrderManager, started_at: datetime) -> None:
        self.om = om
        self.timeout = timedelta(seconds=om.cfg.watchdog_timeout_seconds)
        self.started_at = ensure_utc(started_at)
        self.last_beat: dict[str, datetime] = {}
        self.tripped: set[str] = set()
        self._cancel_done = True

    def beat(self, service: str, at: datetime) -> None:
        self.last_beat[service] = ensure_utc(at)

    def silent(self, now: datetime) -> list[str]:
        return [
            s
            for s in self.om.cfg.watched_services
            if now - self.last_beat.get(s, self.started_at) > self.timeout
        ]

    async def check(self) -> list[str]:
        """Returns the services currently considered lost."""
        now = self.om.clock.now()
        lost = self.silent(now)
        for s in lost:
            if s not in self.tripped:
                self.tripped.add(s)
                self._cancel_done = False
                self.om.alert(
                    Severity.CRITICAL, "heartbeat_lost", f"{s} silent for {self.timeout}", service=s
                )
        for s in sorted(self.tripped - set(lost)):
            self.tripped.discard(s)
            self.om.alert(Severity.INFO, "heartbeat_restored", f"{s} is back", service=s)
        if self.tripped and not self._cancel_done:
            try:
                await self.om.cancel_pending_entries("watchdog: " + ",".join(sorted(self.tripped)))
                self._cancel_done = True
            except BrokerUnavailableError as e:  # retried on the next check
                self.om.alert(Severity.CRITICAL, "watchdog_cancel_failed", str(e))
        return sorted(self.tripped)
