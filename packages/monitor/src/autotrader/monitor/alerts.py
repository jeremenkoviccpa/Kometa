"""Alert routing (spec section 16): Telegram first, email as fallback; critical alerts repeat every
10 minutes until acknowledged with `/ack` in the chat (or through the API).

Every alert is also written to the audit log. Delivery failures never raise into the caller: an alert
that could not be sent by either channel stays queued and is retried on the next tick.
"""

from __future__ import annotations

import asyncio
import smtplib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Protocol

import httpx

from autotrader.core.alerts import Alert, Severity
from autotrader.core.clock import Clock
from autotrader.monitor.audit import AuditLog

REPEAT = timedelta(minutes=10)
ICON = {Severity.CRITICAL: "🔴", Severity.WARNING: "🟠", Severity.INFO: "🔵"}


class Notifier(Protocol):
    async def notify(self, text: str) -> bool: ...


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, client: httpx.AsyncClient | None = None) -> None:
        self._base = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self.client = client or httpx.AsyncClient(timeout=10)
        self._offset = 0

    async def notify(self, text: str) -> bool:
        try:
            r = await self.client.post(
                f"{self._base}/sendMessage", json={"chat_id": self.chat_id, "text": text}
            )
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    async def acks(self) -> list[str]:
        """Alert ids acknowledged in the chat since the last poll ("/ack" alone acknowledges all)."""
        try:
            r = await self.client.get(
                f"{self._base}/getUpdates", params={"offset": self._offset, "timeout": 0}
            )
            data = r.json()
        except (httpx.HTTPError, ValueError):
            return []
        out: list[str] = []
        for u in data.get("result", []):
            self._offset = max(self._offset, int(u["update_id"]) + 1)
            msg = u.get("message") or {}
            if str((msg.get("chat") or {}).get("id")) != str(self.chat_id):
                continue  # only the owner's chat may acknowledge
            text = str(msg.get("text", "")).strip()
            if text == "/ack":
                out.append("*")
            elif text.startswith("/ack "):
                out.append(text.split(maxsplit=1)[1])
        return out


class EmailNotifier:
    def __init__(
        self, host: str, port: int, sender: str, to: str, user: str | None = None, password: str | None = None
    ) -> None:
        self.host, self.port, self.sender, self.to = host, port, sender, to
        self.user, self.password = user, password

    def _send(self, text: str) -> None:
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = self.sender, self.to, text.splitlines()[0][:120]
        m.set_content(text)
        with smtplib.SMTP(self.host, self.port, timeout=10) as s:
            s.starttls()
            if self.user and self.password:
                s.login(self.user, self.password)
            s.send_message(m)

    async def notify(self, text: str) -> bool:
        try:
            await asyncio.to_thread(self._send, text)
        except (OSError, smtplib.SMTPException):
            return False
        return True


@dataclass
class _Active:
    alert: Alert
    last_sent: datetime | None = None


@dataclass
class AlertRouter:
    """AlertSink for the monitor: keeps history, audits, delivers, repeats criticals until acked."""

    clock: Clock
    primary: Notifier | None = None
    fallback: Notifier | None = None
    audit: AuditLog | None = None
    history: deque[tuple[str, Alert]] = field(default_factory=lambda: deque(maxlen=500))
    active: dict[str, _Active] = field(default_factory=dict)
    outbox: list[tuple[str, Alert]] = field(default_factory=list)
    _n: int = 0

    def send(self, alert: Alert) -> None:
        self._n += 1
        aid = f"A{self._n}"
        self.history.append((aid, alert))
        if self.audit is not None:
            self.audit.record("Alert", "monitor", {"id": aid, **alert.model_dump(mode="json")})
        if alert.severity == Severity.CRITICAL:
            self.active[aid] = _Active(alert)
        else:
            self.outbox.append((aid, alert))

    def ack(self, alert_id: str, actor: str = "owner") -> int:
        ids = list(self.active) if alert_id == "*" else [alert_id]
        n = sum(1 for i in ids if self.active.pop(i, None) is not None)
        if n and self.audit is not None:
            self.audit.record("AlertAcknowledged", actor, {"ids": ids})
        return n

    @staticmethod
    def text(aid: str, a: Alert) -> str:
        extra = " ".join(f"{k}={v}" for k, v in sorted(a.details.items()))
        ack = f"\n/ack {aid}" if a.severity == Severity.CRITICAL else ""
        return f"{ICON[a.severity]} [{aid}] {a.kind}: {a.message} {extra}".rstrip() + ack

    async def _deliver(self, text: str) -> bool:
        for n in (self.primary, self.fallback):
            if n is not None and await n.notify(text):
                return True
        return self.primary is None and self.fallback is None  # nothing configured: nothing to retry

    async def tick(self) -> int:
        """Send queued alerts, (re)send unacknowledged criticals, read /ack. Returns messages sent."""
        acks = getattr(self.primary, "acks", None)
        if acks is not None:
            for aid in await acks():
                self.ack(aid, actor="owner:telegram")
        sent = 0
        now = self.clock.now()
        for aid, st in list(self.active.items()):
            due = st.last_sent is None or now - st.last_sent >= REPEAT
            if due and await self._deliver(self.text(aid, st.alert)):
                st.last_sent = now
                sent += 1
        keep = []
        for aid, a in self.outbox:
            if await self._deliver(self.text(aid, a)):
                sent += 1
            else:
                keep.append((aid, a))
        self.outbox = keep
        return sent
