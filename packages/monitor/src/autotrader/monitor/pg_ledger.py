"""The audit log in Postgres (table `audit_log`, migrations 0004 and 0005), same chain as JsonlLedger.

Each row stores the canonical JSON of {kind, at, payload} that was hashed, so `verify` recomputes
exactly `hash = sha256(prev_hash + canonical)` and then checks the queryable columns (event_type,
actor, at, payload) still say the same thing. Appends take a transaction-scoped advisory lock, so
several writers (monitor, API) extend one chain without forking it. The database refuses UPDATE,
DELETE and TRUNCATE (trigger) and the application role may only INSERT and SELECT.

Synchronous on purpose: the Ledger interface is synchronous, audit writes are small, and this code
never runs in the order path.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from autotrader.core.hashing import canonical_json, sha256_hex
from autotrader.core.ledger import GENESIS, LedgerCorruptError

APPEND_LOCK = 7_341_902_001  # pg_advisory_xact_lock key for audit_log appends


def sync_url(url: str) -> str:
    """Settings hold the SQLAlchemy asyncpg URL; psycopg wants the plain libpq form."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


class PgLedger:
    def __init__(self, url: str, tenant_id: str = "default") -> None:
        self.url = sync_url(url)
        self.tenant_id = tenant_id

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.url)

    def append(self, kind: str, payload: dict[str, Any]) -> str:
        at = datetime.now(UTC)
        body = {"kind": kind, "at": at.isoformat(), "payload": payload}
        canonical = canonical_json(body)
        with self._connect() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (APPEND_LOCK,))
            row = conn.execute(
                "SELECT hash FROM audit_log WHERE tenant_id = %s ORDER BY id DESC LIMIT 1", (self.tenant_id,)
            ).fetchone()
            prev = row[0] if row else GENESIS
            h = sha256_hex(prev + canonical)
            conn.execute(
                "INSERT INTO audit_log"
                " (tenant_id, at, event_type, actor, payload, prev_hash, hash, canonical)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.tenant_id,
                    at,
                    kind,
                    str(payload.get("actor", "")),
                    Jsonb(payload.get("data", payload)),
                    prev,
                    h,
                    canonical,
                ),
            )
        return h

    def _rows(self, kind: str | None = None) -> Iterator[tuple[Any, ...]]:
        with self._connect() as conn, conn.cursor(name="audit_scan") as cur:
            cur.execute(
                "SELECT id, at, event_type, actor, payload, prev_hash, hash, canonical FROM audit_log"
                " WHERE tenant_id = %s AND (%s::text IS NULL OR event_type = %s) ORDER BY id",
                (self.tenant_id, kind, kind),
            )
            yield from cur

    def records(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Same shape as JsonlLedger records."""
        for *_, prev, h, canonical in self._rows(kind):
            body = json.loads(canonical)
            yield {**body, "prev_hash": prev, "hash": h}

    def verify(self) -> int:
        """Walk the chain; raise LedgerCorruptError on the first break. Returns the record count."""
        prev, n = GENESIS, 0
        for _id, at, event_type, actor, payload, prev_hash, h, canonical in self._rows():
            if prev_hash != prev or h != sha256_hex(prev + canonical):
                raise LedgerCorruptError(f"chain broken at record {n} (id {_id})")
            body = json.loads(canonical)
            p = body["payload"]
            agrees = (
                body["kind"] == event_type
                and str(p.get("actor", "")) == actor
                and datetime.fromisoformat(body["at"]) == at
                and json.loads(canonical_json(p.get("data", p))) == json.loads(canonical_json(payload))
            )
            if not agrees:
                raise LedgerCorruptError(f"columns disagree with the hashed record {n} (id {_id})")
            prev, n = h, n + 1
        return n
