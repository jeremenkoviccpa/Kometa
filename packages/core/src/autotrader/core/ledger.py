"""Append-only, tamper-evident JSON Lines ledger (spec sections 9, 10 and 16).

One JSON record per line, each carrying `prev_hash` and `hash = sha256(prev_hash + canonical body)`.
There is no update or delete API. Used by the trial registry, the lifecycle registry and (Phase 8)
the audit log; the Postgres implementation keeps the same interface.
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from autotrader.core.hashing import canonical_json, sha256_hex

GENESIS = "0" * 64


class LedgerCorruptError(Exception):
    pass


class Ledger(Protocol):
    def append(self, kind: str, payload: dict[str, Any]) -> str: ...

    def records(self, kind: str | None = None) -> Iterator[dict[str, Any]]: ...


class JsonlLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        self._tail: tuple[int, str] | None = None  # (file size, last hash) after our own last write

    def _last_hash(self) -> str:
        size = self.path.stat().st_size
        if self._tail is not None and self._tail[0] == size:
            return self._tail[1]  # nobody else appended since: no need to re-read the file
        last = GENESIS
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def append(self, kind: str, payload: dict[str, Any]) -> str:
        with self.path.open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                prev = self._last_hash()
                body = {"kind": kind, "at": datetime.now(UTC).isoformat(), "payload": payload}
                h = sha256_hex(prev + canonical_json(body))
                f.write(canonical_json({**body, "prev_hash": prev, "hash": h}) + "\n")
                f.flush()
                self._tail = (self.path.stat().st_size, h)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return h

    def records(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if kind is None or rec["kind"] == kind:
                        yield rec

    def verify(self) -> int:
        """Walk the chain; raise LedgerCorruptError on the first break. Returns the record count."""
        prev, n = GENESIS, 0
        for rec in self.records():
            body = {k: rec[k] for k in ("kind", "at", "payload")}
            if rec["prev_hash"] != prev or rec["hash"] != sha256_hex(prev + canonical_json(body)):
                raise LedgerCorruptError(f"chain broken at record {n}")
            prev, n = rec["hash"], n + 1
        return n
