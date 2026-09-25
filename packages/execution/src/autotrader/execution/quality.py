"""Execution quality log (spec section 12). JSONL now; the `execution_quality` table once Postgres runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from autotrader.core.broker import ExecutionQuality


class QualitySink(Protocol):
    def record(self, q: ExecutionQuality) -> None: ...


@dataclass
class MemoryQualityLog:
    rows: list[ExecutionQuality] = field(default_factory=list)

    def record(self, q: ExecutionQuality) -> None:
        self.rows.append(q)


class JsonlQualityLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, q: ExecutionQuality) -> None:
        with self.path.open("a") as f:
            f.write(q.model_dump_json() + "\n")
