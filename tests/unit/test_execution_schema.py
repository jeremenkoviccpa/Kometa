"""Schema drift check: tables written from Pydantic models have exactly the models' columns."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from autotrader.core.broker import ExecutionQuality
from autotrader.core.models import Fill, Order

MIGRATION = Path(__file__).resolve().parents[2] / "migrations" / "versions" / "0003_execution.py"


def table_columns(sql: str, table: str) -> set[str]:
    m = re.search(rf"CREATE TABLE {table} \((.*?)\n\s*\)\n", sql, flags=re.S)
    assert m, f"table {table} not found"
    cols = set()
    for line in m.group(1).splitlines():
        word = line.strip().split(" ", 1)[0]
        if word and word.islower() and word not in {"primary"} and not line.strip().startswith("PRIMARY"):
            cols.add(word)
    return cols


@pytest.mark.parametrize(
    ("table", "model"), [("orders", Order), ("fills", Fill), ("execution_quality", ExecutionQuality)]
)
def test_table_matches_model(table: str, model: type[BaseModel]) -> None:
    assert table_columns(MIGRATION.read_text(), table) == set(model.model_fields)
