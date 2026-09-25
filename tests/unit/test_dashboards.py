"""Grafana dashboards as code (spec section 16): generated, complete, pointed at the provisioned source."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "grafana" / "provisioning" / "dashboards" / "json"
SPEC_DASHBOARDS = {  # spec section 16, in its words
    "Account overview",
    "Strategy versions by stage",
    "Live vs backtest drift",
    "Execution quality",
    "Risk utilization",
    "Learning activity",
    "Data quality",
}


def generator() -> Any:
    spec = importlib.util.spec_from_file_location("dashboards", ROOT / "scripts" / "dashboards.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboards"] = mod  # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


def panel_sql() -> list[tuple[str, str, str]]:
    """(dashboard, panel, sql) for every panel, from the committed JSON."""
    out = []
    for f in sorted(JSON_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        out += [(d["title"], p["title"], t["rawSql"]) for p in d["panels"] for t in p["targets"]]
    return out


def test_committed_json_is_what_the_generator_writes() -> None:
    gen = generator()
    expected = {f"{d['uid']}.json": gen.render(d) for d in gen.DASHBOARDS}
    actual = {f.name: f.read_text() for f in JSON_DIR.glob("*.json")}
    assert actual == expected, "run `uv run python scripts/dashboards.py` and commit the JSON"


def test_every_spec_dashboard_exists_on_the_provisioned_datasource() -> None:
    boards = [json.loads(f.read_text()) for f in JSON_DIR.glob("*.json")]
    assert {b["title"] for b in boards} == SPEC_DASHBOARDS
    [ds] = yaml.safe_load((ROOT / "grafana/provisioning/datasources/timescale.yml").read_text())[
        "datasources"
    ]
    for b in boards:
        assert len({p["id"] for p in b["panels"]}) == len(b["panels"])
        for p in b["panels"]:
            assert p["datasource"]["uid"] == ds["uid"] and p["datasource"]["type"] == ds["type"]
    assert ds["user"] == "autotrader_readonly"  # never the owner role
    assert all(sql.strip() for _, _, sql in panel_sql())
