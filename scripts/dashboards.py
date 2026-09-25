"""Grafana dashboards as code (spec section 16). Usage: `uv run python scripts/dashboards.py`.

Writes one JSON file per dashboard to grafana/provisioning/dashboards/json/. Edit this file, not the
JSON: tests/unit/test_dashboards.py fails when the committed JSON differs from what this generates, and
tests/integration/test_audit_pg.py runs every panel's SQL against a migrated database.

Sources: the audit log (every decision-relevant event; `payload` is the event's data), account
snapshots written by the monitor, execution_quality and data_quality_issues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "grafana" / "provisioning" / "dashboards" / "json"
DS = {"type": "grafana-postgresql-datasource", "uid": "timescale"}

# money-stage trades only (shadow trades carry account_id 'shadow')
MONEY_TRADE = "event_type = 'PositionClosed' AND payload->'trade'->>'account_id' <> 'shadow'"
SHADOW_TRADE = "event_type = 'PositionClosed' AND payload->'trade'->>'account_id' = 'shadow'"
VERSION = "(payload->'trade'->>'strategy_id') || ' ' || (payload->'trade'->>'strategy_version')"


@dataclass(frozen=True)
class Panel:
    title: str
    sql: str
    kind: str = "timeseries"  # timeseries | table | stat | barchart
    w: int = 12
    h: int = 8
    unit: str | None = None

    @property
    def fmt(self) -> str:
        return "time_series" if self.kind == "timeseries" else "table"


def dashboard(uid: str, title: str, panels: list[Panel]) -> dict[str, Any]:
    out, x, y, row_h = [], 0, 0, 0
    for i, p in enumerate(panels, start=1):
        if x + p.w > 24:
            x, y, row_h = 0, y + row_h, 0
        panel: dict[str, Any] = {
            "id": i,
            "type": p.kind,
            "title": p.title,
            "gridPos": {"x": x, "y": y, "w": p.w, "h": p.h},
            "datasource": DS,
            "targets": [{"refId": "A", "rawSql": p.sql, "format": p.fmt, "rawQuery": True, "datasource": DS}],
        }
        if p.unit:
            panel["fieldConfig"] = {"defaults": {"unit": p.unit}, "overrides": []}
        out.append(panel)
        x, row_h = x + p.w, max(row_h, p.h)
    return {
        "uid": uid,
        "title": title,
        "schemaVersion": 39,
        "time": {"from": "now-30d", "to": "now"},
        "refresh": "1m",
        "tags": ["autotrader"],
        "panels": out,
    }


DASHBOARDS = [
    dashboard(
        "at-account",
        "Account overview",
        [
            Panel(
                "Equity and balance",
                "SELECT at AS time, account_id || ' equity' AS metric, equity::float AS value"
                " FROM account_snapshots WHERE $__timeFilter(at)"
                " UNION ALL SELECT at, account_id || ' balance', balance::float"
                " FROM account_snapshots WHERE $__timeFilter(at) ORDER BY 1",
            ),
            Panel(
                "Drawdown from peak",
                "SELECT at AS time, account_id AS metric,"
                " ((peak_equity - equity) / NULLIF(peak_equity, 0))::float AS value"
                " FROM account_snapshots WHERE $__timeFilter(at) ORDER BY 1",
                unit="percentunit",
            ),
            Panel(
                "Net P&L per day (money stages)",
                "SELECT time_bucket('1 day', (payload->'trade'->>'exit_time')::timestamptz) AS time,"
                " sum((payload->'trade'->>'pnl_net')::numeric)::float AS value"
                f" FROM audit_log WHERE {MONEY_TRADE} AND $__timeFilter(at) GROUP BY 1 ORDER BY 1",
                kind="barchart",
            ),
            Panel(
                "Halts and critical alerts",
                "SELECT at AS time, event_type, coalesce(payload->>'kind', payload->>'state') AS kind,"
                " coalesce(payload->>'message', payload->>'reason') AS message FROM audit_log"
                " WHERE (event_type = 'HaltEntered'"
                " OR (event_type = 'Alert' AND payload->>'severity' = 'critical'))"
                " AND $__timeFilter(at) ORDER BY at DESC LIMIT 200",
                kind="table",
            ),
            Panel(
                "Config files in use",
                "SELECT DISTINCT ON (payload->>'name') at AS time, payload->>'name' AS file,"
                " payload->>'service' AS service, left(payload->>'config_hash', 16) AS sha256"
                " FROM audit_log WHERE event_type = 'ConfigChanged' ORDER BY payload->>'name', at DESC",
                kind="table",
                w=24,
            ),
        ],
    ),
    dashboard(
        "at-stages",
        "Strategy versions by stage",
        [
            Panel(
                "Versions per stage",
                "SELECT stage AS metric, count(*) AS value FROM (SELECT DISTINCT ON"
                " (payload->>'strategy_id', payload->>'strategy_version') payload->>'to_stage' AS stage"
                " FROM audit_log WHERE event_type = 'StageChanged'"
                " ORDER BY payload->>'strategy_id', payload->>'strategy_version', id DESC) s"
                " GROUP BY stage ORDER BY stage",
                kind="barchart",
            ),
            Panel(
                "Current stage of every version",
                "SELECT DISTINCT ON (payload->>'strategy_id', payload->>'strategy_version')"
                " payload->>'strategy_id' AS strategy, payload->>'strategy_version' AS version,"
                " payload->>'to_stage' AS stage, at AS since, payload->>'reason' AS reason"
                " FROM audit_log WHERE event_type = 'StageChanged'"
                " ORDER BY payload->>'strategy_id', payload->>'strategy_version', id DESC",
                kind="table",
            ),
            Panel(
                "Stage changes",
                "SELECT at AS time, payload->>'strategy_id' AS strategy,"
                " payload->>'strategy_version' AS version, payload->>'from_stage' AS from_stage,"
                " payload->>'to_stage' AS to_stage, payload->>'reason' AS reason FROM audit_log"
                " WHERE event_type = 'StageChanged' AND $__timeFilter(at) ORDER BY at DESC LIMIT 500",
                kind="table",
                w=24,
            ),
        ],
    ),
    dashboard(
        "at-drift",
        "Live vs backtest drift",
        [
            # SPEC-QUESTION: the backtest band lives in the lifecycle registry (a ledger, not Postgres
            # yet); until it moves, drift is shown as money-stage R next to shadow R of the same
            # versions, plus the evaluator's demotion reasons (docs/open_questions.md 30).
            Panel(
                "Mean R per week, money stages",
                f"SELECT time_bucket('7 days', at) AS time, {VERSION} AS metric,"
                " avg((payload->'trade'->>'r_multiple')::float) AS value"
                f" FROM audit_log WHERE {MONEY_TRADE} AND $__timeFilter(at) GROUP BY 1, 2 ORDER BY 1",
            ),
            Panel(
                "Mean R per week, shadow",
                f"SELECT time_bucket('7 days', at) AS time, {VERSION} AS metric,"
                " avg((payload->'trade'->>'r_multiple')::float) AS value"
                f" FROM audit_log WHERE {SHADOW_TRADE} AND $__timeFilter(at) GROUP BY 1, 2 ORDER BY 1",
            ),
            Panel(
                "Demotions and retirements",
                "SELECT at AS time, payload->>'strategy_id' AS strategy,"
                " payload->>'strategy_version' AS version, payload->>'from_stage' AS from_stage,"
                " payload->>'to_stage' AS to_stage, payload->>'reason' AS reason FROM audit_log"
                " WHERE event_type = 'StageChanged'"
                " AND payload->>'to_stage' IN ('shadow', 'micro', 'retired')"
                " AND payload->>'from_stage' IN ('micro', 'live', 'scaled', 'shadow')"
                " AND $__timeFilter(at) ORDER BY at DESC",
                kind="table",
                w=24,
            ),
        ],
    ),
    dashboard(
        "at-execution",
        "Execution quality",
        [
            Panel(
                "Mean adverse slippage per hour (price units)",
                "SELECT time_bucket('1 hour', filled_at) AS time, symbol AS metric,"
                " avg(slippage)::float AS value FROM execution_quality"
                " WHERE slippage IS NOT NULL AND $__timeFilter(filled_at) GROUP BY 1, 2 ORDER BY 1",
            ),
            Panel(
                "Fill latency p50 and p95 (ms)",
                "SELECT time_bucket('1 hour', filled_at) AS time, 'p50' AS metric,"
                " percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS value FROM execution_quality"
                " WHERE $__timeFilter(filled_at) GROUP BY 1"
                " UNION ALL SELECT time_bucket('1 hour', filled_at), 'p95',"
                " percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FROM execution_quality"
                " WHERE $__timeFilter(filled_at) GROUP BY 1 ORDER BY 1",
                unit="ms",
            ),
            Panel(
                "Spread at fill",
                "SELECT filled_at AS time, symbol AS metric, spread_at_fill::float AS value"
                " FROM execution_quality WHERE $__timeFilter(filled_at) ORDER BY 1",
            ),
            Panel(
                "Worst fills",
                "SELECT filled_at AS time, symbol, strategy_id, strategy_version, order_type,"
                " requested_price, filled_price, slippage, latency_ms FROM execution_quality"
                " WHERE $__timeFilter(filled_at) ORDER BY slippage DESC NULLS LAST LIMIT 50",
                kind="table",
            ),
        ],
    ),
    dashboard(
        "at-risk",
        "Risk utilization",
        [
            Panel(
                "Open risk as a share of equity",
                "SELECT at AS time, account_id AS metric, (open_risk / NULLIF(equity, 0))::float AS value"
                " FROM account_snapshots WHERE open_risk IS NOT NULL AND $__timeFilter(at) ORDER BY 1",
                unit="percentunit",
            ),
            Panel(
                "Risk decisions per hour",
                "SELECT time_bucket('1 hour', at) AS time, payload->'decision'->>'verdict' AS metric,"
                " count(*) AS value FROM audit_log WHERE event_type = 'RiskDecided'"
                " AND $__timeFilter(at) GROUP BY 1, 2 ORDER BY 1",
            ),
            Panel(
                "Why the gate rejected or resized",
                "SELECT reason, count(*) AS n FROM audit_log,"
                " jsonb_array_elements_text(payload->'decision'->'reasons') AS reason"
                " WHERE event_type = 'RiskDecided' AND $__timeFilter(at) GROUP BY 1 ORDER BY 2 DESC LIMIT 20",
                kind="table",
            ),
            Panel(
                "Halts",
                "SELECT at AS time, event_type, coalesce(payload->>'state', payload->>'previous') AS state,"
                " coalesce(payload->>'reason', payload->>'actor') AS detail FROM audit_log"
                " WHERE event_type IN ('HaltEntered', 'HaltCleared') AND $__timeFilter(at) ORDER BY at DESC",
                kind="table",
            ),
        ],
    ),
    dashboard(
        "at-learning",
        "Learning activity",
        [
            Panel(
                "Model and cost-model activations",
                "SELECT at AS time, payload->>'model_kind' AS kind, payload->>'model_version' AS version"
                " FROM audit_log WHERE event_type = 'ModelVersionActivated' AND $__timeFilter(at)"
                " ORDER BY at DESC",
                kind="table",
            ),
            Panel(
                "Stage entries per week",
                "SELECT time_bucket('7 days', at) AS time, payload->>'to_stage' AS metric, count(*) AS value"
                " FROM audit_log WHERE event_type = 'StageChanged' AND $__timeFilter(at)"
                " GROUP BY 1, 2 ORDER BY 1",
                kind="barchart",
            ),
            Panel(
                "Owner commands (pause learning, retire, resume)",
                "SELECT at AS time, actor, payload->>'action' AS action, payload::text AS detail"
                " FROM audit_log WHERE event_type = 'OwnerCommand' AND $__timeFilter(at) ORDER BY at DESC",
                kind="table",
                w=24,
            ),
        ],
    ),
    dashboard(
        "at-data-quality",
        "Data quality",
        [
            Panel(
                "Issues per day by type",
                "SELECT time_bucket('1 day', start_time) AS time, issue_type AS metric, count(*) AS value"
                " FROM data_quality_issues WHERE $__timeFilter(start_time) GROUP BY 1, 2 ORDER BY 1",
                kind="barchart",
            ),
            Panel(
                "Issues by symbol and severity",
                "SELECT symbol, severity, count(*) AS n FROM data_quality_issues"
                " WHERE $__timeFilter(start_time) GROUP BY 1, 2 ORDER BY 1, 2",
                kind="table",
            ),
            Panel(
                "High-severity windows (excluded from validation)",
                "SELECT start_time AS time, end_time, symbol, issue_type, detail FROM data_quality_issues"
                " WHERE severity = 'high' AND $__timeFilter(start_time) ORDER BY start_time DESC LIMIT 200",
                kind="table",
                w=24,
            ),
        ],
    ),
]


def render(d: dict[str, Any]) -> str:
    return json.dumps(d, indent=2) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = {f"{d['uid']}.json" for d in DASHBOARDS}
    for old in OUT.glob("*.json"):
        if old.name not in keep:
            old.unlink()
    for d in DASHBOARDS:
        (OUT / f"{d['uid']}.json").write_text(render(d))
    print(f"wrote {len(DASHBOARDS)} dashboards to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
