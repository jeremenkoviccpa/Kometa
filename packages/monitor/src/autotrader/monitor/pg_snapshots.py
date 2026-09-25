"""Writes the monitor's account snapshots to `account_snapshots` (migration 0006) for the dashboards."""

from __future__ import annotations

import psycopg

from autotrader.monitor.pg_ledger import sync_url
from autotrader.monitor.state import AccountSnapshot


class PgSnapshots:
    def __init__(self, url: str) -> None:
        self.url = sync_url(url)

    def __call__(self, s: AccountSnapshot) -> None:
        with psycopg.connect(self.url) as conn:
            conn.execute(
                "INSERT INTO account_snapshots (tenant_id, account_id, at, currency, balance, equity,"
                " peak_equity, open_risk, positions) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (tenant_id, account_id, at) DO NOTHING",
                (
                    s.tenant_id,
                    s.account_id,
                    s.at,
                    s.currency,
                    s.balance,
                    s.equity,
                    s.peak_equity,
                    s.open_risk,
                    s.positions,
                ),
            )
