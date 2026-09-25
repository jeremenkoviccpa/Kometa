"""Monitoring: account snapshots for the dashboards, and the read-only role Grafana logs in as.

`account_snapshots` holds at most one row per account per minute (the monitor down-samples), so the
account overview and risk utilization dashboards can plot equity, drawdown and open risk.

`autotrader_readonly` may only SELECT, now and on tables created later. It is created without a
login; deployment gives it one with its own password (`make db-roles`, secrets/db_readonly_password.txt),
so Grafana never holds the owner's password.

Revision ID: 0006
Revises: 0005
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE account_snapshots (
            tenant_id    text NOT NULL DEFAULT 'default',
            account_id   text NOT NULL,
            at           timestamptz NOT NULL,
            currency     text NOT NULL,
            balance      numeric NOT NULL,
            equity       numeric NOT NULL,
            peak_equity  numeric NOT NULL,
            open_risk    numeric,
            positions    integer NOT NULL,
            PRIMARY KEY (tenant_id, account_id, at)
        )
        """
    )
    op.execute(
        "SELECT create_hypertable('account_snapshots', 'at', chunk_time_interval => interval '90 days')"
    )
    op.execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'autotrader_readonly') THEN
                CREATE ROLE autotrader_readonly NOLOGIN;
            END IF;
        END $$
        """
    )
    op.execute("GRANT USAGE ON SCHEMA public TO autotrader_readonly")
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO autotrader_readonly")
    op.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO autotrader_readonly")
    op.execute("GRANT INSERT, SELECT ON account_snapshots TO autotrader_app")


def downgrade() -> None:
    op.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM autotrader_readonly")
    op.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM autotrader_readonly")
    op.execute("REVOKE USAGE ON SCHEMA public FROM autotrader_readonly")
    op.execute("DROP TABLE account_snapshots")
