"""Execution records (spec section 12): orders, fills, execution quality, reconciliation reports.

Every table carries tenant_id and account_id (sections 5 and 21). Column names match the Pydantic
models in core (tests/unit/test_execution_schema.py checks it), so rows are written with model_dump.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE orders (
            tenant_id         text NOT NULL DEFAULT 'default',
            account_id        text NOT NULL,
            client_order_id   text NOT NULL,
            broker_order_id   text,
            strategy_id       text NOT NULL,
            strategy_version  text NOT NULL,
            symbol            text NOT NULL,
            side              text NOT NULL CHECK (side IN ('buy', 'sell')),
            entry_type        text NOT NULL,
            status            text NOT NULL,
            lots              numeric NOT NULL,
            price             numeric,
            sl                numeric NOT NULL,
            tp                numeric,
            created_at        timestamptz NOT NULL,
            updated_at        timestamptz NOT NULL,
            PRIMARY KEY (tenant_id, account_id, client_order_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE fills (
            tenant_id        text NOT NULL DEFAULT 'default',
            account_id       text NOT NULL,
            client_order_id  text NOT NULL,
            symbol           text NOT NULL,
            side             text NOT NULL,
            price            numeric NOT NULL,
            lots             numeric NOT NULL,
            commission       numeric NOT NULL,
            spread_at_fill   numeric NOT NULL,
            requested_price  numeric,
            latency_ms       double precision NOT NULL,
            filled_at        timestamptz NOT NULL
        )
        """
    )
    op.execute("SELECT create_hypertable('fills', 'filled_at', chunk_time_interval => interval '90 days')")
    op.execute(
        """
        CREATE TABLE execution_quality (
            tenant_id         text NOT NULL DEFAULT 'default',
            account_id        text NOT NULL,
            client_order_id   text NOT NULL,
            strategy_id       text NOT NULL,
            strategy_version  text NOT NULL,
            symbol            text NOT NULL,
            side              text NOT NULL,
            order_type        text NOT NULL,
            lots              numeric NOT NULL,
            requested_price   numeric,
            filled_price      numeric NOT NULL,
            spread_at_fill    numeric,
            slippage          numeric,
            latency_ms        double precision NOT NULL,
            filled_at         timestamptz NOT NULL
        )
        """
    )
    op.execute(
        "SELECT create_hypertable('execution_quality', 'filled_at', "
        "chunk_time_interval => interval '90 days')"
    )
    op.execute(
        """
        CREATE TABLE recon_reports (
            tenant_id   text NOT NULL DEFAULT 'default',
            account_id  text NOT NULL,
            at          timestamptz NOT NULL,
            ok          boolean NOT NULL,
            mismatches  jsonb NOT NULL,
            external    jsonb NOT NULL
        )
        """
    )


def downgrade() -> None:
    for t in ("recon_reports", "execution_quality", "fills", "orders"):
        op.execute(f"DROP TABLE IF EXISTS {t}")
