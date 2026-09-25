"""Baseline: extensions and the tenant registry.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE tenants (
            tenant_id  text PRIMARY KEY,
            name       text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("INSERT INTO tenants (tenant_id, name) VALUES ('default', 'default')")


def downgrade() -> None:
    op.execute("DROP TABLE tenants")
