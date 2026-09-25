"""Audit log: keep the exact canonical text each hash was computed over.

`payload` (jsonb) is for queries and dashboards; jsonb normalizes numbers and key order, so the chain
is verified over `canonical`, and verification also checks the queryable columns agree with it.
The table is empty until the monitor writes to Postgres (Phase 8), so NOT NULL needs no backfill.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE audit_log ADD COLUMN canonical text NOT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE audit_log DROP COLUMN canonical")
