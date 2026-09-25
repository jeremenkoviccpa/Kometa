"""Audit log (spec section 16): append-only and hash-chained, enforced by the database.

A trigger rejects UPDATE and DELETE; the application role may only INSERT and SELECT. `at audit verify`
walks the chain (hash = sha256(prev_hash + canonical payload)).

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE audit_log (
            id          bigserial PRIMARY KEY,
            tenant_id   text NOT NULL DEFAULT 'default',
            at          timestamptz NOT NULL,
            event_type  text NOT NULL,
            actor       text NOT NULL,
            payload     jsonb NOT NULL,
            prev_hash   char(64) NOT NULL,
            hash        char(64) NOT NULL UNIQUE
        )
        """
    )
    op.execute("CREATE INDEX audit_log_type_at ON audit_log (tenant_id, event_type, at)")
    op.execute(
        """
        CREATE FUNCTION audit_log_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only (% refused)', TG_OP;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER audit_log_no_update_delete BEFORE UPDATE OR DELETE ON audit_log "
        "FOR EACH ROW EXECUTE FUNCTION audit_log_append_only()"
    )
    op.execute(
        "CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log "
        "EXECUTE FUNCTION audit_log_append_only()"
    )
    op.execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'autotrader_app') THEN
                CREATE ROLE autotrader_app NOLOGIN;
            END IF;
        END $$
        """
    )
    op.execute("REVOKE ALL ON audit_log FROM autotrader_app")
    op.execute("GRANT INSERT, SELECT ON audit_log TO autotrader_app")
    op.execute("GRANT USAGE ON SEQUENCE audit_log_id_seq TO autotrader_app")


def downgrade() -> None:
    op.execute("DROP TABLE audit_log")
    op.execute("DROP FUNCTION audit_log_append_only()")
