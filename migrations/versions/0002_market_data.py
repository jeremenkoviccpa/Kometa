"""Market data storage (spec section 7). Every table carries tenant_id (section 21).

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE instruments (
            tenant_id   text NOT NULL DEFAULT 'default' REFERENCES tenants,
            symbol      text NOT NULL,
            version     int  NOT NULL,
            spec        jsonb NOT NULL,
            config_hash text NOT NULL,
            loaded_at   timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, symbol, version)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE bars_m1 (
            tenant_id text NOT NULL DEFAULT 'default',
            symbol    text NOT NULL,
            open_time timestamptz NOT NULL,
            bid_o double precision NOT NULL, bid_h double precision NOT NULL,
            bid_l double precision NOT NULL, bid_c double precision NOT NULL,
            ask_o double precision NOT NULL, ask_h double precision NOT NULL,
            ask_l double precision NOT NULL, ask_c double precision NOT NULL,
            volume double precision NOT NULL,
            source text NOT NULL,
            PRIMARY KEY (tenant_id, symbol, open_time)
        )
        """
    )
    op.execute("SELECT create_hypertable('bars_m1', 'open_time', chunk_time_interval => interval '30 days')")
    op.execute(
        "ALTER TABLE bars_m1 SET (timescaledb.compress, "
        "timescaledb.compress_segmentby = 'tenant_id, symbol', timescaledb.compress_orderby = 'open_time')"
    )
    op.execute("SELECT add_compression_policy('bars_m1', interval '60 days')")
    op.execute(
        """
        CREATE TABLE ticks (
            tenant_id text NOT NULL DEFAULT 'default',
            symbol text NOT NULL,
            t timestamptz NOT NULL,
            bid double precision NOT NULL,
            ask double precision NOT NULL
        )
        """
    )
    op.execute("SELECT create_hypertable('ticks', 't', chunk_time_interval => interval '1 day')")
    op.execute(
        "ALTER TABLE ticks SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id, symbol')"
    )
    op.execute("SELECT add_compression_policy('ticks', interval '7 days')")
    op.execute(
        """
        CREATE TABLE spread_stats (
            tenant_id text NOT NULL DEFAULT 'default',
            symbol text NOT NULL,
            hour_of_week smallint NOT NULL CHECK (hour_of_week BETWEEN 0 AND 167),
            median_spread double precision NOT NULL,
            p90_spread double precision NOT NULL,
            data_version text NOT NULL,
            computed_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, symbol, hour_of_week, data_version)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE swap_history (
            tenant_id text NOT NULL DEFAULT 'default',
            symbol text NOT NULL,
            day date NOT NULL,
            swap_long numeric NOT NULL,
            swap_short numeric NOT NULL,
            swap_mode text NOT NULL,
            PRIMARY KEY (tenant_id, symbol, day)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE calendar_events (
            tenant_id text NOT NULL DEFAULT 'default',
            event_time timestamptz NOT NULL,
            currency char(3) NOT NULL,
            impact text NOT NULL CHECK (impact IN ('low', 'medium', 'high')),
            name text NOT NULL,
            source text NOT NULL,
            PRIMARY KEY (tenant_id, event_time, currency, name)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE data_quality_issues (
            id bigserial PRIMARY KEY,
            tenant_id text NOT NULL DEFAULT 'default',
            symbol text NOT NULL,
            start_time timestamptz NOT NULL,
            end_time timestamptz NOT NULL,
            issue_type text NOT NULL,
            severity text NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
            detail text NOT NULL DEFAULT '',
            detected_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ON data_quality_issues (tenant_id, symbol, start_time)")


def downgrade() -> None:
    for t in (
        "data_quality_issues",
        "calendar_events",
        "swap_history",
        "spread_stats",
        "ticks",
        "bars_m1",
        "instruments",
    ):
        op.execute(f"DROP TABLE {t}")
