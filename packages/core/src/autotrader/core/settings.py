"""Environment settings (spec section 17). Every variable is documented in config/settings.example.env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AT_", env_file=".env", env_file_encoding="utf-8", secrets_dir=None, extra="ignore"
    )

    env: Literal["dev", "ci", "paper", "live"] = "dev"
    tenant_id: str = "default"
    account_id: str = "default"
    config_dir: Path = Path("config")
    ledger_path: Path = Path("var/ledger.jsonl")  # dev/CI trial registry; Postgres in production
    registry_path: Path = Path("var/registry.jsonl")  # strategy versions and stage history
    reports_dir: Path = Path("reports")
    execution_journal_path: Path = Path("var/execution_journal.json")
    execution_quality_path: Path = Path("var/execution_quality.jsonl")  # until Postgres runs

    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://autotrader:autotrader@localhost:5432/autotrader"
    )
    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    broker_login: SecretStr | None = None
    broker_password: SecretStr | None = None
    broker_server: str | None = None
    bridge_url: str | None = None
    bridge_token: SecretStr | None = None
    oanda_token: SecretStr | None = None  # OANDA v20 API token (practice or live), from .env only
    oanda_account: str | None = None  # e.g. 101-004-1234567-001
    oanda_environment: Literal["practice", "live"] = "practice"
    ctrader_client_id: str | None = None  # cTrader Open API application (openapi.ctrader.com)
    ctrader_client_secret: SecretStr | None = None
    ctrader_access_token: SecretStr | None = None  # from the application's Playground, ~30 days
    ctrader_account_id: int | None = None  # ctidTraderAccountId (`at ctrader accounts` lists them)
    ctrader_environment: Literal["demo", "live"] = "demo"

    # the owner's Claude tracks (packages/ai): paper only, off until switched on in the hub
    ai_model: str = "claude-sonnet-5"
    ai_max_calls_per_day: int = 200
    ai_free_every_s: float = 900.0

    owner_public_key_path: Path = Path("config/owner_ed25519.pub")
    risk_decision_key_path: Path | None = None  # risk-gate container only
    risk_decision_public_key_path: Path | None = None

    max_clock_skew_seconds: float = Field(default=2.0, gt=0)

    api_owner_token: SecretStr | None = None  # control endpoints are disabled without it
    audit_path: Path = Path("var/audit.jsonl")  # dev/CI; Postgres audit_log in production
    learning_freeze_path: Path = Path("var/learning.freeze")  # present = all learning loops paused
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    alert_email_from: str | None = None
    alert_email_to: str | None = None

    def secret_values(self) -> list[str]:
        """All configured secret strings, for the log redaction filter."""
        out: list[str] = []
        for name in type(self).model_fields:
            v = getattr(self, name)
            if isinstance(v, SecretStr):
                s = v.get_secret_value()
                if s:
                    out.append(s)
        return out
