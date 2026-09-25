"""Known secret values never appear in log output (spec section 17)."""

from __future__ import annotations

import io
import logging

import pytest

from autotrader.core.logging import REDACTED, RedactingFilter
from autotrader.core.settings import Settings

SECRETS = {
    "AT_ANTHROPIC_API_KEY": "sk-ant-api03-THISISNOTAREALKEY1234567890",
    "AT_TELEGRAM_BOT_TOKEN": "123456789:AAAbbbCCCdddEEEfffGGGhhhIIIjjjKKK",
    "AT_BROKER_PASSWORD": "hunter2-broker-pw",
    "AT_BRIDGE_TOKEN": "bridge-token-abcdef",
    "AT_DATABASE_URL": "postgresql+asyncpg://autotrader:db-pass-xyz@db:5432/autotrader",
}


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> tuple[logging.Logger, io.StringIO]:
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    settings = Settings(_env_file=None)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter(settings.secret_values()))
    logger = logging.getLogger("leaktest")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_configured_secrets_are_redacted(capture: tuple[logging.Logger, io.StringIO]) -> None:
    logger, stream = capture
    for v in SECRETS.values():
        logger.info("value %s", v)
        logger.warning(f"inline {v}")
    try:
        raise RuntimeError(f"boom with {SECRETS['AT_BROKER_PASSWORD']}")
    except RuntimeError:
        logger.exception("failed")
    out = stream.getvalue()
    for secret in (
        "THISISNOTAREALKEY",
        "AAAbbbCCC",
        "hunter2-broker-pw",
        "bridge-token-abcdef",
        "db-pass-xyz",
    ):
        assert secret not in out
    assert REDACTED in out


def test_unknown_secret_shapes_are_redacted() -> None:
    f = RedactingFilter()
    assert "sk-ant-zzzzzzzzzzzzzzzz" not in f.redact("key sk-ant-zzzzzzzzzzzzzzzz here")
    assert "s3cr3t" not in f.redact("password=s3cr3t")
    assert "pw123" not in f.redact("postgresql://user:pw123@host/db")


def test_settings_repr_hides_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AT_BROKER_PASSWORD", "hunter2-broker-pw")
    s = Settings(_env_file=None)
    assert "hunter2" not in repr(s) and "hunter2" not in str(s.model_dump())
