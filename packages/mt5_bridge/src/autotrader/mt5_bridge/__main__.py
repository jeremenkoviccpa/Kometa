"""Run the bridge on the Windows VPS: `python -m autotrader.mt5_bridge`.

Settings come from the environment (prefix AT_BRIDGE_), never from code. Binds to the tunnel
address only; refuses to start without a server timezone, a long token, or on the wrong account type.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from autotrader.core.logging import configure_logging


class BridgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AT_BRIDGE_", env_file=".env", extra="ignore")

    token: SecretStr
    server_tz: str  # e.g. "NY+7"; never guessed
    expect_trade_mode: Literal["demo", "real"]
    symbols: list[str] = Field(min_length=1)
    host: str = "127.0.0.1"  # set to the Tailscale/WireGuard address
    port: int = 8765
    broker_login: int | None = None
    broker_password: SecretStr | None = None
    broker_server: str | None = None


def main() -> None:  # pragma: no cover - needs Windows and a terminal
    import MetaTrader5  # type: ignore[import-not-found]  # noqa: PLC0415
    import uvicorn  # noqa: PLC0415

    from autotrader.mt5_bridge.app import create_app  # noqa: PLC0415
    from autotrader.mt5_bridge.servertime import ServerTimeZone  # noqa: PLC0415
    from autotrader.mt5_bridge.terminal import MT5Terminal  # noqa: PLC0415

    s = BridgeSettings()
    secrets = [s.token.get_secret_value()]
    if s.broker_password is not None:
        secrets.append(s.broker_password.get_secret_value())
    configure_logging(secrets)
    term = MT5Terminal(
        MetaTrader5, ServerTimeZone(s.server_tz), s.symbols, expect_trade_mode=s.expect_trade_mode
    )
    term.connect(
        s.broker_login, s.broker_password.get_secret_value() if s.broker_password else None, s.broker_server
    )
    uvicorn.run(create_app(term, s.token.get_secret_value()), host=s.host, port=s.port, log_config=None)


if __name__ == "__main__":
    main()
