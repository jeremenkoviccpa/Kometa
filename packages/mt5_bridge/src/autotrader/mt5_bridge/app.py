"""HTTP API of the bridge: the BrokerAdapter methods, one endpoint each.

Every endpoint needs `Authorization: Bearer <token>` (constant-time compare). Transport security is
the private tunnel (WireGuard or Tailscale) or TLS in front; the bridge never listens publicly.
Terminal failures return 503 so the adapter treats them as "broker unavailable" and never as a
rejection. A trade the broker refuses is a normal 200 answer with `ok: false`.
"""

from __future__ import annotations

import hmac
from datetime import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from autotrader.core.broker import (
    AccountInfo,
    BrokerAck,
    BrokerDeal,
    BrokerOrder,
    BrokerPosition,
    ModifyRequest,
    PlaceRequest,
    Quote,
    SymbolInfo,
)
from autotrader.core.models import Bar, Timeframe
from autotrader.mt5_bridge.terminal import MT5Terminal, TerminalError

MIN_TOKEN_LENGTH = 32


class CloseBody(BaseModel):
    lots: Decimal | None = None


def create_app(terminal: MT5Terminal, token: str) -> FastAPI:
    if len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(f"bridge token must be at least {MIN_TOKEN_LENGTH} characters")
    expected = f"Bearer {token}".encode()

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if authorization is None or not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    app = FastAPI(title="autotrader mt5 bridge", dependencies=[Depends(auth)], docs_url=None, redoc_url=None)

    @app.exception_handler(TerminalError)
    async def _terminal_down(_req: Request, exc: TerminalError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # plain `def` endpoints: MetaTrader5 calls block, FastAPI runs them in a thread pool
    @app.get("/account")
    def account() -> AccountInfo:
        return terminal.account()

    @app.get("/symbols")
    def symbols() -> list[SymbolInfo]:
        return terminal.symbols()

    @app.get("/quotes")
    def quotes(symbols: Annotated[str, Query(min_length=1)]) -> list[Quote]:
        return terminal.quotes(symbols.split(","))

    @app.get("/bars")
    def bars(symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]:
        return terminal.history_bars(symbol, tf, start, end)

    @app.post("/orders")
    def place(req: PlaceRequest) -> BrokerAck:
        return terminal.place(req)

    @app.post("/modify")
    def modify(req: ModifyRequest) -> BrokerAck:
        return terminal.modify(req)

    @app.post("/orders/{broker_order_id}/cancel")
    def cancel(broker_order_id: str) -> BrokerAck:
        return terminal.cancel(broker_order_id)

    @app.post("/positions/{position_id}/close")
    def close(position_id: str, body: CloseBody) -> BrokerAck:
        return terminal.close_position(position_id, body.lots)

    @app.get("/positions")
    def positions() -> list[BrokerPosition]:
        return terminal.open_positions()

    @app.get("/orders")
    def orders() -> list[BrokerOrder]:
        return terminal.pending_orders()

    @app.get("/deals")
    def deals(since: datetime) -> list[BrokerDeal]:
        return terminal.deals(since)

    return app
