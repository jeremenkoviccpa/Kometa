"""MT5Adapter: BrokerAdapter over HTTP to `mt5_bridge` on the Windows VPS (spec section 12).

Any transport failure, timeout or 5xx becomes BrokerUnavailableError ("may or may not have been
applied"), which the order manager answers by searching the broker for its client order id before
it retries. A 401 is a configuration error and raises PermissionError: never retried silently.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
from pydantic import BaseModel, TypeAdapter

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
from autotrader.execution.adapter import BrokerUnavailableError, register_adapter

_LISTS: dict[type[Any], TypeAdapter[Any]] = {}


def _list_of[M: BaseModel](model: type[M], data: bytes) -> list[M]:
    ta = _LISTS.setdefault(model, TypeAdapter(list[model]))  # type: ignore[valid-type]
    out: list[M] = ta.validate_json(data)
    return out


class MT5Adapter:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 10.0,
        quote_poll_s: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
        verify: str | bool = True,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=timeout_s,
            transport=transport,
            verify=verify,
        )
        self.quote_poll_s = quote_poll_s

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _req(self, method: str, path: str, **kw: Any) -> bytes:
        try:
            r = await self._client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise BrokerUnavailableError(f"{method} {path}: {type(e).__name__}") from e
        if r.status_code == 401:
            raise PermissionError("bridge rejected the token")
        if r.status_code >= 500:
            raise BrokerUnavailableError(f"{method} {path}: {r.status_code} {r.text[:200]}")
        r.raise_for_status()
        return r.content

    async def connect(self) -> None:
        await self.account()

    async def account(self) -> AccountInfo:
        return AccountInfo.model_validate_json(await self._req("GET", "/account"))

    async def symbols(self) -> list[SymbolInfo]:
        return _list_of(SymbolInfo, await self._req("GET", "/symbols"))

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """Polls the bridge and yields each quote whose time moved."""
        last: dict[str, datetime] = {}
        while True:
            for q in _list_of(
                Quote, await self._req("GET", "/quotes", params={"symbols": ",".join(symbols)})
            ):
                if last.get(q.symbol) != q.time:
                    last[q.symbol] = q.time
                    yield q
            await asyncio.sleep(self.quote_poll_s)

    async def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]:
        params = {"symbol": symbol, "tf": tf.value, "start": start.isoformat(), "end": end.isoformat()}
        return _list_of(Bar, await self._req("GET", "/bars", params=params))

    async def place(self, req: PlaceRequest) -> BrokerAck:
        return BrokerAck.model_validate_json(
            await self._req("POST", "/orders", content=req.model_dump_json())
        )

    async def modify(self, req: ModifyRequest) -> BrokerAck:
        return BrokerAck.model_validate_json(
            await self._req("POST", "/modify", content=req.model_dump_json())
        )

    async def cancel(self, broker_order_id: str) -> BrokerAck:
        return BrokerAck.model_validate_json(await self._req("POST", f"/orders/{broker_order_id}/cancel"))

    async def close_position(self, position_id: str, lots: Decimal | None = None) -> BrokerAck:
        body = {"lots": str(lots) if lots is not None else None}
        return BrokerAck.model_validate_json(
            await self._req("POST", f"/positions/{position_id}/close", json=body)
        )

    async def open_positions(self) -> list[BrokerPosition]:
        return _list_of(BrokerPosition, await self._req("GET", "/positions"))

    async def pending_orders(self) -> list[BrokerOrder]:
        return _list_of(BrokerOrder, await self._req("GET", "/orders"))

    async def deals(self, since: datetime) -> list[BrokerDeal]:
        return _list_of(BrokerDeal, await self._req("GET", "/deals", params={"since": since.isoformat()}))


register_adapter("mt5", MT5Adapter)
