"""Broker adapter interface (spec section 12). Adapters are registered by name (section 21)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from decimal import Decimal
from typing import Protocol

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


class BrokerUnavailableError(ConnectionError):
    """The broker or bridge did not answer. The request may or may not have been applied."""


class BrokerAdapter(Protocol):
    async def connect(self) -> None: ...

    async def account(self) -> AccountInfo: ...

    async def symbols(self) -> list[SymbolInfo]: ...

    def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]: ...

    async def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]: ...

    async def place(self, req: PlaceRequest) -> BrokerAck: ...

    async def modify(self, req: ModifyRequest) -> BrokerAck: ...

    async def cancel(self, broker_order_id: str) -> BrokerAck: ...

    async def close_position(self, position_id: str, lots: Decimal | None = None) -> BrokerAck: ...

    async def open_positions(self) -> list[BrokerPosition]: ...

    async def pending_orders(self) -> list[BrokerOrder]: ...

    async def deals(self, since: datetime) -> list[BrokerDeal]: ...


_REGISTRY: dict[str, Callable[..., BrokerAdapter]] = {}


def register_adapter(name: str, factory: Callable[..., BrokerAdapter]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"adapter {name!r} already registered")
    _REGISTRY[name] = factory


def make_adapter(name: str, **kwargs: object) -> BrokerAdapter:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise KeyError(f"unknown broker adapter {name!r}; known: {sorted(_REGISTRY)}") from None
