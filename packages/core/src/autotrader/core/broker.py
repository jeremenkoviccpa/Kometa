"""Broker-facing models shared by execution (client) and mt5_bridge (server), spec section 12."""

from __future__ import annotations

import zlib
from decimal import Decimal
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from autotrader.core.models import Frozen, Side, UtcDatetime

OrderType = Literal["market", "limit", "stop"]
TradeMode = Literal["demo", "contest", "real"]
MarginMode = Literal["hedging", "netting", "exchange"]

# Every order the system sends carries a comment "at" + 29 hex chars (fits MT5's 31-char comment).
CLIENT_ORDER_PREFIX = "at"


def magic_number(strategy_id: str, version: str) -> int:
    """Stable per strategy version, never 0 (0 is what manual trades carry)."""
    return (zlib.crc32(f"{strategy_id}|{version}".encode()) & 0x7FFFFFFF) or 1


INTENT_NAMESPACE = uuid5(NAMESPACE_URL, "autotrader/intent")


def intent_id_for(signal_id: UUID) -> UUID:
    """One intent per signal, ever: allocator retries and restarts cannot create a second order."""
    return uuid5(INTENT_NAMESPACE, str(signal_id))


def client_order_id(intent_id: UUID) -> str:
    """Derived from the approved intent, so a retried or re-decided intent can never place twice."""
    return CLIENT_ORDER_PREFIX + intent_id.hex[:29]


def is_system_comment(comment: str) -> bool:
    return len(comment) == 31 and comment.startswith(CLIENT_ORDER_PREFIX)


class AccountInfo(Frozen):
    account_id: str
    currency: str
    balance: Decimal
    equity: Decimal
    margin: Decimal
    free_margin: Decimal
    server_time: UtcDatetime
    trade_mode: TradeMode  # paper must run on demo, live on real; checked at startup
    margin_mode: MarginMode  # the order manager requires hedging (one position per order)


class SymbolInfo(Frozen):
    symbol: str
    digits: int
    point: Decimal
    contract_size: Decimal
    min_lot: Decimal
    lot_step: Decimal
    max_lot: Decimal
    margin_per_lot: Decimal | None = None
    trade_allowed: bool = True


class Quote(Frozen):
    symbol: str
    bid: Decimal
    ask: Decimal
    time: UtcDatetime


class PlaceRequest(Frozen):
    client_order_id: str = Field(min_length=1, max_length=31)  # fits the MT5 comment field
    symbol: str
    side: Side
    order_type: OrderType
    lots: Decimal = Field(gt=0)
    price: Decimal | None = None
    sl: Decimal  # always attached at placement
    tp: Decimal | None = None
    magic: int
    expires_at: UtcDatetime | None = None


class ModifyRequest(Frozen):
    position_id: str | None = None
    broker_order_id: str | None = None
    sl: Decimal | None = None
    tp: Decimal | None = None


class BrokerAck(Frozen):
    ok: bool
    broker_order_id: str | None = None
    position_id: str | None = None
    filled_price: Decimal | None = None
    filled_lots: Decimal | None = None
    error: str | None = None
    latency_ms: float = 0.0


class BrokerPosition(Frozen):
    position_id: str
    symbol: str
    side: Side
    lots: Decimal
    price_open: Decimal
    sl: Decimal | None
    tp: Decimal | None
    magic: int
    comment: str
    opened_at: UtcDatetime
    profit: Decimal = Decimal(0)


class BrokerOrder(Frozen):
    broker_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    lots: Decimal
    price: Decimal
    sl: Decimal | None
    tp: Decimal | None
    magic: int
    comment: str
    created_at: UtcDatetime
    expires_at: UtcDatetime | None = None


class BrokerDeal(Frozen):
    """A broker deal. kind="balance" is a deposit, withdrawal or credit: only `profit` and `time` mean
    anything (position_id and symbol are empty, lots is 0)."""

    deal_id: str
    kind: Literal["trade", "balance"] = "trade"
    position_id: str
    symbol: str
    side: Side
    entry: Literal["in", "out"]
    lots: Decimal
    price: Decimal
    commission: Decimal
    swap: Decimal
    profit: Decimal
    time: UtcDatetime
    magic: int
    comment: str


class ExecutionQuality(Frozen):
    """One row of `execution_quality` (spec section 12): requested vs filled, spread and latency.

    slippage is in price units, positive = adverse to us.
    """

    client_order_id: str
    account_id: str
    tenant_id: str = "default"
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Side
    order_type: OrderType
    lots: Decimal
    requested_price: Decimal | None
    filled_price: Decimal
    spread_at_fill: Decimal | None
    slippage: Decimal | None
    latency_ms: float
    filled_at: UtcDatetime
