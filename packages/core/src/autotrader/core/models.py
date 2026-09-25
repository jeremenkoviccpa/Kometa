"""Domain models shared by every package (spec section 5).

All models are frozen. Times are timezone-aware UTC. Prices are Decimal at the
broker/DB boundary and float inside numeric hot loops.
"""

from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from autotrader.core.timeutil import ensure_utc

DEFAULT_TENANT = "default"

UtcDatetime = Annotated[datetime, AfterValidator(ensure_utc)]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Timeframe(StrEnum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def minutes(self) -> int:
        return _TF_MINUTES[self]


_TF_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}


class Stage(StrEnum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    MICRO = "micro"
    LIVE = "live"
    SCALED = "scaled"
    RETIRED = "retired"
    DEMO_ONLY = "demo_only"


class HaltState(StrEnum):
    NORMAL = "NORMAL"
    DAILY_HALT = "DAILY_HALT"
    WEEKLY_HALT = "WEEKLY_HALT"
    RECON_HALT = "RECON_HALT"
    FULL_HALT = "FULL_HALT"


HALT_CLOSES_POSITIONS = frozenset({HaltState.DAILY_HALT, HaltState.WEEKLY_HALT, HaltState.FULL_HALT})


Side = Literal["buy", "sell"]
EntryType = Literal["market", "limit", "stop"]


class SessionWindow(Frozen):
    """A daily trading window in a named timezone. `weekdays` uses Monday=0."""

    name: str
    tz: str
    start: time
    end: time
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)


class Instrument(Frozen):
    symbol: str
    asset_class: Literal["fx", "metal", "index", "crypto"]
    base: str
    quote: str
    contract_size: Decimal
    pip_size: Decimal
    tick_size: Decimal
    min_lot: Decimal
    lot_step: Decimal
    max_lot: Decimal
    commission_per_lot: Decimal  # round turn, account currency
    swap_long: Decimal
    swap_short: Decimal
    swap_mode: Literal["points", "money", "percent"]
    triple_swap_weekday: int = Field(ge=0, le=6)
    trading_sessions: tuple[SessionWindow, ...] = ()
    daily_close: str = "17:00 America/New_York"

    @model_validator(mode="after")
    def _check_lots(self) -> Instrument:
        if not (Decimal(0) < self.min_lot <= self.max_lot):
            raise ValueError("require 0 < min_lot <= max_lot")
        if self.lot_step <= 0 or self.tick_size <= 0 or self.contract_size <= 0:
            raise ValueError("lot_step, tick_size and contract_size must be positive")
        return self


class Bar(Frozen):
    symbol: str
    timeframe: Timeframe
    open_time: UtcDatetime
    close_time: UtcDatetime
    bid_o: float
    bid_h: float
    bid_l: float
    bid_c: float
    ask_o: float
    ask_h: float
    ask_l: float
    ask_c: float
    volume: float

    @model_validator(mode="after")
    def _check(self) -> Bar:
        if self.close_time <= self.open_time:
            raise ValueError("close_time must be after open_time")
        return self


class Signal(Frozen):
    signal_id: UUID
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Side
    entry_type: EntryType
    entry_price: float | None
    stop_price: float
    target_price: float | None
    expiry_bars: int | None = Field(default=None, ge=1)
    created_at: UtcDatetime
    reason: str
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Signal:
        if self.entry_type != "market" and self.entry_price is None:
            raise ValueError("entry_price is required for limit and stop entries")
        if self.entry_price is not None:
            wrong_side = (self.side == "buy" and self.stop_price >= self.entry_price) or (
                self.side == "sell" and self.stop_price <= self.entry_price
            )
            if wrong_side:
                raise ValueError("stop_price is on the wrong side of entry_price")
        return self


class CancelRequest(Frozen):
    strategy_id: str
    strategy_version: str
    signal_id: UUID
    reason: str


class OrderIntent(Frozen):
    intent_id: UUID
    signal: Signal
    proposed_lots: Decimal = Field(ge=0)
    risk_fraction: float = Field(ge=0)
    account_id: str
    tenant_id: str = DEFAULT_TENANT


class RiskDecision(Frozen):
    intent_id: UUID
    verdict: Literal["approve", "resize", "reject"]
    approved_lots: Decimal = Field(ge=0)
    reasons: tuple[str, ...]
    limits_snapshot_hash: str
    decided_at: UtcDatetime
    expires_at: UtcDatetime
    sequence: int = Field(ge=0)
    signature: str | None = None  # Ed25519 over the canonical payload, hex


class OrderStatus(StrEnum):
    NEW = "new"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(Frozen):
    client_order_id: str
    broker_order_id: str | None
    account_id: str
    tenant_id: str = DEFAULT_TENANT
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Side
    entry_type: EntryType
    status: OrderStatus
    lots: Decimal
    price: Decimal | None
    sl: Decimal
    tp: Decimal | None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class Fill(Frozen):
    client_order_id: str
    account_id: str
    tenant_id: str = DEFAULT_TENANT
    symbol: str
    side: Side
    price: Decimal
    lots: Decimal
    commission: Decimal
    spread_at_fill: Decimal
    requested_price: Decimal | None
    latency_ms: float
    filled_at: UtcDatetime


class Position(Frozen):
    position_id: str
    account_id: str
    tenant_id: str = DEFAULT_TENANT
    symbol: str
    side: Side
    lots: Decimal
    avg_price: Decimal
    sl: Decimal | None
    tp: Decimal | None
    opened_at: UtcDatetime
    strategy_id: str | None  # None for positions opened outside the system
    strategy_version: str | None
    swap_accrued: Decimal = Decimal(0)
    external: bool = False


class Trade(Frozen):
    """A closed round trip. `r_multiple` = net P&L / money at risk at entry."""

    trade_id: str
    account_id: str
    tenant_id: str = DEFAULT_TENANT
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Side
    lots: Decimal
    entry_time: UtcDatetime
    entry_price: Decimal
    stop_price: Decimal
    exit_time: UtcDatetime
    exit_price: Decimal
    pnl_gross: Decimal
    costs: Decimal
    pnl_net: Decimal
    money_at_risk: Decimal = Field(gt=0)
    r_multiple: float
    mae: float  # in R
    mfe: float  # in R


class CalendarEvent(Frozen):
    time: UtcDatetime
    currency: str = Field(min_length=3, max_length=3)
    impact: Literal["low", "medium", "high"]
    name: str


class CloseRequest(Frozen):
    """Strategy asks to close one of its positions at market. Always risk-reducing."""

    strategy_id: str
    strategy_version: str
    position_id: str
    reason: str


class ModifyStopRequest(Frozen):
    """Strategy moves a stop. Only tightening (toward price) is accepted; loosening is rejected."""

    strategy_id: str
    strategy_version: str
    position_id: str
    new_stop: float
    reason: str


class HaltCommand(Frozen):
    """Risk gate -> execution: what to do on entering a halt (spec section 11, "Halt states").

    Every halt cancels pending entries; daily, weekly and full halts also close open positions.
    """

    state: HaltState
    reason: str
    at: UtcDatetime
    cancel_pending_entries: bool = True
    close_positions: bool

    @classmethod
    def for_state(cls, state: HaltState, reason: str, at: datetime) -> HaltCommand:
        return cls(state=state, reason=reason, at=at, close_positions=state in HALT_CLOSES_POSITIONS)
