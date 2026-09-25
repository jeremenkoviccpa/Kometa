"""Engine events (spec section 8). One schema for in-process and Redis transport."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import Field

from autotrader.core.alerts import Alert
from autotrader.core.broker import Quote
from autotrader.core.models import (
    DEFAULT_TENANT,
    Bar,
    CancelRequest,
    CloseRequest,
    Fill,
    Frozen,
    HaltState,
    ModifyStopRequest,
    Order,
    OrderIntent,
    RiskDecision,
    Side,
    Signal,
    Stage,
    Timeframe,
    Trade,
    UtcDatetime,
)


class Event(Frozen):
    at: UtcDatetime


class BarClosed(Event):
    kind: Literal["BarClosed"] = "BarClosed"
    symbol: str
    timeframe: Timeframe
    bar: Bar


class QuoteUpdate(Event):
    kind: Literal["QuoteUpdate"] = "QuoteUpdate"
    symbol: str
    bid: float
    ask: float


class SignalEmitted(Event):
    kind: Literal["SignalEmitted"] = "SignalEmitted"
    signal: Signal
    timeframe: Timeframe = Timeframe.H1  # the strategy's own timeframe (pending order expiry)
    shadow: bool = False  # shadow signals are recorded for the lifecycle, never sized


class OrderIntentCreated(Event):
    kind: Literal["OrderIntentCreated"] = "OrderIntentCreated"
    intent: OrderIntent
    timeframe: Timeframe = Timeframe.H1


class RiskDecided(Event):
    kind: Literal["RiskDecided"] = "RiskDecided"
    decision: RiskDecision


class OrderPlaced(Event):
    kind: Literal["OrderPlaced"] = "OrderPlaced"
    order: Order


class OrderModified(Event):
    kind: Literal["OrderModified"] = "OrderModified"
    order: Order


class OrderFilled(Event):
    kind: Literal["OrderFilled"] = "OrderFilled"
    fill: Fill


class OrderCancelled(Event):
    kind: Literal["OrderCancelled"] = "OrderCancelled"
    client_order_id: str
    reason: str


class PositionClosed(Event):
    kind: Literal["PositionClosed"] = "PositionClosed"
    trade: Trade


class StageChanged(Event):
    kind: Literal["StageChanged"] = "StageChanged"
    strategy_id: str
    strategy_version: str
    from_stage: Stage
    to_stage: Stage
    reason: str


class HaltEntered(Event):
    kind: Literal["HaltEntered"] = "HaltEntered"
    state: HaltState
    reason: str


class HaltCleared(Event):
    kind: Literal["HaltCleared"] = "HaltCleared"
    previous: HaltState
    actor: str


class ConfigChanged(Event):
    """A config file as a service read it (spec section 17): name, path and sha256 of its bytes."""

    kind: Literal["ConfigChanged"] = "ConfigChanged"
    service: str
    name: str
    path: str
    config_hash: str


class ModelVersionActivated(Event):
    kind: Literal["ModelVersionActivated"] = "ModelVersionActivated"
    model_kind: str
    model_version: str
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- service messages (spec section 18)


class Heartbeat(Event):
    kind: Literal["Heartbeat"] = "Heartbeat"
    service: str


class Exposure(Frozen):
    """One open position or pending entry as the broker reports it."""

    symbol: str
    side: Side
    lots: Decimal
    entry: Decimal
    stop: Decimal | None
    strategy_id: str | None
    strategy_version: str | None
    position_id: str | None = None
    pending: bool = False
    external: bool = False


class AccountUpdate(Event):
    """Execution -> risk gate, allocator, engine: the account as the broker sees it."""

    kind: Literal["AccountUpdate"] = "AccountUpdate"
    account_id: str
    tenant_id: str = DEFAULT_TENANT
    currency: str
    balance: Decimal
    equity: Decimal
    free_margin: Decimal
    exposures: tuple[Exposure, ...]
    quotes: tuple[Quote, ...]
    margin_per_lot: dict[str, Decimal]


class StageSnapshot(Event):
    """Lifecycle -> everyone: every version's stage (sent at start and hourly; StageChanged in between)."""

    kind: Literal["StageSnapshot"] = "StageSnapshot"
    stages: tuple[tuple[str, str, Stage], ...]


class DemotionOrder(Event):
    """Lifecycle -> execution: act on a demotion now."""

    kind: Literal["DemotionOrder"] = "DemotionOrder"
    strategy_id: str
    strategy_version: str
    close_positions: bool
    reason: str


class StrategyRequestEmitted(Event):
    """Engine -> execution: risk-reducing requests from a money-stage strategy."""

    kind: Literal["StrategyRequestEmitted"] = "StrategyRequestEmitted"
    request: CancelRequest | CloseRequest | ModifyStopRequest


class AlertRaised(Event):
    kind: Literal["AlertRaised"] = "AlertRaised"
    alert: Alert


class ResumeRequested(Event):
    """Owner (via the API) -> risk gate: an owner-signed token that clears FULL_HALT."""

    kind: Literal["ResumeRequested"] = "ResumeRequested"
    token: dict[str, str]
    signature: str


BusMessage = Annotated[
    BarClosed
    | QuoteUpdate
    | SignalEmitted
    | OrderIntentCreated
    | RiskDecided
    | OrderPlaced
    | OrderModified
    | OrderFilled
    | OrderCancelled
    | PositionClosed
    | StageChanged
    | HaltEntered
    | HaltCleared
    | ConfigChanged
    | ModelVersionActivated
    | Heartbeat
    | AccountUpdate
    | StageSnapshot
    | DemotionOrder
    | StrategyRequestEmitted
    | AlertRaised
    | ResumeRequested,
    Field(discriminator="kind"),
]
