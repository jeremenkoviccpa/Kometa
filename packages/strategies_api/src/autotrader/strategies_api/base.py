"""Strategy plugin interface (spec section 6).

A strategy receives closed bars and returns requests. It has no access to
money, the broker, the database, the network, the file system or other
strategies: everything it can see comes through `StrategyContext`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal, Protocol

import numpy as np

from autotrader.core.events import BarClosed
from autotrader.core.models import (
    CancelRequest,
    CloseRequest,
    EntryType,
    ModifyStopRequest,
    Side,
    Signal,
    Timeframe,
)
from autotrader.core.series import BarsArray
from autotrader.strategies_api.manifest import ParamValue, StrategyManifest

Request = Signal | CancelRequest | CloseRequest | ModifyStopRequest


@dataclass(frozen=True)
class PositionView:
    position_id: str
    signal_id: str
    symbol: str
    side: Side
    lots: float
    entry_price: float
    stop_price: float
    target_price: float | None
    opened_at: datetime


@dataclass(frozen=True)
class PendingView:
    signal_id: str
    symbol: str
    side: Side
    entry_type: EntryType
    entry_price: float | None
    stop_price: float
    target_price: float | None
    created_at: datetime


@dataclass(frozen=True)
class FillView:
    signal_id: str
    position_id: str
    symbol: str
    side: Side
    kind: Literal["entry", "exit"]
    price: float
    lots: float
    time: datetime
    exit_reason: str | None = None  # "stop", "target", "close_request", "demotion", ...


class MarketView(Protocol):
    @property
    def now(self) -> datetime:
        """close_time of the bar being processed."""
        ...

    def bars(self, symbol: str, tf: Timeframe, n: int) -> BarsArray:
        """The last n CLOSED bars (fewer if not available), oldest first. Always a copy."""
        ...

    def spread(self, symbol: str) -> float:
        """Current (live) or modelled (backtest) spread in price units."""
        ...


class StrategyContext(Protocol):
    @property
    def market(self) -> MarketView: ...

    @property
    def params(self) -> Mapping[str, ParamValue]: ...

    @property
    def state(self) -> MutableMapping[str, Any]:
        """Persisted per strategy version; must stay JSON serializable."""
        ...

    @property
    def rng(self) -> np.random.Generator:
        """The only allowed source of randomness; seeded per strategy version and run."""
        ...

    def my_positions(self, symbol: str | None = None) -> list[PositionView]: ...

    def my_pending(self, symbol: str | None = None) -> list[PendingView]: ...

    def signal(
        self,
        symbol: str,
        side: Side,
        stop_price: float,
        *,
        entry_type: EntryType = "market",
        entry_price: float | None = None,
        target_price: float | None = None,
        expiry_bars: int | None = None,
        reason: str = "",
        tags: Mapping[str, str] | None = None,
    ) -> Signal:
        """Build a Signal with a deterministic id, stamped with this strategy and `market.now`."""
        ...

    def cancel(self, signal_id: str, reason: str = "") -> CancelRequest: ...

    def close(self, position_id: str, reason: str = "") -> CloseRequest: ...

    def modify_stop(self, position_id: str, new_stop: float, reason: str = "") -> ModifyStopRequest: ...


class Strategy(ABC):
    manifest: ClassVar[StrategyManifest]

    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        """Bars needed per series before on_bar is called. Default: none."""
        return {}

    @abstractmethod
    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        """Called once per closed bar of every subscribed series."""

    def on_fill(self, ctx: StrategyContext, fill: FillView) -> list[Request]:
        return []
