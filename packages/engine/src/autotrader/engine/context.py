"""StrategyContext implementation shared by every mode."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, MutableMapping
from types import MappingProxyType
from typing import Any

import numpy as np

from autotrader.core.hashing import hash_obj
from autotrader.core.models import CancelRequest, CloseRequest, EntryType, ModifyStopRequest, Side, Signal
from autotrader.engine.market import EngineMarketView
from autotrader.strategies_api.base import PendingView, PositionView
from autotrader.strategies_api.manifest import ParamValue

SIGNAL_NAMESPACE = uuid.UUID("5b0c1e7a-2f64-4d7e-9a53-6f1f0e3c9a11")


def signal_uuid(strategy_id: str, version: str, now_ns: int, seq: int) -> uuid.UUID:
    """Deterministic: same strategy, bar and call order give the same id in every run and mode."""
    return uuid.uuid5(SIGNAL_NAMESPACE, f"{strategy_id}|{version}|{now_ns}|{seq}")


class EngineContext:
    def __init__(
        self,
        strategy_id: str,
        version: str,
        market: EngineMarketView,
        params: Mapping[str, ParamValue],
        positions_fn: Callable[[str, str | None], list[PositionView]],
        pending_fn: Callable[[str, str | None], list[PendingView]],
        seed: int = 0,
        state: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.version = version
        self._market = market
        self._params = MappingProxyType(dict(params))
        self._state: MutableMapping[str, Any] = state if state is not None else {}
        self._positions_fn = positions_fn
        self._pending_fn = pending_fn
        rng_seed = int(hash_obj({"id": strategy_id, "v": version, "p": dict(params), "s": seed})[:16], 16)
        self._rng = np.random.default_rng(rng_seed)
        self._seq_ns = -1
        self._seq = 0

    @property
    def market(self) -> EngineMarketView:
        return self._market

    @property
    def params(self) -> Mapping[str, ParamValue]:
        return self._params

    @property
    def state(self) -> MutableMapping[str, Any]:
        return self._state

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def my_positions(self, symbol: str | None = None) -> list[PositionView]:
        return self._positions_fn(self.strategy_id, symbol)

    def my_pending(self, symbol: str | None = None) -> list[PendingView]:
        return self._pending_fn(self.strategy_id, symbol)

    def _next_id(self) -> uuid.UUID:
        now = self._market.now_ns
        if now != self._seq_ns:
            self._seq_ns, self._seq = now, 0
        self._seq += 1
        return signal_uuid(self.strategy_id, self.version, now, self._seq)

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
        return Signal(
            signal_id=self._next_id(),
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            symbol=symbol,
            side=side,
            entry_type=entry_type,
            entry_price=entry_price,
            stop_price=float(stop_price),
            target_price=None if target_price is None else float(target_price),
            expiry_bars=expiry_bars,
            created_at=self._market.now,
            reason=reason,
            tags=dict(tags or {}),
        )

    def cancel(self, signal_id: str, reason: str = "") -> CancelRequest:
        return CancelRequest(
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            signal_id=uuid.UUID(signal_id),
            reason=reason,
        )

    def close(self, position_id: str, reason: str = "") -> CloseRequest:
        return CloseRequest(
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            position_id=position_id,
            reason=reason,
        )

    def modify_stop(self, position_id: str, new_stop: float, reason: str = "") -> ModifyStopRequest:
        return ModifyStopRequest(
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            position_id=position_id,
            new_stop=float(new_stop),
            reason=reason,
        )
