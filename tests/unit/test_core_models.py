from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from autotrader.core.clock import SimClock
from autotrader.core.hashing import canonical_json, hash_obj
from autotrader.core.models import Bar, Signal, Timeframe
from autotrader.core.timeutil import utc


def _signal(**kw: object) -> Signal:
    base: dict[str, object] = {
        "signal_id": uuid4(),
        "strategy_id": "s",
        "strategy_version": "1.0.0",
        "symbol": "EURUSD",
        "side": "buy",
        "entry_type": "limit",
        "entry_price": 1.10,
        "stop_price": 1.09,
        "target_price": None,
        "created_at": utc(2026, 1, 5, 10),
        "reason": "test",
    }
    base.update(kw)
    return Signal.model_validate(base)


def test_signal_requires_entry_price_for_pending() -> None:
    with pytest.raises(ValidationError, match="entry_price is required"):
        _signal(entry_price=None)


def test_signal_stop_wrong_side_rejected() -> None:
    with pytest.raises(ValidationError, match="wrong side"):
        _signal(stop_price=1.11)
    with pytest.raises(ValidationError, match="wrong side"):
        _signal(side="sell", stop_price=1.09)


def test_naive_datetime_rejected_and_converted_to_utc() -> None:
    with pytest.raises(ValidationError, match="naive"):
        _signal(created_at=datetime(2026, 1, 5, 10))  # noqa: DTZ001
    cet = timezone(timedelta(hours=1))
    s = _signal(created_at=datetime(2026, 1, 5, 11, tzinfo=cet))
    assert s.created_at == utc(2026, 1, 5, 10)
    assert s.created_at.utcoffset() == timedelta(0)


def test_models_are_frozen() -> None:
    s = _signal()
    with pytest.raises(ValidationError):
        s.symbol = "GBPUSD"  # type: ignore[misc]


def test_bar_close_after_open() -> None:
    with pytest.raises(ValidationError):
        Bar(
            symbol="X",
            timeframe=Timeframe.M1,
            open_time=utc(2026, 1, 1, 0, 1),
            close_time=utc(2026, 1, 1, 0, 1),
            bid_o=1,
            bid_h=1,
            bid_l=1,
            bid_c=1,
            ask_o=1,
            ask_h=1,
            ask_l=1,
            ask_c=1,
            volume=0,
        )


def test_canonical_json_is_order_independent() -> None:
    a = {"b": Decimal("1.10"), "a": [1, 2], "t": utc(2026, 1, 1)}
    b = {"t": utc(2026, 1, 1), "a": [1, 2], "b": Decimal("1.10")}
    assert canonical_json(a) == canonical_json(b)
    assert hash_obj(a) == hash_obj(b)
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json({"x": float("nan")})


def test_sim_clock_monotonic() -> None:
    c = SimClock(utc(2026, 1, 1))
    c.advance_to(utc(2026, 1, 2))
    with pytest.raises(ValueError, match="backwards"):
        c.advance_to(utc(2026, 1, 1))
