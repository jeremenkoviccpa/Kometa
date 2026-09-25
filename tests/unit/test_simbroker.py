"""Fill simulator rules on hand-built M1 bars (spec section 8)."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from autotrader.core.models import Signal
from autotrader.core.series import NS_PER_MINUTE, BarsArray, from_ns
from autotrader.engine.costs import CostModel, InstrumentCosts, SpreadModel, StaticRates
from autotrader.engine.simbroker import SimBroker

T0 = 1_767_607_200_000_000_000  # 2026-01-05 10:00 UTC, a Monday
SPREAD = 0.02
SLIP_MULT = 0.5  # slippage = 0.01
INST = InstrumentCosts("X", "USD", 1000.0, 0.001, 0.01, 0.01, 100.0, 1.0, -1.0, -1.0, "points", 2)


def bars(ohlc: list[tuple[float, float, float, float]]) -> BarsArray:
    n = len(ohlc)
    a = np.asarray(ohlc, dtype=np.float64)
    ot = T0 + np.arange(n, dtype=np.int64) * NS_PER_MINUTE
    return BarsArray(
        open_time=ot,
        close_time=ot + NS_PER_MINUTE,
        bid_o=a[:, 0],
        bid_h=a[:, 1],
        bid_l=a[:, 2],
        bid_c=a[:, 3],
        ask_o=a[:, 0] + SPREAD,
        ask_h=a[:, 1] + SPREAD,
        ask_l=a[:, 2] + SPREAD,
        ask_c=a[:, 3] + SPREAD,
        volume=np.ones(n),
    )


def broker(b: BarsArray) -> SimBroker:
    cm = CostModel(SpreadModel({"X": np.full(168, SPREAD)}), slippage_mult=SLIP_MULT)
    return SimBroker.build({"X": b}, {"X": INST}, cm, StaticRates(), "USD", 10_000.0)


def sig(
    side: str,
    stop: float,
    *,
    entry_type: str = "market",
    entry: float | None = None,
    target: float | None = None,
    expiry: int | None = None,
) -> Signal:
    return Signal(
        signal_id=uuid4(),
        strategy_id="s",
        strategy_version="1.0.0",
        symbol="X",
        side=side,
        entry_type=entry_type,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        expiry_bars=expiry,
        created_at=from_ns(T0),
        reason="t",
    )


def test_market_buy_fills_next_open_at_ask_plus_slippage_then_target() -> None:
    b = bars([(100, 100.1, 99.9, 100), (100.5, 101.2, 100.4, 101), (101, 103, 101, 102.5)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 99.0, target=102.0), 1.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 3)
    (t,) = br.trades
    assert t.entry_price == pytest.approx(100.5 + SPREAD + 0.01)
    assert t.exit_reason == "target"
    assert t.exit_price == 102.0
    assert t.pnl_gross == pytest.approx((102.0 - 100.53) * 1000)
    assert t.commission == pytest.approx(2.0)  # 1.0 per side per lot
    assert t.r_multiple == pytest.approx(t.pnl_net / ((100.53 - 99.0) * 1000))


def test_stop_and_target_in_same_bar_assumes_stop() -> None:
    b = bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 105, 95, 100)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 97.0, target=103.0), 1.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 3)
    (t,) = br.trades
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(97.0 - 0.01)


def test_stop_gap_fills_at_open() -> None:
    b = bars([(100, 100, 100, 100), (100, 100.2, 99.9, 100), (96, 96.5, 95.5, 96)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 98.0), 1.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 3)
    assert br.trades[0].exit_price == pytest.approx(96.0 - 0.01)


def test_limit_needs_trade_through_by_one_tick() -> None:
    # buy limit 99.50 on ask; ask_l = bid_l + 0.02
    b = bars([(100, 100, 100, 100), (100, 100, 99.48, 99.9), (99.9, 100, 99.477, 99.9)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 99.0, entry_type="limit", entry=99.5), 1.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 2)
    assert not br.positions.get("X")  # ask_l 99.50 touches but does not trade through
    br.advance("X", 2, 3)
    (p,) = br.positions["X"]
    assert p.entry_price == 99.5  # ask_l 99.497 <= 99.499


def test_pending_fill_bar_can_stop_but_not_take_profit() -> None:
    b = bars([(100, 100, 100, 100), (100, 101.5, 99.6, 101)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 99.0, entry_type="stop", entry=100.5, target=101.0), 1.0, T0, 1)
    br.advance("X", 1, 2)
    assert br.trades == []  # target inside the fill bar is not allowed
    (p,) = br.positions["X"]
    assert p.entry_price == pytest.approx(100.5 + 0.01)


def test_stop_entry_gap_fills_at_open() -> None:
    b = bars([(100, 100, 100, 100), (101, 101.2, 100.9, 101)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 99.0, entry_type="stop", entry=100.5), 1.0, T0, 1)
    br.advance("X", 1, 2)
    assert br.positions["X"][0].entry_price == pytest.approx(101 + SPREAD + 0.01)


def test_expiry_cancels_before_trigger() -> None:
    b = bars([(100, 100, 100, 100)] * 3 + [(100, 101, 100, 100.8)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 99.0, entry_type="stop", entry=100.5, expiry=2), 1.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 4)
    assert not br.positions.get("X")
    assert not br.pending.get("X")


def test_modify_stop_only_tightens_and_close_request() -> None:
    b = bars([(100, 100, 100, 100), (100, 100.5, 99.9, 100.4), (100.4, 100.6, 100.3, 100.5)])
    br = broker(b)
    br.last_idx["X"] = 0
    s = sig("buy", 99.0)
    br.submit_entry(s, 1.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 2)
    pid = str(s.signal_id)
    assert not br.modify_stop("s", pid, 98.0, T0)  # loosening
    assert not br.modify_stop("s", pid, 100.5, T0)  # beyond current bid
    assert br.modify_stop("s", pid, 99.5, T0)
    assert br.request_close("s", pid, T0, "close_request")
    br.advance("X", 2, 3)
    (t,) = br.trades
    assert t.exit_reason == "close_request"
    assert t.exit_price == pytest.approx(100.4 - 0.01)
    assert t.stop_price == 99.0  # R is measured against the initial stop


def test_swaps_triple_on_wednesday() -> None:
    b = bars([(100, 100, 100, 100)] * 3)
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 99.0), 2.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 2)
    before = br.balance
    br.charge_swaps(weekday=1)
    br.charge_swaps(weekday=2)
    # -1 point * 0.001 * 1000 units * 2 lots = -2 per night; Tue 1 night + Wed 3 nights
    assert br.balance - before == pytest.approx(-8.0)


def test_entry_beyond_stop_is_not_opened() -> None:
    b = bars([(100, 100, 100, 100), (98, 98, 98, 98)])
    br = broker(b)
    br.last_idx["X"] = 0
    br.submit_entry(sig("buy", 99.0), 1.0, T0 + NS_PER_MINUTE, 1)
    br.advance("X", 1, 2)
    assert not br.positions.get("X")
    assert br.rejections[-1].reason == "fill price beyond stop"
