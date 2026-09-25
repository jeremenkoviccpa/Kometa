"""Shadow mode (spec sections 8, 10, 19): backtest/shadow parity on one recorded quote stream,
and the ShadowBroker's fill rules."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import polars as pl
import pytest

from autotrader.core.broker import Quote
from autotrader.core.events import BarClosed
from autotrader.core.models import Bar, CloseRequest, ModifyStopRequest, Signal, Timeframe, Trade
from autotrader.core.timeutil import utc
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.backtest import run_backtest
from autotrader.engine.costs import InstrumentCosts, StaticRates
from autotrader.engine.live import LiveRunner
from autotrader.engine.live_bars import LiveBarBuilder
from autotrader.engine.requests import StrategyError
from autotrader.engine.shadow import ShadowBroker, ShadowSession
from autotrader.strategies_api import Request, Strategy, StrategyContext, StrategyManifest
from autotrader.strategies_api.loader import LoadedStrategy, load_strategy
from autotrader.validation.inputs import prepare

ROOT = Path(__file__).resolve().parents[2]
INSTR = {"SYNTH": synthetic_instrument()}
COSTS = {"SYNTH": InstrumentCosts.from_instrument(INSTR["SYNTH"])}


@pytest.fixture(scope="module")
def demo() -> LoadedStrategy:
    return load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")


@pytest.fixture(scope="module")
def recorded() -> list[Quote]:
    """A recorded quote stream: one quote per minute at second 30, 40 trading days."""
    df = generate(SyntheticSpec(days=40, seed=3))
    return [
        Quote(
            symbol="SYNTH",
            bid=Decimal(str(r["bid_c"])),
            ask=Decimal(str(r["ask_c"])),
            time=r["open_time"] + timedelta(seconds=30),
        )
        for r in df.select("open_time", "bid_c", "ask_c").iter_rows(named=True)
    ]


def key(s: Signal) -> tuple[object, ...]:
    return (s.signal_id, s.created_at, s.symbol, s.side, s.entry_type, s.stop_price, s.target_price)


def test_backtest_and_shadow_signals_are_identical(demo: LoadedStrategy, recorded: list[Quote]) -> None:
    # shadow: quotes -> live bars -> strategy -> ShadowBroker
    trades: list[Trade] = []
    session = ShadowSession(demo.cls, COSTS, on_trade=trades.append)
    for q in recorded:
        session.on_quote(q)
    session.on_time(recorded[-1].time + timedelta(minutes=2))

    # backtest: the same quotes recorded as M1 bars, historical aggregation, SimBroker
    m1: list[Bar] = []
    builder = LiveBarBuilder([("SYNTH", Timeframe.M1)], on_m1=m1.append)
    for q in recorded:
        builder.on_quote(q)
    builder.on_time(recorded[-1].time + timedelta(minutes=2))
    frame = pl.DataFrame([b.model_dump(exclude={"symbol", "timeframe", "close_time"}) for b in m1])
    inp = prepare({"SYNTH": frame}, demo.manifest, INSTR, synthetic=True)
    bt = run_backtest(demo.cls, inp.m1, inp.series, inp.instruments, inp.cost_model)

    assert len(bt.signals) >= 5, "too few signals for a meaningful parity check"
    assert [key(s) for s in session.signals] == [key(s) for s in bt.signals]
    # shadow trades carry R and are marked as not money
    assert trades and all(t.account_id == "shadow" and t.money_at_risk > 0 for t in trades)


# ---------------------------------------------------------------- ShadowBroker rules

T0 = utc(2026, 1, 7, 12)


def broker() -> ShadowBroker:
    return ShadowBroker(COSTS, StaticRates())


def q(bid: str, ask: str, sec: int) -> Quote:
    return Quote(symbol="SYNTH", bid=Decimal(bid), ask=Decimal(ask), time=T0 + timedelta(seconds=sec))


def sig(side: str = "buy", entry_type: str = "market", entry: float | None = None, stop: float = 99.0,
        target: float | None = 102.0, expiry: int | None = None) -> Signal:  # fmt: skip
    return Signal(
        signal_id=uuid4(),
        strategy_id="s",
        strategy_version="1",
        symbol="SYNTH",
        side=side,
        entry_type=entry_type,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        expiry_bars=expiry,
        created_at=T0,
        reason="t",
    )


def test_market_entry_fills_at_next_quote_then_stop_at_quote() -> None:
    b = broker()
    b.on_quote(q("100.00", "100.02", 0))
    b.submit(sig(), Timeframe.H1)
    [f] = b.on_quote(q("100.10", "100.12", 1))
    assert f.kind == "entry" and f.price == pytest.approx(100.12)
    assert b.on_quote(q("99.50", "99.52", 2)) == []
    [x] = b.on_quote(q("98.90", "98.92", 3))  # gapped through 99.00: fills at the quote, not the stop
    assert x.exit_reason == "stop" and x.price == pytest.approx(98.90)
    [t] = b.trades
    assert t.r_multiple < -1.0


def test_target_fills_at_target_and_same_quote_cannot_exit() -> None:
    b = broker()
    b.on_quote(q("100.00", "100.02", 0))
    b.submit(sig(), Timeframe.H1)
    b.on_quote(q("100.00", "100.02", 1))
    [x] = b.on_quote(q("102.50", "102.52", 2))
    assert x.exit_reason == "target" and x.price == pytest.approx(102.0)


def test_limit_needs_one_tick_through_and_expires() -> None:
    b = broker()
    b.on_quote(q("100.00", "100.02", 0))
    b.submit(sig(entry_type="limit", entry=99.50, stop=98.0), Timeframe.M1)
    assert b.on_quote(q("99.48", "99.50", 1)) == []  # touched, not through
    [f] = b.on_quote(q("99.47", "99.49", 2))
    assert f.price == pytest.approx(99.50)
    b.submit(sig(entry_type="limit", entry=90.0, stop=89.0, expiry=1), Timeframe.M1)
    b.on_quote(q("99.47", "99.49", 61))
    assert not b.pending  # expired after one M1 bar


def test_stop_too_close_is_rejected() -> None:
    b = broker()
    b.on_quote(q("100.00", "100.10", 0))
    b.submit(sig(stop=100.00), Timeframe.H1)  # 0.10 away with a 0.10 spread
    assert not b.pending and b.rejections


def test_close_request_and_tighten_only() -> None:
    b = broker()
    b.on_quote(q("100.00", "100.02", 0))
    s = sig()
    b.submit(s, Timeframe.H1)
    b.on_quote(q("100.00", "100.02", 1))
    pid = str(s.signal_id)
    loosen = ModifyStopRequest(
        strategy_id="s", strategy_version="1", position_id=pid, new_stop=98.0, reason=""
    )
    b.handle(loosen, Timeframe.H1)
    assert b.positions[pid].stop == 99.0  # loosening ignored
    b.handle(CloseRequest(strategy_id="s", strategy_version="1", position_id=pid, reason="x"), Timeframe.H1)
    [x] = b.on_quote(q("100.30", "100.32", 2))
    assert x.exit_reason == "close_request" and x.price == pytest.approx(100.30)


# ---------------------------------------------------------------- live runner contract


class _EveryBar(Strategy):
    """Signals on every H1 bar and does NOT guard its own warmup: the runner must."""

    manifest: ClassVar[StrategyManifest] = StrategyManifest.model_validate(
        {
            "id": "every_bar",
            "version": "1.0.0",
            "origin": "owner",
            "family": "test",
            "symbols": ["SYNTH"],
            "timeframes": ["H1"],
            "expected": {"trades_per_month": 1, "win_rate": 0.5, "avg_r": 0},
            "demo_only": True,
        }
    )

    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        return {("SYNTH", Timeframe.H1): 3}

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        return [ctx.signal("SYNTH", "buy", event.bar.bid_c - 1.0)]


def h1(i: int) -> BarClosed:
    o = T0 + timedelta(hours=i)
    bar = Bar(symbol="SYNTH", timeframe=Timeframe.H1, open_time=o, close_time=o + timedelta(hours=1),
              bid_o=100, bid_h=101, bid_l=99, bid_c=100, ask_o=100.1, ask_h=101.1, ask_l=99.1, ask_c=100.1,
              volume=1)  # fmt: skip
    return BarClosed(at=bar.close_time, symbol="SYNTH", timeframe=Timeframe.H1, bar=bar)


def runner(cls: type[Strategy]) -> LiveRunner:
    return LiveRunner(
        cls, spread_fn=lambda _s, _t: 0.1, positions_fn=lambda _a, _b: [], pending_fn=lambda _a, _b: []
    )


def test_runner_respects_warmup_and_sets_now_to_bar_close() -> None:
    r = runner(_EveryBar)
    out = [x for i in range(5) for x in r.on_bars([h1(i)])]
    assert len(out) == 3  # bars 3, 4, 5 of 5
    assert [s.created_at for s in out if isinstance(s, Signal)] == [
        T0 + timedelta(hours=h) for h in (3, 4, 5)
    ]


class _Twice(_EveryBar):
    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        s = ctx.signal("SYNTH", "buy", event.bar.bid_c - 1.0)
        return [s, s]


def test_runner_fails_closed_on_contract_violation() -> None:
    r = runner(_Twice)
    for i in range(2):
        r.on_bars([h1(i)])  # still warming up (needs 3 bars): not called yet
    with pytest.raises(StrategyError, match="duplicate"):
        r.on_bars([h1(2)])
