"""The trade journal (spec 14.2): features see no future, every signal is followed to its outcome by the
triple barrier (pessimistic when stop and target share a bar), and the pipeline's verdicts and real trades
land on the right entry. Every "not yet"/"refused" case has a control that resolves."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from autotrader.core.bus import TRADES, InMemoryBus
from autotrader.core.clock import SimClock
from autotrader.core.events import BarClosed, OrderIntentCreated, PositionClosed, RiskDecided, SignalEmitted
from autotrader.core.models import Bar, OrderIntent, RiskDecision, Signal, Timeframe, Trade
from autotrader.data.instruments import load_instruments
from autotrader.data.synthetic import SyntheticSpec, generate
from autotrader.learning.features import snapshot
from autotrader.learning.journal import JournalEntry, JournalService, follow
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.inputs import prepare
from autotrader.validation.poisoning import poison_after

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)


# ---------------------------------------------------------------- features


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return generate(
        SyntheticSpec(symbol="XAUUSD", days=120, seed=5, start_price=2000.0, pip_size=0.01, annual_vol=0.16)
    )


def _series(df: pl.DataFrame) -> dict[Timeframe, object]:
    smc = load_strategy(ROOT / "strategies" / "library" / "smc_sniper").manifest
    inst, _ = load_instruments(ROOT / "config" / "instruments.yaml")
    inp = prepare({"XAUUSD": df}, smc, {"XAUUSD": inst["XAUUSD"]}, synthetic=True)
    return {tf: b for (_, tf), b in inp.series.items() if tf != Timeframe.M1}


def test_features_see_no_future(frame: pl.DataFrame) -> None:
    cut = frame["open_time"][0] + timedelta(days=100, hours=13)
    clean = snapshot(_series(frame), cut, spread=0.3)  # type: ignore[arg-type]
    poisoned = snapshot(_series(poison_after(frame, cut, seed=3)), cut, spread=0.3)  # type: ignore[arg-type]
    assert clean == poisoned
    assert None not in (clean.atr_h1, clean.slope_h1, clean.slope_h4, clean.slope_d1, clean.realized_vol_d)
    later = snapshot(_series(frame), cut + timedelta(days=3), spread=0.3)  # type: ignore[arg-type]
    assert later != clean  # control: the features do move with the data they are allowed to see


def test_features_say_unknown_rather_than_zero_without_history() -> None:
    f = snapshot({}, T0, spread=0.3)
    assert f.atr_h1 is None and f.slope_d1 is None and f.level_dist_atr is None
    assert f.hour == 10 and f.weekday == 0 and "london" in f.session


# ---------------------------------------------------------------- the triple barrier


def entry(side: str = "buy", target: float | None = 2030.0, tf: Timeframe = Timeframe.H1) -> JournalEntry:
    buy = side == "buy"
    return JournalEntry(
        signal_id="s",
        strategy_id="x",
        strategy_version="1.0.0",
        symbol="XAUUSD",
        side=side,
        timeframe=tf,
        created_at=T0,
        entry=2000.0,
        stop=1990.0 if buy else 2010.0,
        target=target if buy else (1970.0 if target else None),
        shadow=False,
        strategy_win_rate_20=None,
        features=snapshot({}, T0, spread=0.3),
    )


def bar(minute: int, lo: float, hi: float, close: float | None = None, spread: float = 0.2) -> Bar:
    c = close if close is not None else (lo + hi) / 2
    t = T0 + timedelta(minutes=minute)
    return Bar(
        symbol="XAUUSD",
        timeframe=Timeframe.M1,
        open_time=t,
        close_time=t + timedelta(minutes=1),
        bid_o=c,
        bid_h=hi,
        bid_l=lo,
        bid_c=c,
        ask_o=c + spread,
        ask_h=hi + spread,
        ask_l=lo + spread,
        ask_c=c + spread,
        volume=1.0,
    )


def test_target_first_is_a_win_with_its_excursions() -> None:
    e = follow(entry(), bar(0, 1995.0, 2005.0))
    assert e.outcome is None and e.mae_r == pytest.approx(0.5) and e.mfe_r == pytest.approx(0.5)
    e = follow(e, bar(1, 2004.0, 2031.0))
    assert e.outcome is not None and e.outcome.label == 1 and e.outcome.r == pytest.approx(3.0)
    assert e.outcome.mfe_r == pytest.approx(3.1) and e.outcome.minutes == 2


def test_stop_and_target_in_one_bar_count_as_the_stop() -> None:
    e = follow(entry(), bar(0, 1989.0, 2031.0))
    assert e.outcome is not None and e.outcome.label == -1 and e.outcome.r == -1.0


def test_a_short_is_followed_on_the_ask() -> None:
    e = follow(entry("sell"), bar(0, 2009.0, 2009.9, spread=0.2))  # ask high 2010.1: stopped
    assert e.outcome is not None and e.outcome.label == -1
    e = follow(entry("sell"), bar(0, 1969.0, 1999.0, spread=0.2))  # control: ask low 1969.2 hits 1970
    assert e.outcome is not None and e.outcome.label == 1 and e.outcome.r == pytest.approx(3.0)


def test_untouched_signals_time_out_at_the_horizon_marked_to_market() -> None:
    e = follow(entry(target=None, tf=Timeframe.M5), bar(60, 1998.0, 2003.0, close=2002.0))
    assert e.outcome is None
    e = follow(e, bar(24 * 60, 2001.0, 2006.0, close=2005.0))  # a day later
    assert e.outcome is not None and e.outcome.label == 0 and e.outcome.r == pytest.approx(0.5)


def test_bars_before_the_signal_do_not_count() -> None:
    assert follow(entry(), bar(-5, 1980.0, 2040.0)).outcome is None


# ---------------------------------------------------------------- the service


def sig(sid: str = "a") -> Signal:
    return Signal(
        signal_id=uuid.uuid5(uuid.NAMESPACE_URL, sid),
        strategy_id="swing",
        strategy_version="1.0.0",
        symbol="XAUUSD",
        side="buy",
        entry_type="limit",
        entry_price=2000.0,
        stop_price=1990.0,
        target_price=2030.0,
        created_at=T0,
        reason="",
    )


async def test_every_signal_gets_its_verdict_outcome_and_real_trade(tmp_path: Path) -> None:
    j = JournalService(SimClock(T0), ["XAUUSD"], path=tmp_path / "journal.jsonl")
    s, other = sig("a"), sig("b")
    await j._on_signal(SignalEmitted(at=T0, signal=s, timeframe=Timeframe.H1))
    await j._on_signal(SignalEmitted(at=T0, signal=other, timeframe=Timeframe.H1, shadow=True))
    intent = OrderIntent(
        intent_id=uuid.uuid4(), signal=s, proposed_lots=Decimal("0.1"), risk_fraction=0.001, account_id="a"
    )
    await j._on_intent(OrderIntentCreated(at=T0, intent=intent))
    decision = RiskDecision(
        intent_id=intent.intent_id,
        verdict="reject",
        approved_lots=Decimal(0),
        reasons=("size below min lot (never rounded up)",),
        limits_snapshot_hash="h",
        decided_at=T0,
        expires_at=T0,
        sequence=1,
    )
    await j._on_decision(RiskDecided(at=T0, decision=decision))
    e = j.entries[str(s.signal_id)]
    assert e.status == "rejected" and "min lot" in e.reasons[0]
    assert j.entries[str(other.signal_id)].shadow and j.entries[str(other.signal_id)].status == "signalled"
    j.on_bars([_closed(bar(3, 2002.0, 2031.0))])  # the rejected signal would have won: that is learned too
    assert j.entries[str(s.signal_id)].outcome is not None
    assert j.entries[str(s.signal_id)].outcome.label == 1  # type: ignore[union-attr]
    # a reload sees the same journal
    again = JournalService(SimClock(T0), ["XAUUSD"], path=tmp_path / "journal.jsonl")
    assert again.entries == j.entries and not again.open
    v = again.view()
    assert (
        v["signals"] == 2
        and v["by_strategy"][0]["target_first"] == 2
        and v["by_strategy"][0]["rejected"] == 1
    )


async def test_a_real_trade_lands_on_the_signal_that_became_it(tmp_path: Path) -> None:
    j = JournalService(SimClock(T0), ["XAUUSD"])
    s = sig("a")
    await j._on_signal(SignalEmitted(at=T0, signal=s, timeframe=Timeframe.H1))
    trade = Trade(
        trade_id="t",
        account_id="a",
        strategy_id="swing",
        strategy_version="1.0.0",
        symbol="XAUUSD",
        side="buy",
        lots=Decimal("0.1"),
        entry_time=T0 + timedelta(minutes=2),
        entry_price=Decimal(2000),
        stop_price=Decimal(1990),
        exit_time=T0 + timedelta(hours=1),
        exit_price=Decimal(2030),
        pnl_gross=Decimal(300),
        costs=Decimal(5),
        pnl_net=Decimal(295),
        money_at_risk=Decimal(100),
        r_multiple=2.95,
        mae=0.2,
        mfe=3.0,
    )
    await j._on_trade(PositionClosed(at=trade.exit_time, trade=trade))
    assert j.entries[str(s.signal_id)].trade_r is None  # not approved by the gate: not this signal's trade
    j.entries[str(s.signal_id)] = j.entries[str(s.signal_id)].model_copy(update={"status": "approved"})
    await j._on_trade(PositionClosed(at=trade.exit_time, trade=trade))  # control
    assert j.entries[str(s.signal_id)].trade_r == 2.95


def _closed(b: Bar) -> BarClosed:
    return BarClosed(at=b.close_time, symbol=b.symbol, timeframe=b.timeframe, bar=b)


async def test_a_resolved_shadow_signal_becomes_a_shadow_trade_and_a_paper_signal_does_not(
    tmp_path: Path,
) -> None:
    bus = InMemoryBus()
    j = JournalService(SimClock(T0), ["XAUUSD"], bus=bus)
    await j._on_signal(SignalEmitted(at=T0, signal=sig("paper"), timeframe=Timeframe.H1))
    await j._on_signal(SignalEmitted(at=T0, signal=sig("shadow"), timeframe=Timeframe.H1, shadow=True))
    j.on_bars([_closed(bar(3, 2002.0, 2031.0))])  # both reach the 2030 target
    await j.flush()
    [(_, msg)] = await bus.read(TRADES, "t", "t-1")
    t = msg.trade
    assert t.account_id == "shadow" and t.strategy_id == "swing" and t.lots == 0
    assert t.r_multiple == pytest.approx(3.0) and t.exit_price == Decimal("2030.0")
    assert t.trade_id == f"shadow-{sig('shadow').signal_id}"  # deterministic: one per signal, ever
