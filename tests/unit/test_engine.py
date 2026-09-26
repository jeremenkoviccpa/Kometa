from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import pytest

from autotrader.core.events import BarClosed
from autotrader.core.indicators import EventIndex
from autotrader.core.models import CalendarEvent, Timeframe
from autotrader.core.series import from_ns, to_ns
from autotrader.data.convert import to_bars_array
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.backtest import BacktestResult, run_backtest
from autotrader.engine.market import EngineMarketView, SeriesBuffer, news_from_index, news_from_times
from autotrader.engine.requests import StrategyError
from autotrader.strategies_api import Request, Strategy, StrategyContext, StrategyManifest
from autotrader.strategies_api.loader import LoadedStrategy, load_strategy
from autotrader.validation.inputs import EngineInputs, prepare
from autotrader.validation.poisoning import future_poisoning_test

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "demo_ma_cross.json"
INSTR = {"SYNTH": synthetic_instrument()}


@pytest.fixture(scope="module")
def demo() -> LoadedStrategy:
    return load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return generate(SyntheticSpec(days=180, seed=11))


@pytest.fixture(scope="module")
def inputs(demo: LoadedStrategy, frame: pl.DataFrame) -> EngineInputs:
    return prepare({"SYNTH": frame}, demo.manifest, INSTR, synthetic=True)


def _run(cls: type[Strategy], inp: EngineInputs) -> BacktestResult:
    return run_backtest(cls, inp.m1, inp.series, inp.instruments, inp.cost_model)


def test_demo_runs_and_is_deterministic(demo: LoadedStrategy, inputs: EngineInputs) -> None:
    a = _run(demo.cls, inputs)
    b = _run(demo.cls, inputs)
    assert a.metrics.trades > 20
    assert a.result_hash() == b.result_hash()
    for t in a.trades:
        assert t.entry_time_ns < t.exit_time_ns or t.exit_reason == "stop"
        assert t.money_at_risk > 0
        assert t.lots > 0
    assert all(s.created_at.minute == 0 for s in a.signals)  # H1 closes only


def test_golden_backtest(demo: LoadedStrategy, inputs: EngineInputs) -> None:
    r = _run(demo.cls, inputs)
    got = {
        "code_hash": demo.code_hash,
        "data_version": inputs.data_versions["SYNTH"],
        "result_hash": r.result_hash(),
        "trades": r.metrics.trades,
        "total_r": round(r.metrics.total_r, 6),
    }
    if os.environ.get("AT_UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(got, indent=2) + "\n")
        pytest.skip("golden file written; rerun to compare")
    expected = json.loads(GOLDEN.read_text())
    assert got == expected, "engine output changed; if intended, rerun with AT_UPDATE_GOLDEN=1 and log why"


def test_demo_passes_future_poisoning(demo: LoadedStrategy, frame: pl.DataFrame) -> None:
    cut = frame["open_time"][frame.height // 2]
    rep = future_poisoning_test(demo.cls, {"SYNTH": frame}, INSTR, cut)
    assert rep.passed, rep.first_difference
    assert rep.signals_checked > 5


class _Cheater(Strategy):
    """Reaches behind the MarketView into the next, not yet closed, bar. Must be caught."""

    manifest: ClassVar[StrategyManifest] = StrategyManifest.model_validate(
        {
            "id": "cheater",
            "version": "1.0.0",
            "origin": "owner",
            "family": "cheater",
            "symbols": ["SYNTH"],
            "timeframes": ["H1"],
            "expected": {"trades_per_month": 1, "win_rate": 0.5, "avg_r": 0},
            "demo_only": True,
        }
    )

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        buf = ctx.market.series("SYNTH", Timeframe.H1)  # type: ignore[attr-defined]
        if buf.visible >= buf.size:
            return []
        future = float(buf._cols["bid_c"][buf.visible])
        now = event.bar.bid_c
        if future > now:
            return [ctx.signal("SYNTH", "buy", now - 1.0, reason="peek")]
        return []


def test_poisoning_catches_lookahead(frame: pl.DataFrame) -> None:
    cut = frame["open_time"][frame.height // 2]
    rep = future_poisoning_test(_Cheater, {"SYNTH": frame}, INSTR, cut)
    assert not rep.passed


class _Broken(Strategy):
    manifest = _Cheater.manifest

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        s = ctx.signal("SYNTH", "buy", event.bar.bid_c - 1.0)
        return [s, s]  # duplicate id


def test_strategy_contract_violation_fails_closed(inputs: EngineInputs) -> None:
    with pytest.raises(StrategyError, match="duplicate"):
        _run(_Broken, inputs)


def test_warmup_is_respected(demo: LoadedStrategy, inputs: EngineInputs) -> None:
    r = _run(demo.cls, inputs)
    slow = int(demo.manifest.params["slow"].value)
    first_h1_close = int(inputs.series[("SYNTH", Timeframe.H1)].close_time[slow + 1])
    assert all(s.created_at >= from_ns(first_h1_close) for s in r.signals)


def test_series_buffer_never_exposes_future() -> None:
    ba = generate(SyntheticSpec(days=3, seed=1)).head(10)
    buf = SeriesBuffer(to_bars_array(ba))
    buf.advance_to(4)
    got = buf.last(100)
    assert len(got) == 4
    got.bid_c[:] = -1
    assert buf.last(1).bid_c[0] != -1  # copy, not a view
    with pytest.raises(IndexError):
        buf.row(4)
    with pytest.raises(ValueError, match="forward"):
        buf.advance_to(3)
    mv = EngineMarketView(lambda s, t: 0.1)
    mv.set_now(100)
    with pytest.raises(ValueError, match="backwards"):
        mv.set_now(99)


def test_live_append_matches_backtest_view() -> None:
    ba = to_bars_array(generate(SyntheticSpec(days=3, seed=1)).head(50))
    pre, live = SeriesBuffer(ba), SeriesBuffer(capacity=4)
    for i in range(len(ba)):
        pre.advance_to(i + 1)
        live.append({k: getattr(ba, k)[i].item() for k in ba.__dataclass_fields__})
        a, b = pre.last(20), live.last(20)
        for k in ba.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))


def test_strategies_see_scheduled_news_from_backtest_times_and_the_live_calendar() -> None:
    t0 = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    nfp = t0 + timedelta(minutes=30)
    fn = news_from_times({"XAUUSD": np.array([to_ns(t0 - timedelta(hours=2)), to_ns(nfp)], dtype=np.int64)})
    assert fn("XAUUSD", to_ns(t0)) == (30.0, 120.0)
    assert fn("XAUUSD", to_ns(nfp)) == (0.0, 150.0)  # an event exactly now is the next one, 0 minutes away
    assert fn("EURUSD", to_ns(t0)) == (None, None)  # nothing known is None, never 0
    idx = EventIndex([CalendarEvent(time=nfp, currency="USD", impact="high", name="Non-Farm Payrolls")])
    live = news_from_index(lambda: idx, {"XAUUSD": ("XAU", "USD")})
    assert live("XAUUSD", to_ns(t0)) == (30.0, None)
    assert news_from_index(lambda: None, {"XAUUSD": ("XAU", "USD")})("XAUUSD", to_ns(t0)) == (None, None)
    mv = EngineMarketView(lambda s, t: 0.1)
    assert mv.minutes_to_news("XAUUSD") == (None, None)  # no calendar wired: unknown
