"""Every strategy in strategies/library: loads through the static checks, sees no future (future poisoning
with signals before the cut, so the check is not vacuous), and is a candidate, never pre-approved."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from autotrader.core.events import BarClosed
from autotrader.core.models import Instrument
from autotrader.core.series import to_ns
from autotrader.data.instruments import load_instruments
from autotrader.data.synthetic import SyntheticSpec, generate
from autotrader.engine.backtest import run_backtest
from autotrader.strategies_api import Request, Strategy, StrategyContext
from autotrader.strategies_api.loader import load_strategy
from autotrader.strategies_api.manifest import ParamValue
from autotrader.validation.inputs import prepare
from autotrader.validation.poisoning import future_poisoning_test, poison_after

ROOT = Path(__file__).resolve().parents[2]
# The owner's sniper method trades about once in 300 random-walk days, so comparing signals would prove
# nothing. For it the check compares its whole state after every bar instead (bias, zones, liquidity, M15
# arming): hundreds of decisions, each of which would differ if it saw the future.
STATE_TRACED = {"smc_sniper"}
LOOSE = {
    "smc_sniper": {"disp_atr": 0.6, "window_m": 480, "liq_lookback": 6, "zone_atr": 0.8, "max_sl_atr": 2.0}
}
LIBRARY = sorted(p for p in (ROOT / "strategies" / "library").iterdir() if (p / "strategy.yaml").exists())


@pytest.fixture(scope="module")
def gold() -> pl.DataFrame:
    """Gold-like synthetic prices (the real XAUUSD contract spec comes from config/instruments.yaml)."""
    return generate(
        SyntheticSpec(
            symbol="XAUUSD",
            days=300,
            seed=11,
            start_price=2000.0,
            pip_size=0.01,
            annual_vol=0.16,
            base_spread_pips=25,
        )
    )


def test_the_library_has_the_owner_requested_styles() -> None:
    assert {p.name for p in LIBRARY} >= {
        "swing_trend_pullback",
        "candle_sr_reversal",
        "scalp_session_breakout",
        "smc_sniper",
    }


@pytest.mark.parametrize("path", LIBRARY, ids=lambda p: p.name)
def test_library_strategy_is_a_candidate_without_lookahead(path: Path, gold: pl.DataFrame) -> None:
    ls = load_strategy(path)  # AST static checks run here
    m = ls.manifest
    assert not m.demo_only and m.origin == "owner"
    assert "CANDIDATE" in (path / "strategy.py").read_text()  # says so where a reader looks first
    instruments, _ = load_instruments(ROOT / "config" / "instruments.yaml")
    t0 = gold["open_time"][0]
    cut = t0 + timedelta(days=260)
    if m.id in STATE_TRACED:
        clean, poisoned = (
            _state_trace(ls.cls, df, gold, instruments["XAUUSD"], cut, LOOSE[m.id])
            for df in (gold, poison_after(gold, cut, seed=0))
        )
        before = [t for t in clean if t[0] < to_ns(cut)]
        assert before == [t for t in poisoned if t[0] < to_ns(cut)]
        assert sum('"hunt:XAUUSD": {' in st for _, st in before) > 0, (
            "never armed: the check would prove little"
        )
        return
    rep = future_poisoning_test(ls.cls, {"XAUUSD": gold}, {"XAUUSD": instruments["XAUUSD"]}, cut)
    assert rep.passed, rep.first_difference
    assert rep.signals_checked > 0, "no signals before the cut: the poisoning check would prove nothing"


def _state_trace(
    cls: type[Strategy],
    df: pl.DataFrame,
    clean: pl.DataFrame,
    inst: Instrument,
    cut: datetime,
    params: Mapping[str, ParamValue],
) -> list[tuple[int, str]]:
    """The strategy's state after every bar it sees; the cost model comes from clean data before the cut."""
    trace: list[tuple[int, str]] = []

    class Traced(cls):  # type: ignore[valid-type,misc]
        def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
            out: list[Request] = super().on_bar(ctx, event)
            trace.append((to_ns(event.at), json.dumps(dict(ctx.state), sort_keys=True, default=str)))
            return out

    spread_src = {"XAUUSD": clean.filter(pl.col("open_time") < cut)}
    inp = prepare({"XAUUSD": df}, cls.manifest, {"XAUUSD": inst}, spread_frames=spread_src)
    run_backtest(Traced, inp.m1, inp.series, inp.instruments, inp.cost_model, params=params)
    return trace
