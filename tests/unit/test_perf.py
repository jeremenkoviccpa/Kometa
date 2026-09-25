"""Phase 2 performance target (spec section 8): 10 years, one symbol, H1 signals, M1 fills, < 2 minutes."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.backtest import run_backtest
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.inputs import prepare

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_ten_year_backtest_under_two_minutes() -> None:
    ls = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
    df = generate(SyntheticSpec(days=3652, seed=5))
    t0 = time.perf_counter()
    inp = prepare({"SYNTH": df}, ls.manifest, {"SYNTH": synthetic_instrument()}, synthetic=True)
    r = run_backtest(ls.cls, inp.m1, inp.series, inp.instruments, inp.cost_model)
    elapsed = time.perf_counter() - t0
    print(f"\n10y backtest: {len(df)} M1 bars, {r.metrics.trades} trades, {elapsed:.1f}s")
    assert r.metrics.trades > 500
    assert elapsed < 120
