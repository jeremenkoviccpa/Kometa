"""Every strategy in strategies/library: loads through the static checks, sees no future (future poisoning
with signals before the cut, so the check is not vacuous), and is a candidate, never pre-approved."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from autotrader.data.instruments import load_instruments
from autotrader.data.synthetic import SyntheticSpec, generate
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.poisoning import future_poisoning_test

ROOT = Path(__file__).resolve().parents[2]
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
    rep = future_poisoning_test(ls.cls, {"XAUUSD": gold}, {"XAUUSD": instruments["XAUUSD"]}, cut)
    assert rep.passed, rep.first_difference
    assert rep.signals_checked > 0, "no signals before the cut: the poisoning check would prove nothing"
