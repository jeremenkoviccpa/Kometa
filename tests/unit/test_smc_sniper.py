"""The owner's SMC sniper method, step by step, on hand-built bars: the full M1 sequence triggers, and each
missing step means NO TRADE (every rejection has an accepted control, the same bars with one step changed)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from autotrader.strategies_api.loader import load_strategy

ROOT = Path(__file__).resolve().parents[2]
Bar = tuple[float, float, float, float]  # open, high, low, close


@pytest.fixture(scope="module")
def smc() -> ModuleType:
    return sys.modules[load_strategy(ROOT / "strategies" / "library" / "smc_sniper").cls.__module__]


def _sequence(**change: Bar) -> list[Bar]:
    """Quiet bars with a lower high at 18, then sweep, displacement with a gap, BOS, pullback, engulfing."""
    bars: list[Bar] = [(100.0, 100.3, 99.7, 100.0)] * 25
    bars[18] = (100.0, 101.0, 99.8, 100.0)  # the last lower high: the CHOCH level
    bars += [
        (100.0, 100.1, 99.0, 99.9),  # 25 sweeps the 99.7 lows and closes back above: rejection
        (99.9, 100.2, 99.8, 100.0),  # 26
        (100.0, 101.7, 99.95, 101.6),  # 27 displacement closes through 101.0: CHOCH
        (101.6, 102.0, 100.5, 101.8),  # 28 its low stays above bar 26's high: FVG 100.2-100.5
        (101.8, 102.4, 101.7, 102.3),  # 29 closes above the leg high: BOS
        (102.3, 102.35, 101.2, 101.3),  # 30 pullback
        (101.3, 101.35, 100.4, 100.6),  # 31 into the gap, no close through it
        (100.55, 101.5, 100.45, 101.4),  # 32 bullish engulfing: confirmation
    ]
    for k, v in change.items():
        bars[int(k[1:])] = v
    return bars


def _cols(bars: list[Bar]) -> tuple[list[float], list[float], list[float], list[float]]:
    o, h, lo, c = (list(x) for x in zip(*bars, strict=True))
    return o, h, lo, c


def test_the_full_m1_sequence_triggers(smc: ModuleType) -> None:
    seq = smc.m1_sequence(_cols(_sequence()), 30, 10, 1.5)
    assert seq is not None
    assert (seq["sweep"], seq["swept"], seq["choch"]) == (99.0, 99.7, 101.0)
    assert (seq["fvg_lo"], seq["fvg_hi"], seq["confirm"]) == (100.2, 100.5, 2.0)


@pytest.mark.parametrize(
    "change",
    [
        {"b32": (101.4, 101.5, 100.45, 100.55)},  # no confirmation candle (bearish instead)
        {"b27": (100.9, 101.2, 99.95, 101.05)},  # CHOCH by a weak candle, not a displacement
        {"b28": (101.6, 102.0, 100.1, 101.8)},  # no fair value gap
        {"b29": (101.8, 101.95, 101.7, 101.9)},  # no BOS
        {"b31": (101.3, 101.35, 100.0, 100.1)},  # the pullback closes through the gap
        {"b25": (100.0, 100.1, 99.75, 99.9)},  # no sweep of the lows
        {"b25": (100.0, 100.1, 99.0, 99.3), "b26": (99.3, 99.6, 99.1, 99.5)},  # no rejection
    ],
    ids=[
        "no-confirmation",
        "weak-choch",
        "no-fvg",
        "no-bos",
        "gap-closed-through",
        "no-sweep",
        "no-rejection",
    ],
)
def test_a_missing_step_is_no_trade(smc: ModuleType, change: dict[str, Bar]) -> None:
    assert smc.m1_sequence(_cols(_sequence(**change)), 30, 10, 1.5) is None


def test_pin_bar_also_confirms(smc: ModuleType) -> None:
    seq = smc.m1_sequence(_cols(_sequence(b32=(100.9, 101.0, 100.3, 100.95))), 30, 10, 1.5)
    assert seq is not None and seq["confirm"] == 1.0


def _block(tail: list[Bar]) -> list[Bar]:
    bars: list[Bar] = [(100.0, 100.3, 99.7, 100.0)] * 20
    bars += [(100.0, 100.1, 99.6, 99.7), (99.7, 102.0, 99.65, 101.9), (101.9, 102.2, 100.6, 102.1)]
    return bars + tail


def test_order_block_is_the_last_opposite_candle_before_a_displacement_with_a_gap(smc: ModuleType) -> None:
    zones = smc.order_blocks(_cols(_block([(102.1, 102.3, 101.0, 101.5)])), 1.5)
    assert [(lo, hi) for lo, hi, _ in zones] == [(99.6, 100.1)]


def test_a_mitigated_order_block_is_dropped(smc: ModuleType) -> None:
    assert smc.order_blocks(_cols(_block([(102.1, 102.3, 99.0, 99.5)])), 1.5) == []


def test_liquidity_is_only_swing_highs_nothing_has_traded_through(smc: ModuleType) -> None:
    bars: list[Bar] = [(100.0, 100.5, 99.5, 100.0)] * 5
    bars[2] = (100.0, 103.0, 99.5, 100.0)  # a swing high at 103
    bars += [(100.0, 100.5, 99.5, 100.0)] * 3
    assert smc.liquidity_above(_cols(bars)) == [103.0]
    bars.append((100.0, 103.5, 99.5, 101.0))  # traded through: the liquidity is taken
    assert smc.liquidity_above(_cols(bars)) == []
