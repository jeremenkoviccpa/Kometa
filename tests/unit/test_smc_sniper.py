"""The owner's sniper method (smc_sniper 2.0), engine by engine, on hand-built bars: the full M5 entry
sequence triggers, and each missing step means NO TRADE (each rejection is the accepted sequence with one step
changed). Zones and liquidity are checked the same way."""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path
from types import ModuleType

import pytest

from autotrader.strategies_api.loader import load_strategy

ROOT = Path(__file__).resolve().parents[2]
Bar = tuple[float, float, float, float]  # open, high, low, close


@pytest.fixture(scope="module")
def smc() -> ModuleType:
    return sys.modules[load_strategy(ROOT / "strategies" / "library" / "smc_sniper").cls.__module__]


def _cols(bars: list[Bar]) -> tuple[list[float], list[float], list[float], list[float]]:
    o, h, lo, c = (list(x) for x in zip(*bars, strict=True))
    return o, h, lo, c


# ---------------------------------------------------------------- the structure shift and the retest


def _sequence(**change: Bar) -> list[Bar]:
    """Quiet bars with a swing high at 20 (101.0), then sweep, reject, break, and the first retest."""
    bars: list[Bar] = [(100.0, 100.3, 99.7, 100.0)] * 28
    bars[20] = (100.0, 101.0, 99.8, 100.0)  # the last swing high before the sweep: the level to break
    bars += [
        (100.0, 100.1, 99.0, 99.9),  # 28 sweeps the 99.7 lows and closes back above: rejection
        (99.9, 100.2, 99.8, 100.0),  # 29
        (100.0, 101.6, 99.95, 101.5),  # 30 a strong candle closes through 101.0: displacement + BOS
        (101.5, 101.9, 101.3, 101.8),  # 31 away from the level
        (101.2, 101.7, 100.9, 101.5),  # 32 back to 101.0, holds, closes up: the first retest, entry
    ]
    for k, v in change.items():
        bars[int(k[1:])] = v
    return bars


def _trigger(smc: ModuleType, bars: list[Bar]) -> bool:
    sh = smc.shift(_cols(bars), 13, 10, 1.0)
    return sh is not None and smc.first_retest(_cols(bars), sh["level"], int(sh["break"]))


def test_sweep_rejection_displacement_break_and_first_retest_trigger(smc: ModuleType) -> None:
    sh = smc.shift(_cols(_sequence()), 13, 10, 1.0)
    assert sh is not None and (sh["sweep"], sh["swept"], sh["level"], sh["break"]) == (
        99.0,
        99.7,
        101.0,
        30.0,
    )
    assert _trigger(smc, _sequence())


@pytest.mark.parametrize(
    "change",
    [
        {"b28": (100.0, 100.1, 99.75, 99.9)},  # no sweep of the lows
        {"b30": (100.6, 101.2, 99.95, 101.05)},  # a weak candle through the level: no displacement
        {"b31": (101.5, 101.9, 100.8, 100.9)},  # closes back below the broken level
        {"b32": (101.8, 102.2, 101.6, 102.1)},  # no retest yet
        {"b32": (101.5, 101.7, 100.9, 101.2)},  # the retest bar closes down: no rejection
        {"b31": (101.5, 101.9, 100.95, 101.8)},  # an earlier retest already happened: never chase
    ],
    ids=[
        "no-sweep",
        "no-displacement",
        "lost-level",
        "no-retest",
        "retest-down",
        "second-retest",
    ],
)
def test_a_missing_step_is_no_trade(smc: ModuleType, change: dict[str, Bar]) -> None:
    assert not _trigger(smc, _sequence(**change))


def test_no_rejection_within_three_bars_is_no_trade(smc: ModuleType) -> None:
    held = [
        (100.0, 100.1, 99.0, 99.3),  # 28 sweeps and closes below the swept 99.7
        (99.3, 99.6, 99.1, 99.5),  # 29 still below
        (99.5, 99.65, 99.2, 99.6),  # 30 still below: three bars without a rejection
        (99.6, 101.6, 99.55, 101.5),  # 31 the break comes too late
        (101.5, 101.9, 101.3, 101.8),
        (101.2, 101.7, 100.9, 101.5),
    ]
    assert not _trigger(smc, _sequence()[:28] + held)
    held[2] = (99.5, 99.9, 99.2, 99.8)  # control: bar 30 closes back above 99.7, a rejection in time
    assert _trigger(smc, _sequence()[:28] + held)


def test_a_pullback_within_the_tolerance_is_the_retest(smc: ModuleType) -> None:
    near = _sequence(b32=(101.3, 101.7, 101.05, 101.5))  # comes within 0.05 of 101.0, closes up
    sh = smc.shift(_cols(near), 13, 10, 1.0)
    assert sh is not None
    assert not smc.first_retest(_cols(near), sh["level"], int(sh["break"]))  # exact: not a touch
    assert smc.first_retest(_cols(near), sh["level"], int(sh["break"]), tol=0.1)  # within 0.1: the retest
    deep = _sequence(b31=(101.5, 101.9, 101.3, 100.85))  # a close 0.15 below the level since the break
    assert not smc.first_retest(_cols(deep), sh["level"], int(sh["break"]), tol=0.1)  # the level is lost


def test_the_sweep_must_come_after_the_start(smc: ModuleType) -> None:
    assert smc.shift(_cols(_sequence()), 29, 10, 1.0) is None  # the M5 hunt starts at the M15 sweep
    assert smc.shift(_cols(_sequence()), 28, 10, 1.0) is not None  # control


# ---------------------------------------------------------------- the score, sessions, levels


def test_the_owner_weights_as_written_and_their_decision_bands(smc: ModuleType) -> None:
    # the owner's table sums to 95, not 100 (25 + 15 + 15 + 15 + 20 + 5): kept as written, owner told
    keys = [
        "d1_bias", "h4_bias", "h1_bias", "clear_structure", "bos_choch", "htf_confluence",
        "liquidity_target", "sweep", "supply_demand", "support_resistance", "location",
        "displacement", "retest", "rejection", "ema", "rr",
    ]  # fmt: skip
    assert smc.score_setup(dict.fromkeys(keys, True)) == 95
    assert smc.score_setup({**dict.fromkeys(keys, True), "h4_bias": False, "d1_bias": False}) == 80
    assert [smc.quality(x) for x in (69, 70, 80, 90)] == ["NO TRADE", "WATCH", "VALID SNIPER", "A+ SNIPER"]


def test_sessions_by_utc_hour(smc: ModuleType) -> None:
    hour = 3_600_000_000_000
    assert [smc.session_of(h * hour) for h in (3, 8, 13, 20)] == ["Asia", "London", "New York", None]


def test_previous_day_and_week_highs(smc: ModuleType) -> None:
    day = 86_400_000_000_000
    mon = 4 * day  # 1970-01-05 was a Monday
    opens = [mon + i * day for i in range(10)]  # Mon..Fri, Sat, Sun, Mon, Tue, Wed
    highs = [100.0, 104.0, 101.0, 102.0, 103.0, 99.0, 99.0, 98.0, 97.0, 96.0]
    bars = [(0.0, h, 0.0, 0.0) for h in highs]
    assert smc.calendar_levels(_cols(bars), opens) == [96.0, 104.0]  # yesterday's high, last week's high


# ---------------------------------------------------------------- the zone engine


def _block(tail: list[Bar]) -> list[Bar]:
    bars: list[Bar] = [(100.0, 100.3, 99.7, 100.0)] * 20
    bars += [(100.0, 100.1, 99.6, 99.7), (99.7, 102.0, 99.65, 101.9), (101.9, 102.2, 100.6, 102.1)]
    return bars + tail


def test_order_block_is_the_last_opposite_candle_before_a_displacement_with_a_gap(smc: ModuleType) -> None:
    zones = smc.order_blocks(_cols(_block([(102.1, 102.3, 101.0, 101.5)])), 1.5)
    assert [(lo, hi) for lo, hi, _ in zones] == [(99.6, 100.1)]


def test_a_mitigated_order_block_is_dropped(smc: ModuleType) -> None:
    assert smc.order_blocks(_cols(_block([(102.1, 102.3, 99.0, 99.5)])), 1.5) == []


def _ranging(legs: list[tuple[float, float]]) -> list[Bar]:
    path = [a + (b - a) * i / 8 for a, b in legs for i in range(8)]
    return [(p, max(p, q) + 0.5, min(p, q) - 0.5, q) for p, q in pairwise(path)]


def test_a_tested_support_below_price_is_a_zone_until_it_breaks(smc: ModuleType) -> None:
    legs = [(110, 100), (100, 110), (110, 100), (100, 110), (110, 100.2), (100.2, 110), (110, 112)]
    [(lo, hi)] = smc.support_zones(_cols(_ranging(legs)), 0.3)
    assert lo < 100.0 < hi and hi - lo < 1.5
    broken = [*legs[:-1], (110, 95), (95, 98)]  # price falls through it and stays below
    assert smc.support_zones(_cols(_ranging(broken)), 0.3) == []


# ---------------------------------------------------------------- liquidity (targets)


def test_liquidity_is_untaken_swing_highs_and_equal_highs_pool(smc: ModuleType) -> None:
    bars: list[Bar] = [(100.0, 100.5, 99.5, 100.0)] * 20
    bars[5] = (100.0, 103.05, 99.5, 100.0)  # a swing high at 103.05
    bars[12] = (100.0, 103.0, 99.5, 100.0)  # a second one just under it (a higher one would take the first)
    liq = smc.liquidity_above(_cols(bars))
    assert 103.0 in liq and liq.count(103.05) == 2  # the swing itself and the equal-highs pool
    bars.append((100.0, 103.5, 99.5, 101.0))  # price trades through both: the liquidity is taken
    assert smc.liquidity_above(_cols(bars)) == []
