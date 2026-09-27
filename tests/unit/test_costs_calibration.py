"""L8 cost-model calibration (spec 14.11): "spread by hour of week from live quotes, slippage from
execution_quality", "a new version that is more pessimistic activates automatically; one that is more
optimistic needs at least 200 fills per affected symbol and an owner-visible alert, and activates only after
4 weeks of consistent evidence", "cost model versions ... their hash is part of every trial". Each refusal
has an accepted control."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from autotrader.cli.main import _cost_override
from autotrader.core.broker import ExecutionQuality
from autotrader.core.settings import Settings
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.costs import hour_of_week
from autotrader.learning.costs import (
    MIN_FILLS,
    CostRegistry,
    CostVersion,
    SpreadSamples,
    calibrate,
    decide,
    slippage_mults,
)
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.inputs import CostOverride, prepare

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 21, 10, tzinfo=UTC)  # a Monday


def fill(slip: str, spread: str, symbol: str = "XAUUSD") -> ExecutionQuality:
    return ExecutionQuality(
        client_order_id="c",
        account_id="a",
        strategy_id="s",
        strategy_version="1.0.0",
        symbol=symbol,
        side="buy",
        order_type="market",
        lots=Decimal("0.1"),
        requested_price=Decimal("2000"),
        filled_price=Decimal("2000.1"),
        spread_at_fill=Decimal(spread),
        slippage=Decimal(slip),
        latency_ms=80.0,
        filled_at=T0,
    )


def version(spread: float, slip: float, fills: int, status: str = "candidate") -> CostVersion:
    return CostVersion(
        version_id=f"v{spread}{slip}{fills}",
        created_at=T0.isoformat(),
        spread_by_hour={"XAUUSD": [spread] * 168},
        slippage_mult={"XAUUSD": slip},
        evidence={"XAUUSD": {"quotes": 1000, "fills": fills}},
        status=status,
    )


def test_spreads_by_hour_of_week_and_slippage_from_fills() -> None:
    s = SpreadSamples()
    t = int(T0.timestamp() * 1e9)
    for v in (0.2, 0.3, 0.4):
        s.add("XAUUSD", t, v)
    s.add("XAUUSD", t + 3_600_000_000_000, 1.0)
    med = s.medians()["XAUUSD"]
    h = hour_of_week(t)
    assert med[h] == pytest.approx(0.3) and med[h + 1] == 1.0 and med[(h + 5) % 168] is None
    mults, n = slippage_mults([fill("0.1", "0.4"), fill("0.3", "0.4"), fill("0.1", "0")])
    assert mults == {"XAUUSD": pytest.approx(0.5)} and n == {"XAUUSD": 2}  # a zero spread cannot scale
    cal = calibrate(s, [fill("0.1", "0.4")], T0)
    assert cal.evidence["XAUUSD"] == {"quotes": 4, "fills": 1} and len(cal.version_id) == 16


def test_the_first_calibration_and_any_costlier_one_activate_at_once() -> None:
    first = decide(version(0.3, 0.2, 10), None)
    assert first.activate and first.version.status == "active"
    costlier = decide(version(0.4, 0.3, 10), version(0.3, 0.2, 10, "active"))
    assert costlier.activate and costlier.alert is None


def test_a_cheaper_model_waits_for_fills_and_four_weeks_and_says_so() -> None:
    active = version(0.4, 0.3, 10, "active")
    cheap = version(0.3, 0.2, MIN_FILLS)
    d = decide(cheap, active)
    assert not d.activate and d.version.status == "pending" and "week 1" in (d.alert or "")
    pending = [version(0.3, 0.2, MIN_FILLS, "pending")] * 3
    ok = decide(cheap, active, [active, *pending])
    assert ok.activate and ok.alert is not None and "activated" in ok.alert  # the 4th week in a row
    few = decide(version(0.3, 0.2, MIN_FILLS - 1), active, [active, *pending])
    assert not few.activate  # 4 weeks but 199 fills: still waiting


def test_the_registry_keeps_every_calibration_and_the_newest_active_is_in_force(tmp_path: Path) -> None:
    reg = CostRegistry(tmp_path / "costs.jsonl")
    assert reg.active() is None
    reg.add(version(0.3, 0.2, 10, "active"))
    reg.add(version(0.2, 0.1, 10, "pending"))
    assert reg.active() == version(0.3, 0.2, 10, "active") and len(reg.versions()) == 2


def test_validation_runs_under_a_calibrated_model() -> None:
    demo = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
    frame = generate(SyntheticSpec(symbol="SYNTH", days=30, seed=1))
    plain = prepare({"SYNTH": frame}, demo.manifest, {"SYNTH": synthetic_instrument()}, synthetic=True)
    hours: list[float | None] = [None] * 168
    hours[10] = 9.99
    costs = CostOverride("v1", {"SYNTH": hours}, {"SYNTH": 0.7})
    cal = prepare(
        {"SYNTH": frame}, demo.manifest, {"SYNTH": synthetic_instrument()}, synthetic=True, costs=costs
    )
    a, b = plain.cost_model.spreads.median_by_hour["SYNTH"], cal.cost_model.spreads.median_by_hour["SYNTH"]
    assert b[10] == 9.99 and np.array_equal(np.delete(a, 10), np.delete(b, 10))  # None keeps the data's hour
    assert (
        cal.cost_model.slippage_mult_by_symbol == {"SYNTH": 0.7}
        and plain.cost_model.slippage_mult_by_symbol == {}
    )


def test_validate_takes_the_active_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AT_MODELS_DIR", str(tmp_path))
    assert _cost_override(Settings()) is None
    CostRegistry(tmp_path / "cost_registry.jsonl").add(version(0.3, 0.2, 10, "active"))
    o = _cost_override(Settings())
    assert o is not None and o.version_id == "v0.30.210" and o.slippage_mult == {"XAUUSD": 0.2}
