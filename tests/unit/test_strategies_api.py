from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autotrader.engine.costs import InstrumentCosts
from autotrader.engine.gate import BacktestGuard, lots_for_risk
from autotrader.strategies_api.loader import StrategyLoadError, load_strategy
from autotrader.strategies_api.manifest import StrategyManifest
from autotrader.strategies_api.static_checks import check_source

ROOT = Path(__file__).resolve().parents[2]

BASE = {
    "id": "s1_test",
    "version": "1.0.0",
    "origin": "owner",
    "family": "test",
    "symbols": ["EURUSD"],
    "timeframes": ["H1"],
    "expected": {"trades_per_month": 5, "win_rate": 0.4, "avg_r": 0.3},
}


def test_manifest_limits_tunable_params() -> None:
    params = {f"p{i}": {"value": 1, "min": 0, "max": 2, "tunable": True} for i in range(7)}
    with pytest.raises(ValidationError, match="max is 6"):
        StrategyManifest.model_validate({**BASE, "params": params})
    with pytest.raises(ValidationError, match="need min and max"):
        StrategyManifest.model_validate({**BASE, "params": {"a": {"value": 1, "tunable": True}}})
    with pytest.raises(ValidationError, match="outside"):
        StrategyManifest.model_validate({**BASE, "params": {"a": {"value": 5, "min": 0, "max": 2}}})
    m = StrategyManifest.model_validate({**BASE, "params": {"a": {"value": 1, "min": 0, "max": 2}}})
    assert m.param_values({"a": 2}) == {"a": 2}
    with pytest.raises(ValueError, match="outside"):
        m.param_values({"a": 3})
    with pytest.raises(KeyError):
        m.param_values({"b": 1})


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        ("import os", "import of 'os'"),
        ("from subprocess import run", "subprocess"),
        ("x = open('f')", "'open'"),
        ("eval('1')", "'eval'"),
        ("import datetime", "datetime"),
        ("def f(ctx):\n    return ctx.bar.open_time.now()", "wall clock"),
        ("def f(a):\n    return a.base", "'base'"),
        ("def f(a):\n    return a.__class__", "__class__"),
        ("import numpy as np\nx = np.random.rand()", "'random'"),
        ("async def f():\n    pass", "async"),
        ("from . import x", "relative"),
        ("t = type(1)", "'type'"),
    ],
)
def test_static_checks_reject(code: str, fragment: str) -> None:
    v = check_source(code)
    assert v, f"not rejected: {code}"
    assert any(fragment in str(x) for x in v), [str(x) for x in v]


def test_static_checks_allow_normal_strategy_code() -> None:
    ok = (
        "from autotrader.core.indicators import ema\n"
        "import numpy as np\nimport math\n"
        "def f(ctx):\n    t = ctx.market.now\n    return math.sqrt(2) + float(np.mean([1.0]))\n"
    )
    assert check_source(ok) == []


def test_loader_loads_demo_and_refuses_generated(tmp_path: Path) -> None:
    ls = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
    assert ls.manifest.demo_only
    assert ls.cls.manifest.id == "demo_ma_cross"
    assert len(ls.code_hash) == 64
    gen = tmp_path / "generated" / "x"
    gen.mkdir(parents=True)
    with pytest.raises(StrategyLoadError, match="sandbox"):
        load_strategy(gen)


def test_loader_rejects_bad_code(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    (d / "strategy.yaml").write_text(
        (ROOT / "strategies" / "examples" / "demo_ma_cross" / "strategy.yaml").read_text()
    )
    (d / "strategy.py").write_text("import os\n")
    with pytest.raises(StrategyLoadError, match="static checks failed"):
        load_strategy(d)


XAU = InstrumentCosts("XAUUSD", "USD", 100.0, 0.01, 0.01, 0.01, 20.0, 3.5, 0, 0, "points", 2)


def test_xauusd_sizing_example() -> None:
    # spec 11: 20,000 USD, 0.5% = 100 USD, stop 10 USD away, 100 oz -> 0.10 lot
    assert lots_for_risk(20_000, 0.005, 4342.00, 4332.00, XAU, 1.0) == pytest.approx(0.10)
    # 9 lots would risk 9,000 USD (45%); sizing never produces that
    assert lots_for_risk(20_000, 0.005, 4342.00, 4332.00, XAU, 1.0) < 9
    # below min lot rejects instead of rounding up
    assert lots_for_risk(100, 0.005, 4342.00, 4332.00, XAU, 1.0) == 0.0


def test_backtest_guard_only_in_backtest() -> None:
    with pytest.raises(RuntimeError, match="backtest"):
        BacktestGuard("live")
