"""Phase 3 acceptance: a full validation report for demo_ma_cross with every section."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.strategies_api.loader import LoadedStrategy, load_strategy
from autotrader.validation.config import ValidationConfig
from autotrader.validation.report import to_html, to_json, write_html, write_json
from autotrader.validation.runner import ValidationReport, Validator, default_epoch
from autotrader.validation.store import (
    HoldoutLock,
    HoldoutRefusedError,
    JsonlLedger,
    LedgerCorruptError,
    TrialRegistry,
)

ROOT = Path(__file__).resolve().parents[2]
SYMS = ["SYNTH", "SYN2", "SYN3"]

SMALL: dict[str, dict[str, float]] = {
    "walk_forward": {"train_years": 0.2, "test_months": 1, "step_months": 1, "search_budget": 2},
    "holdout": {"months": 1},
    "monte_carlo": {"runs": 300},
    "cross_market": {"min_pairs_passing": 2, "of_pairs": 3},
    "thresholds": {"min_oos_trades": 10},
}


@pytest.fixture(scope="module")
def demo() -> LoadedStrategy:
    return load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")


@pytest.fixture(scope="module")
def frames() -> dict[str, pl.DataFrame]:
    return {s: generate(SyntheticSpec(symbol=s, days=170, seed=40 + i)) for i, s in enumerate(SYMS)}


def _validate(
    demo: LoadedStrategy,
    frames: dict[str, pl.DataFrame],
    ledger: JsonlLedger,
    overrides: dict[str, dict[str, float]],
) -> ValidationReport:
    cfg = ValidationConfig.model_validate({**SMALL, **overrides})
    v = Validator(cfg, TrialRegistry(ledger), HoldoutLock(ledger, "e1", 5), config_hash="test")
    ep = default_epoch(frames["SYNTH"]["open_time"][-1], cfg.holdout.months, "e1")
    return v.validate(demo, frames, {s: synthetic_instrument(s) for s in SYMS}, ep, synthetic=True)


@pytest.fixture(scope="module")
def report(
    demo: LoadedStrategy, frames: dict[str, pl.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> tuple[ValidationReport, JsonlLedger]:
    ledger = JsonlLedger(tmp_path_factory.mktemp("v") / "ledger.jsonl")
    # thresholds relaxed so the run reaches the holdout and exercises every section
    lenient = {
        "thresholds": {
            **SMALL["thresholds"],
            "min_deflated_sharpe_prob": 0.0,
            "min_profit_factor": 0.0,
            "min_stability_profit_factor": 0.0,
            "suspicious_monthly_return": 10.0,
        }
    }
    return _validate(demo, frames, ledger, lenient), ledger


def test_report_has_every_section(report: tuple[ValidationReport, JsonlLedger], tmp_path: Path) -> None:
    rep, _ = report
    assert len(rep.windows) >= 2
    assert rep.oos_trades
    assert rep.dsr is not None
    assert rep.monte_carlo is not None
    assert len(rep.stability) == 8  # 4 tunable params x (+, -)
    assert len(rep.cross_market) == 3
    assert rep.holdout is not None  # every earlier check passed under lenient thresholds
    assert not rep.eligible_for_promotion  # synthetic data never is
    page = to_html(rep)
    for section in (
        "Thresholds",
        "Monthly returns",
        "R distribution",
        "By symbol",
        "By session",
        "By weekday",
        "Walk-forward windows",
        "Deflated Sharpe",
        "Monte Carlo",
        "Parameter stability",
        "Cross-market",
        "Locked holdout",
        "Costs",
        "Synthetic data",
    ):
        assert section in page, section
    js = to_json(rep)
    json.dumps(js)  # serializable, no NaN/inf
    assert js["code_hash"] == rep.code_hash
    assert {c["name"] for c in js["checks"]} >= {
        "oos_trades",
        "deflated_sharpe_prob",
        "holdout_profit_factor",
    }
    write_html(rep, tmp_path / "r.html")
    write_json(rep, tmp_path / "r.json")
    assert (tmp_path / "r.html").stat().st_size > 5000


def test_every_backtest_is_a_trial_and_ledger_verifies(report: tuple[ValidationReport, JsonlLedger]) -> None:
    rep, ledger = report
    reg = TrialRegistry(ledger)
    trials = reg.trials("demo_ma_cross")
    kinds = {t.kind for t in trials}
    assert kinds == {"wf_train", "wf_test", "stability", "cross_market", "holdout"}
    assert rep.dsr is not None
    assert rep.dsr.n_trials == len(trials) - 1  # DSR computed before the holdout trial was added
    assert ledger.verify() == len(list(ledger.records()))


def test_holdout_once_per_version(
    report: tuple[ValidationReport, JsonlLedger], demo: LoadedStrategy, frames: dict[str, pl.DataFrame]
) -> None:
    _, ledger = report
    lenient = {
        "thresholds": {
            **SMALL["thresholds"],
            "min_deflated_sharpe_prob": 0.0,
            "min_profit_factor": 0.0,
            "min_stability_profit_factor": 0.0,
            "suspicious_monthly_return": 10.0,
        }
    }
    again = _validate(demo, frames, ledger, lenient)
    assert again.holdout is None
    assert any("already used its holdout attempt" in w for w in again.warnings)
    assert not again.passed


def test_strict_thresholds_fail_and_do_not_consume_holdout(
    demo: LoadedStrategy, frames: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    ledger = JsonlLedger(tmp_path / "l.jsonl")
    rep = _validate(demo, frames, ledger, {})
    assert not rep.passed
    assert rep.holdout is None
    assert list(ledger.records("holdout_attempt")) == []


def test_holdout_family_budget(tmp_path: Path) -> None:
    lock = HoldoutLock(JsonlLedger(tmp_path / "h.jsonl"), "e1", max_per_family=2)
    lock.open("fam", "a", "1.0.0")
    lock.open("fam", "a", "1.0.1")
    with pytest.raises(HoldoutRefusedError, match="family"):
        lock.open("fam", "a", "1.0.2")
    lock.open("other", "b", "1.0.0")


def test_ledger_detects_tampering(tmp_path: Path) -> None:
    led = JsonlLedger(tmp_path / "t.jsonl")
    for i in range(3):
        led.append("trial", {"i": i})
    assert led.verify() == 3
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"i":1', '"i":9')
    (tmp_path / "t.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerCorruptError):
        led.verify()
