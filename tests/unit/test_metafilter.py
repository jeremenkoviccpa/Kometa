"""L3 on live signals: a meta-filtered version only drops signals (never adds, moves or grows one), sees no
future, loads its model only if the file matches the registry record, and can win the champion swap like a
re-optimized child. The model here is a stand-in with a known rule, so every outcome is predictable."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autotrader.cli.demo import meta_versions
from autotrader.cli.main import _meta_challenger
from autotrader.cli.main import _registry as _registry_cli
from autotrader.core.models import Stage
from autotrader.core.profile import BacktestProfile
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.backtest import run_backtest
from autotrader.learning.metafilter import ModelIntegrityError, load_model, meta_filtered, model_records
from autotrader.lifecycle.registry import VersionInfo
from autotrader.strategies_api.base import Strategy
from autotrader.strategies_api.loader import load_strategy
from autotrader.validation.inputs import prepare
from autotrader.validation.poisoning import future_poisoning_test

ROOT = Path(__file__).resolve().parents[2]
DEMO = load_strategy(ROOT / "strategies" / "examples" / "demo_ma_cross")
FEATURES = ["slope_h1", "side_buy"]


class Rule:
    """p = 0.9 when the H1 trend agrees with the side, else 0.1 (a stand-in for a trained model)."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        slope, buy = x[0, 0], x[0, 1]
        agree = (slope > 0) == (buy == 1.0) if not np.isnan(slope) else False
        p = 0.9 if agree else 0.1
        return np.array([[1 - p, p]])


class Constant:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.array([[1 - self.p, self.p]])


def model(predictor: Any, base_rate: float = 0.5) -> dict[str, Any]:
    return {"model": predictor, "features": FEATURES, "base_rate": base_rate}


def trades(cls: type[Strategy], days: int = 200) -> list[Any]:
    frame = generate(SyntheticSpec(symbol="SYNTH", days=days, seed=4))
    inp = prepare({"SYNTH": frame}, cls.manifest, {"SYNTH": synthetic_instrument()}, synthetic=True)
    return run_backtest(cls, inp.m1, inp.series, inp.instruments, inp.cost_model).trades


def test_the_filtered_version_is_a_new_learned_version_of_the_same_code() -> None:
    cls = meta_filtered(DEMO.cls, model(Constant(0.9)), "m1", "1.0.1")
    m = cls.manifest
    assert (m.id, m.version, m.origin) == ("demo_ma_cross", "1.0.1", "learning_meta")
    assert set(DEMO.manifest.timeframes) <= set(m.timeframes) and cls.meta_model_id == "m1"  # type: ignore[attr-defined]
    assert DEMO.cls.manifest.version == "1.0.0"  # the plain version is untouched


def test_it_only_drops_signals() -> None:
    plain = trades(DEMO.cls)
    keep_all = trades(meta_filtered(DEMO.cls, model(Constant(0.9)), "m", "1.0.1"))
    drop_all = trades(meta_filtered(DEMO.cls, model(Constant(0.1)), "m", "1.0.1"))
    ruled = trades(meta_filtered(DEMO.cls, model(Rule()), "m", "1.0.1"))
    assert len(plain) > 10 and drop_all == []
    ready = min(t.entry_time_ns for t in keep_all)  # the model's features need 60 daily bars first

    def key(ts: list[Any]) -> list[tuple[int, str, float]]:
        return [(t.entry_time_ns, t.side, t.stop_price) for t in ts if t.entry_time_ns >= ready]

    assert key(keep_all) == key(plain)  # kept signals are the strategy's own, unchanged
    assert 0 < len(ruled) < len(plain) and set(key(ruled)) <= set(key(plain))  # a subset, nothing new


def test_the_filtered_version_sees_no_future() -> None:
    cls = meta_filtered(DEMO.cls, model(Rule()), "m", "1.0.1")
    frame = generate(SyntheticSpec(symbol="SYNTH", days=200, seed=6))
    cut = frame["open_time"][0] + timedelta(days=160)
    rep = future_poisoning_test(cls, {"SYNTH": frame}, {"SYNTH": synthetic_instrument()}, cut)
    assert rep.passed, rep.first_difference
    assert rep.signals_checked > 0


def _registry(tmp: Path, predictor: Any, strategy: str = "scalp_session_breakout") -> dict[str, Any]:
    blob = pickle.dumps(model(predictor))
    (tmp / "m.pkl").write_bytes(blob)
    rec = {
        "model_id": "abc",
        "strategy_id": strategy,
        "strategy_version": "1.0.0",
        "file": "m.pkl",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "status": "passed_oos",
    }
    (tmp / "registry.jsonl").write_text(json.dumps(rec) + "\n")
    return rec


def test_a_model_loads_only_if_its_file_matches_the_record(tmp_path: Path) -> None:
    rec = _registry(tmp_path, Constant(0.9))
    assert load_model(tmp_path, rec)["base_rate"] == 0.5
    (tmp_path / "m.pkl").write_bytes((tmp_path / "m.pkl").read_bytes() + b"x")
    with pytest.raises(ModelIntegrityError):
        load_model(tmp_path, rec)


def test_the_demo_runs_passed_models_beside_their_strategy_and_skips_tampered_ones(tmp_path: Path) -> None:
    lib = [load_strategy(ROOT / "strategies" / "library" / "scalp_session_breakout")]
    good, bad, refused = tmp_path / "good", tmp_path / "bad", tmp_path / "refused"
    for d in (good, bad, refused):
        d.mkdir()
    _registry(good, Constant(0.9))
    _registry(bad, Constant(0.9))
    (bad / "m.pkl").write_bytes(b"tampered")
    rec = _registry(refused, Constant(0.9))
    (refused / "registry.jsonl").write_text(json.dumps({**rec, "status": "refused"}) + "\n")
    [v] = meta_versions(lib, [good, bad, refused, tmp_path / "missing"])
    assert (v.manifest.id, v.manifest.version, v.manifest.origin) == (
        "scalp_session_breakout",
        "1.0.1",
        "learning_meta",
    )
    assert model_records(tmp_path / "missing") == []


def test_the_shipped_model_matches_its_record() -> None:
    d = ROOT / "models" / "meta"
    for rec in model_records(d):
        load_model(d, rec)  # raises if the committed file and its record ever disagree
    assert model_records(d), "the demo ships at least one meta model"


def test_a_passing_model_registers_its_version_as_a_challenger_in_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AT_REGISTRY_PATH", str(tmp_path / "registry.jsonl"))
    reg, _ = _registry_cli()
    reg.submit_candidate(
        VersionInfo(
            strategy_id="s1",
            version="1.0.0",
            family="f",
            origin="owner",
            demo_only=False,
            code_hash="c",
            created_by="t",
        ),
        BacktestProfile(
            strategy_id="s1",
            strategy_version="1.0.0",
            trade_r=tuple([0.2, -1.0, 1.5, 0.4] * 100),
            weekly_entries=(5,) * 12,
            mc_dd_p95_r=8.0,
            model_slippage={},
            source="t",
            synthetic=False,
        ),
    )
    _meta_challenger("s1", "1.0.0", "abc")
    _meta_challenger("nope", "1.0.0", "abc")
    v = _registry_cli()[0].get("s1", "1.0.1")
    assert (v.stage, v.info.origin, v.info.parent_version, v.info.params) == (
        Stage.SHADOW,
        "learning_meta",
        "1.0.0",
        {"meta_model": "abc"},
    )
    assert "nope 1.0.0 is not in the registry" in capsys.readouterr().out
