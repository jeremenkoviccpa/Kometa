from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

import autotrader.cli.main as cli
from autotrader.cli.main import main
from autotrader.core.broker import SymbolInfo
from autotrader.execution.fake import FakeBroker


def test_synth_then_check_is_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "synth", "--days", "7", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "SYNTH_M1.parquet").exists()
    assert "data_version=synthetic-" in capsys.readouterr().out
    assert main(["data", "check", "SYNTH", "--root", str(tmp_path)]) == 0


def test_risk_sign_verify_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = tmp_path / "risk.yaml"
    cfg.write_text((root / "config" / "risk.yaml").read_text())
    key, pub = tmp_path / "k", tmp_path / "p.pub"
    assert main(["risk", "keygen", "--key", str(key), "--pub", str(pub)]) == 0
    assert oct(key.stat().st_mode)[-3:] == "600"
    assert main(["risk", "keygen", "--key", str(key), "--pub", str(pub)]) == 1  # no silent overwrite
    assert main(["risk", "sign", str(cfg), "--key", str(key)]) == 0
    assert main(["risk", "verify", str(cfg), "--pub", str(pub)]) == 0
    cfg.write_text(cfg.read_text().replace("0.015", "0.5"))
    assert main(["risk", "verify", str(cfg), "--pub", str(pub)]) == 1
    capsys.readouterr()
    assert main(["risk", "resume-token", "--key", str(key)]) == 0
    assert '"resume_full_halt"' in capsys.readouterr().out


def test_execution_check_is_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Now:
        def now(self) -> datetime:
            return datetime.now(UTC)

    xau = SymbolInfo(
        symbol="XAUUSD",
        digits=2,
        point=Decimal("0.01"),
        contract_size=Decimal(100),
        min_lot=Decimal("0.01"),
        lot_step=Decimal("0.01"),
        max_lot=Decimal(50),
    )
    broker = FakeBroker(symbols=[xau], clock=Now())
    broker.set_quote("XAUUSD", "4341.80", "4342.00")
    broker.open_external("XAUUSD", "buy", Decimal("0.01"))
    monkeypatch.setattr(cli, "_make_adapter", lambda _s: broker)
    monkeypatch.setenv("AT_ENV", "paper")
    assert main(["execution", "check"]) == 0
    out = capsys.readouterr().out
    assert "fake-1 demo hedging" in out and "external 1" in out
    assert not [c for c in broker.calls if c[0] in ("place", "modify", "cancel", "close_position")]
    monkeypatch.setenv("AT_ENV", "live")  # a demo account must never pass as live
    assert main(["execution", "check"]) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_lifecycle_submit_status_retire(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("AT_REGISTRY_PATH", str(tmp_path / "registry.jsonl"))
    assert main(["lifecycle", "submit", str(root / "strategies" / "examples" / "demo_ma_cross")]) == 0
    assert "shadow (demo_only, capped)" in capsys.readouterr().out
    assert main(["lifecycle", "status"]) == 0
    assert "demo_ma_cross" in (out := capsys.readouterr().out) and "shadow" in out
    assert main(["lifecycle", "retire", "demo_ma_cross", "1.0.0", "--reason", "done"]) == 0
    assert main(["lifecycle", "retire", "demo_ma_cross", "1.0.0", "--reason", "again"]) == 1


def test_audit_verify_reports_intact_and_broken(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from autotrader.core.ledger import JsonlLedger  # noqa: PLC0415
    from autotrader.monitor.audit import AuditLog  # noqa: PLC0415

    p = tmp_path / "audit.jsonl"
    log = AuditLog(JsonlLedger(p))
    for i in range(3):
        log.record("x", "t", {"i": i})
    assert main(["audit", "verify", "--path", str(p)]) == 0
    assert "3 records, chain intact" in capsys.readouterr().out
    p.write_text(p.read_text().replace('"i":1', '"i":7'))
    assert main(["audit", "verify", "--path", str(p)]) == 1
    assert "BROKEN" in capsys.readouterr().out
