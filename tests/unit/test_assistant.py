"""The owner's strategy assistant: the conversation, the draft checks, saving (only with the owner's
confirmation, as a new version) and modifying. Claude is faked; the draft checker is faked here except in
the last tests, which run the real checks in a child process (a good draft, a forbidden import, a loop)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from autotrader.ai.builder import Assistant, next_version, normalize_manifest
from autotrader.ai.checks import check_draft
from autotrader.cli.demo import load_owner_strategies

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "strategies" / "library" / "scalp_session_breakout"
MANIFEST = (EXAMPLE / "strategy.yaml").read_text()
CODE = (EXAMPLE / "strategy.py").read_text()


def propose(uid: str, code: str = CODE, sid: str = "my_breakout") -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": uid,
        "name": "propose_strategy",
        "input": {
            "strategy_id": sid,
            "summary": "Asian range breakout.",
            "manifest_yaml": MANIFEST,
            "code": code,
        },
    }


class FakeClaude:
    def __init__(self, *replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    async def __call__(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        assert "strategy_id" in str(tools) and "Indicators" in system and "ctx.signal" in system
        assert "same language as the owner's latest message" in system
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


def assistant(tmp: Path, claude: FakeClaude | None, ok: list[bool] | None = None) -> Assistant:
    results = list(ok or [])

    async def checker(d: Path) -> dict[str, Any]:
        assert (d / "strategy.yaml").exists() and (d / "strategy.py").exists()
        good = results.pop(0) if results else True
        return {
            "ok": good,
            "stage": None if good else "load",
            "error": None if good else "static checks failed: import os",
        }

    return Assistant(ROOT, tmp, claude, sources=lambda: {"scalp_session_breakout": EXAMPLE}, checker=checker)


def text(t: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": t}]


async def test_it_asks_then_drafts_fixes_a_failed_draft_and_explains(tmp_path: Path) -> None:
    claude = FakeClaude(
        text("Which timeframe, and where does the stop go?"),
        [*text("Here is a first draft."), propose("u1", code="import os\n" + CODE)],
        [propose("u2")],  # fixed after the failed check
        text("It passed: it trades the London breakout of the Asian range. Press Save if you want it."),
    )
    a = assistant(tmp_path, claude, ok=[False, True])
    s = await a.message(None, "Trade the Asian range breakout on gold")
    assert [m["role"] for m in s["display"]] == ["owner", "assistant"] and s["draft"] is None
    s = await a.message(s["id"], "M5, stop at the middle of the range")
    roles = [m["role"] for m in s["display"]]
    assert roles == ["owner", "assistant", "owner", "assistant", "checks", "checks", "assistant"]
    assert s["draft"]["checks"]["ok"] and s["draft"]["strategy_id"] == "my_breakout"
    last_api = claude.seen[-1]
    assert [m["role"] for m in last_api] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "import os" in str(claude.seen[2][-1])  # Claude saw why the first draft failed
    assert not s["thinking"]


async def test_saving_needs_the_owners_responsibility_and_makes_a_new_version(tmp_path: Path) -> None:
    a = assistant(tmp_path, FakeClaude([propose("u1")], text("done")))
    s = await a.message(None, "breakout")
    with pytest.raises(PermissionError):
        a.save(s["id"], False, {})
    saved = a.save(s["id"], True, {})
    assert saved == {**saved, "strategy_id": "my_breakout", "version": "1.0.0", "restart_needed": True}
    m = yaml.safe_load((tmp_path / "strategies" / "my_breakout" / "strategy.yaml").read_text())
    assert (m["id"], m["version"], m["origin"]) == ("my_breakout", "1.0.0", "owner")
    head = (tmp_path / "strategies" / "my_breakout" / "strategy.yaml").read_text().splitlines()[0]
    assert "the owner is" in head
    again = a.save(s["id"], True, {"my_breakout": ["1.0.0"]})  # a second save is the next version,
    assert again["version"] == "1.1.0"
    assert (
        tmp_path / "strategies" / "_history" / "my_breakout" / "1.0.0" / "strategy.py"
    ).exists()  # old kept


async def test_nothing_is_saved_from_a_draft_that_failed(tmp_path: Path) -> None:
    a = assistant(tmp_path, FakeClaude([propose("u1")], text("a"), text("b"), text("c")), ok=[False])
    s = await a.message(None, "breakout")
    with pytest.raises(ValueError, match="passed the checks"):
        a.save(s["id"], True, {})


async def test_modifying_shows_claude_the_current_code_and_the_owner_only_their_words(tmp_path: Path) -> None:
    claude = FakeClaude(text("I will widen the range filter."))
    a = assistant(tmp_path, claude)
    s = await a.message(None, "only trade on Tuesdays", base="scalp_session_breakout")
    assert s["display"][0]["text"] == "only trade on Tuesdays" and s["base"] == "scalp_session_breakout"
    first = claude.seen[0][0]["content"]
    assert "Current strategy.py" in first and "class ScalpSessionBreakout" in first
    with pytest.raises(KeyError):
        a.post(None, "x", base="no_such_strategy")


async def test_one_reply_at_a_time_and_off_without_a_key(tmp_path: Path) -> None:
    a = assistant(tmp_path, FakeClaude(text("?")))
    sid = a.post(None, "an idea")
    assert a.view(a.session(sid))["thinking"]
    with pytest.raises(RuntimeError, match="still answering"):
        a.post(sid, "another")
    await a.reply(sid)
    assert not a.view(a.session(sid))["thinking"]
    off = assistant(tmp_path / "off", None)
    assert not off.status()["enabled"]
    with pytest.raises(RuntimeError):
        off.post(None, "x")


def test_the_manifest_is_the_owners_and_versions_count_up() -> None:
    m = yaml.safe_load(normalize_manifest(MANIFEST + "demo_only: true\n", "my_id", "2.1.0"))
    assert (m["id"], m["version"], m["origin"]) == ("my_id", "2.1.0", "owner") and "demo_only" not in m
    with pytest.raises(ValueError, match="not valid YAML"):
        normalize_manifest("a: [", "x", "1.0.0")
    assert next_version([]) == "1.0.0" and next_version(["1.0.0", "2.0.0", "1.4.0"]) == "2.1.0"


def test_saved_strategies_load_and_a_broken_one_is_skipped(tmp_path: Path) -> None:
    good = tmp_path / "my_breakout"
    shutil.copytree(EXAMPLE, good)
    (good / "strategy.yaml").write_text(normalize_manifest(MANIFEST, "my_breakout", "1.0.0"))
    bad = tmp_path / "broken"
    shutil.copytree(EXAMPLE, bad)
    (bad / "strategy.py").write_text("import os\n" + CODE)
    (tmp_path / "_history").mkdir()
    assert [ls.manifest.id for ls in load_owner_strategies(tmp_path)] == ["my_breakout"]


# ---------------------------------------------------------------- the real checks (child process)


def _draft(tmp: Path, code: str) -> Path:
    d = tmp / "draft"
    d.mkdir()
    (d / "strategy.yaml").write_text(normalize_manifest(MANIFEST, "my_breakout", "1.0.0"))
    (d / "strategy.py").write_text(code)
    return d


def test_the_real_checks_refuse_a_forbidden_import(tmp_path: Path) -> None:
    r = check_draft(_draft(tmp_path, "import os\n" + CODE), ROOT)
    assert not r["ok"] and r["stage"] == "load" and "static checks" in r["error"]


def test_the_real_checks_stop_a_draft_that_never_finishes(tmp_path: Path) -> None:
    looping = CODE.replace(
        "        if event.timeframe != Timeframe.M5:",
        "        while True:\n            pass\n        if event.timeframe != Timeframe.M5:",
    )
    r = check_draft(_draft(tmp_path, looping), ROOT, timeout=20)
    assert not r["ok"] and r["stage"] == "timeout"


@pytest.mark.slow
def test_the_real_checks_pass_a_working_strategy(tmp_path: Path) -> None:
    r = check_draft(_draft(tmp_path, CODE), ROOT)
    assert r["ok"], r
    assert r["backtest"]["trades"] > 0 and r["lookahead"]["passed"] and r["lookahead"]["signals_checked"] > 0
