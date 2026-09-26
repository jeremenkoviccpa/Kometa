"""The build process checks itself (spec section 0, item 9).

Rules that a machine can enforce live here instead of only in CLAUDE.md.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIRS = ("packages", "config", "strategies", "migrations")
FENCED_CONFIG = ("risk.yaml", "validation.yaml", "promotion.yaml")


def _files(*dirs: str) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        out += [
            p for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".sql"}
        ]
    return out


def test_every_spec_question_is_logged() -> None:
    open_q = (ROOT / "docs" / "open_questions.md").read_text()
    missing = []
    for f in _files(*CODE_DIRS):
        if "SPEC-QUESTION" in f.read_text(errors="ignore"):
            rel = f.relative_to(ROOT).as_posix()
            # accept the path with or without the packages/<pkg>/src/autotrader/ prefix
            short = re.sub(r"^packages/\w+/src/autotrader/", "", rel)
            if rel not in open_q and short not in open_q:
                missing.append(rel)
    assert missing == [], f"SPEC-QUESTION without an entry in docs/open_questions.md: {missing}"


def test_done_phases_have_retros() -> None:
    progress = (ROOT / "docs" / "progress.md").read_text()
    done = re.findall(r"^\|\s*(\d+)\s*\|\s*done\s*\|", progress, flags=re.MULTILINE)
    assert done, "progress.md has no done phases or the table format changed"
    missing = [n for n in done if not (ROOT / "docs" / "retros" / f"phase_{n}.md").exists()]
    assert missing == [], f"phases marked done without a retro: {missing}"


def test_learning_and_research_do_not_reference_fenced_config() -> None:
    offenders = []
    for pkg in ("research", "learning"):
        for f in (ROOT / "packages" / pkg).rglob("*.py"):
            text = f.read_text()
            offenders += [f"{f.relative_to(ROOT)}: {name}" for name in FENCED_CONFIG if name in text]
    assert offenders == []


def test_claude_md_has_lessons_section() -> None:
    text = (ROOT / "CLAUDE.md").read_text()
    assert "# Lessons" in text
    assert "TRADING_SYSTEM_SPEC.md" in text


def test_mutation_anchors_still_exist() -> None:
    """Every mutation in tests/mutation/mutations.py still applies (the suite cannot rot silently)."""
    spec = importlib.util.spec_from_file_location("mutations", ROOT / "tests" / "mutation" / "mutations.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    MUTATIONS = mod.MUTATIONS  # noqa: N806

    stale = [m.name for m in MUTATIONS if (ROOT / m.file).read_text().count(m.old) != 1]
    assert stale == [], f"mutation anchors no longer match the code: {stale}"
    assert all((ROOT / t).exists() for m in MUTATIONS for t in m.tests)


def test_no_mutation_run_left_the_tree_mutated() -> None:
    """scripts/mutate.py writes this file before each edit and deletes it after the restore. Its presence
    means a run died with a safety rule broken in the source; `make mutate` restores it. (Phase 8: an
    interrupted run left the live-bar grace rule mutated, and nothing noticed until tests failed.)"""
    leftover = ROOT / "var" / "mutate-restore.json"
    assert not leftover.exists(), f"{leftover} exists: run `make mutate` to restore the mutated file"


# modules that parse YAML that is not a config/ file loaded by a running service
YAML_NOT_CONFIG = {
    "packages/strategies_api/src/autotrader/strategies_api/manifest.py",  # strategy manifests (code hash)
    "packages/cli/src/autotrader/cli/main.py",  # `at risk sign` validates a file before the owner signs it
    "packages/ai/src/autotrader/ai/builder.py",  # the assistant's strategy manifests (code hash), not config
}


def test_config_loaders_hash_what_they_read() -> None:
    """Spec section 17: every config file is hashed at load time. A module that parses YAML must read it
    through core.configs.read_config, which records the hash for the ConfigChanged audit record."""
    offenders = []
    for path in sorted((ROOT / "packages").glob("*/src/autotrader/**/*.py")):
        rel = str(path.relative_to(ROOT))
        text = path.read_text()
        if "yaml.safe_load(" in text and "read_config(" not in text and rel not in YAML_NOT_CONFIG:
            offenders.append(rel)
    assert offenders == [], f"config read without core.configs.read_config: {offenders}"


def test_the_vercel_frontend_is_the_current_hub() -> None:
    """web/index.html is the hub page copied for Vercel; a stale copy once hid every new screen (2026-09-26).
    After changing the hub: `uv run python scripts/build_web.py <railway url>`, then redeploy `web/`."""
    web = ROOT / "web" / "index.html"
    hub = ROOT / "packages" / "api" / "src" / "autotrader" / "api" / "dashboard.html"
    assert not web.exists() or web.read_bytes() == hub.read_bytes(), (
        "web/index.html is stale: run build_web.py"
    )
