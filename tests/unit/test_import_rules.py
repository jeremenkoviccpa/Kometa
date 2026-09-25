"""Package dependency rules (spec section 4), enforced by static import analysis.

Two checks: (1) actual imports stay inside the allowed graph; (2) pyproject
internal dependencies never widen it. Changing ALLOWED is a design decision and
must be logged in docs/decisions.md.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

from autotrader.strategies_api.static_checks import ALLOWED_IMPORT_PREFIXES

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "packages"

ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "strategies_api": {"core"},
    "data": {"core"},
    "engine": {"core", "strategies_api"},
    "validation": {"core", "engine", "strategies_api", "data"},
    "lifecycle": {"core"},
    "allocator": {"core"},
    "risk": {"core"},
    "execution": {"core"},  # talks to risk only via bus message schema in core
    "mt5_bridge": {"core"},
    "research": {"core", "lifecycle", "strategies_api", "data", "engine", "validation"},
    "learning": {"core", "lifecycle", "strategies_api", "data", "engine", "validation"},
    "monitor": {"core"},
    "api": {"core", "lifecycle", "monitor"},
    # the owner's Claude tracks: they read bars and publish signals; sizing and orders stay elsewhere
    "ai": {"core", "engine", "strategies_api"},
    "cli": {
        "core",
        "data",
        "engine",
        "validation",
        "lifecycle",
        "risk",
        "strategies_api",
        "monitor",
        "execution",
        "allocator",  # `at demo run` composes every service in one process (decisions.md, Phase 8)
        "api",
        "ai",
    },
}
NEVER = {
    "research": {"risk", "execution", "allocator"},
    "learning": {"risk", "execution", "allocator"},
    "ai": {"risk", "execution", "allocator"},
}
# The CLI may import risk only for owner-side signing (`at risk sign`); nothing else may.
RISK_IMPORTERS = {"cli"}

# single source of truth: the same list the loader enforces before importing strategy code
STRATEGY_ALLOWED_PREFIXES = ALLOWED_IMPORT_PREFIXES


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(f"{path}: relative imports are not allowed")
            if node.module:
                mods.add(node.module)
    return mods


def internal_targets(mods: set[str]) -> set[str]:
    return {m.split(".")[1] for m in mods if m.startswith("autotrader.") and m.count(".") >= 1}


@pytest.mark.parametrize("pkg", sorted(ALLOWED))
def test_package_imports_follow_rules(pkg: str) -> None:
    src = PACKAGES / pkg / "src" / "autotrader" / pkg
    assert src.is_dir(), f"missing package dir {src}"
    violations = []
    for f in src.rglob("*.py"):
        for target in internal_targets(imported_modules(f)) - {pkg}:
            if target not in ALLOWED[pkg] or target in NEVER.get(pkg, set()):
                violations.append(f"{f.relative_to(ROOT)} imports autotrader.{target}")
            if target == "risk" and pkg not in RISK_IMPORTERS:
                violations.append(f"{f.relative_to(ROOT)} imports risk")
    assert violations == []


def test_allowed_graph_respects_never_rules() -> None:
    for pkg, banned in NEVER.items():
        assert not (ALLOWED[pkg] & banned)
    assert all("risk" not in deps for p, deps in ALLOWED.items() if p not in RISK_IMPORTERS)
    assert ALLOWED["core"] == set() and ALLOWED["strategies_api"] == {"core"} and ALLOWED["risk"] == {"core"}


@pytest.mark.parametrize("pkg", sorted(ALLOWED))
def test_pyproject_does_not_widen_graph(pkg: str) -> None:
    meta = tomllib.loads((PACKAGES / pkg / "pyproject.toml").read_text())
    deps = {
        d.split(">")[0].split("=")[0].strip().removeprefix("autotrader-").replace("-", "_")
        for d in meta["project"]["dependencies"]
        if d.startswith("autotrader-")
    }
    assert deps <= ALLOWED[pkg], f"{pkg} pyproject declares {deps - ALLOWED[pkg]}"


def test_strategy_code_imports_are_restricted() -> None:
    violations = []
    for f in (ROOT / "strategies").rglob("*.py"):
        if "tests" in f.parts or f.name.startswith("test_"):
            continue
        for m in imported_modules(f):
            if not m.startswith(STRATEGY_ALLOWED_PREFIXES):
                violations.append(f"{f.relative_to(ROOT)} imports {m}")
    assert violations == []


# The owner's one exception to "no LLM in the live order path" is packages/ai (paper only, decisions.md
# 2026-09-26). Research and learning may use Claude offline; no other package may import an LLM client.
LLM_CLIENTS = ("anthropic", "openai")
LLM_ALLOWED = {"ai", "research", "learning"}


@pytest.mark.parametrize("pkg", sorted(set(ALLOWED) - LLM_ALLOWED))
def test_no_llm_client_outside_the_owner_exception(pkg: str) -> None:
    for path in (PACKAGES / pkg).rglob("*.py"):
        bad = {m for m in imported_modules(path) if m.split(".")[0] in LLM_CLIENTS}
        assert not bad, f"{path}: imports {bad}; only packages/ai may reach the order path with an LLM"


def test_the_llm_check_sees_the_ai_package() -> None:  # control: the check would find a real import
    assert any(
        m.split(".")[0] == "anthropic" for p in (PACKAGES / "ai").rglob("*.py") for m in imported_modules(p)
    )
