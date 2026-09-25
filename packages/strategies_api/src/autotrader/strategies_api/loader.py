"""Load a strategy plugin from its directory: manifest, AST checks, then import.

Layout: `<dir>/strategy.yaml` and `<dir>/strategy.py` defining exactly one
`Strategy` subclass. Code under `strategies/generated/` is refused here; it
may only run inside the container sandbox (spec 14.3).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

from autotrader.strategies_api.base import Strategy
from autotrader.strategies_api.manifest import StrategyManifest
from autotrader.strategies_api.static_checks import Violation, check_file


class StrategyLoadError(Exception):
    pass


@dataclass(frozen=True)
class LoadedStrategy:
    cls: type[Strategy]
    manifest: StrategyManifest
    code_hash: str
    path: Path


def code_hash(directory: Path) -> str:
    """sha256 over the relative paths and contents of all .py and .yaml files, sorted."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".yaml"} and "__pycache__" not in p.parts:
            h.update(p.relative_to(directory).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def static_check_dir(directory: Path) -> list[tuple[Path, Violation]]:
    out: list[tuple[Path, Violation]] = []
    for p in sorted(directory.rglob("*.py")):
        if p.name.startswith("test_") or "tests" in p.relative_to(directory).parts:
            continue
        out += [(p, v) for v in check_file(p)]
    return out


def load_strategy(directory: Path, *, allow_generated: bool = False) -> LoadedStrategy:
    directory = directory.resolve()
    if "generated" in directory.parts and not allow_generated:
        raise StrategyLoadError(f"{directory}: generated strategies may only be loaded inside the sandbox")
    manifest_path, code_path = directory / "strategy.yaml", directory / "strategy.py"
    if not manifest_path.exists() or not code_path.exists():
        raise StrategyLoadError(f"{directory}: needs strategy.yaml and strategy.py")
    manifest = StrategyManifest.load(manifest_path)
    violations = static_check_dir(directory)
    if violations:
        detail = "; ".join(f"{p.name} {v}" for p, v in violations)
        raise StrategyLoadError(f"{manifest.id}: static checks failed: {detail}")

    digest = code_hash(directory)
    mod_name = f"autotrader_strategy_{manifest.id}_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(mod_name, code_path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"cannot import {code_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    classes = [
        c
        for _, c in inspect.getmembers(module, inspect.isclass)
        if issubclass(c, Strategy) and c is not Strategy and c.__module__ == mod_name
    ]
    if len(classes) != 1:
        raise StrategyLoadError(f"{code_path}: expected exactly one Strategy subclass, found {len(classes)}")
    cls = classes[0]
    cls.manifest = manifest
    return LoadedStrategy(cls, manifest, digest, directory)
