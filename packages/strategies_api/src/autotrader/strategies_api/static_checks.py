"""AST checks run on strategy code BEFORE it is imported or executed (spec section 6).

These checks catch honest mistakes and obvious escapes. They are not a security
boundary on their own: generated code additionally runs in the network-less
container sandbox (spec 14.3).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

MAX_LINES = 2000

ALLOWED_IMPORT_PREFIXES = (
    "autotrader.strategies_api",
    "autotrader.core.indicators",
    "autotrader.core.models",
    "autotrader.core.events",
    "autotrader.core.series",
    "numpy",
    "math",
    "dataclasses",
    "typing",
    "enum",
    "__future__",
    "collections.abc",
)
FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "memoryview",
        "type",
        "object",
    }
)
# attribute names that reach outside the sandbox or behind a copy
FORBIDDEN_ATTRS = frozenset(
    {
        "utcnow",
        "today",
        "fromtimestamp",
        "base",  # numpy view base -> underlying buffer
        "__dict__",
        "__class__",
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "f_back",
        "f_globals",
        "gi_frame",
        "tofile",
        "fromfile",
        "load",
        "save",
        "savez",
        "loadtxt",
        "genfromtxt",
        "memmap",
        "ctypeslib",
        "random",  # np.random: use ctx.rng
    }
)


@dataclass(frozen=True)
class Violation:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


def _is_market(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "market"


def check_source(source: str, filename: str = "<strategy>") -> list[Violation]:
    out: list[Violation] = []
    lines = source.count("\n") + 1
    if lines > MAX_LINES:
        out.append(Violation(1, f"{lines} lines; max is {MAX_LINES}"))
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [Violation(e.lineno or 1, f"syntax error: {e.msg}")]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not a.name.startswith(ALLOWED_IMPORT_PREFIXES):
                    out.append(Violation(node.lineno, f"import of {a.name!r} not allowed"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                out.append(Violation(node.lineno, "relative imports not allowed"))
            elif not (node.module or "").startswith(ALLOWED_IMPORT_PREFIXES):
                out.append(Violation(node.lineno, f"import from {node.module!r} not allowed"))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(Violation(node.lineno, f"use of {node.id!r} not allowed"))
        elif isinstance(node, ast.Attribute) and node.attr == "now" and not _is_market(node.value):
            out.append(Violation(node.lineno, "wall clock not allowed; use ctx.market.now"))
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            out.append(Violation(node.lineno, f"attribute {node.attr!r} not allowed"))
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            out.append(Violation(node.lineno, f"dunder attribute {node.attr!r} not allowed"))
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            out.append(Violation(node.lineno, "async code not allowed"))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.append(Violation(node.lineno, "global/nonlocal not allowed"))
    return sorted(out, key=lambda v: v.line)


def check_file(path: Path) -> list[Violation]:
    return check_source(path.read_text(), str(path))
