"""Run the mutation suite: every mutation must make its tests fail. Usage: `make mutate`.

The run edits source files in place, so it guards the tree: an exclusive lock (two overlapping runs
would "restore" each other's mutations) and a restore file written before each edit. If a run dies
mid-mutation, the next run puts the original back first; `make check` fails while the file exists.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "var" / "mutate.lock"
RESTORE = ROOT / "var" / "mutate-restore.json"


def load_mutations() -> Any:
    spec = importlib.util.spec_from_file_location("mutations", ROOT / "tests" / "mutation" / "mutations.py")
    if spec is None or spec.loader is None:
        raise SystemExit("tests/mutation/mutations.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MUTATIONS


def restore_interrupted() -> None:
    """Put back the file an interrupted run left mutated."""
    if RESTORE.exists():
        saved = json.loads(RESTORE.read_text())
        (ROOT / saved["file"]).write_text(saved["original"])
        RESTORE.unlink()
        print(f"restored {saved['file']} from an interrupted run")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another mutation run is active; refusing to overlap it")
            return 2
        restore_interrupted()
        return run()


def run() -> int:
    MUTATIONS = load_mutations()  # noqa: N806
    survivors = []
    for m in MUTATIONS:
        path = ROOT / m.file
        original = path.read_text()
        if original.count(m.old) != 1:
            print(f"STALE   {m.name}: anchor text not found exactly once")
            survivors.append(m.name)
            continue
        RESTORE.write_text(json.dumps({"file": m.file, "original": original}))
        try:
            path.write_text(original.replace(m.old, m.new, 1))
            r = subprocess.run(  # noqa: S603
                ["uv", "run", "pytest", "-x", "-q", "-o", "addopts=", "-p", "no:cacheprovider", *m.tests],  # noqa: S607
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_text(original)
            RESTORE.unlink()
        killed = r.returncode != 0
        print(f"{'killed ' if killed else 'SURVIVED'} {m.name}")
        if not killed:
            survivors.append(m.name)
    print(f"{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} mutations killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
