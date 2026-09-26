"""The checks every assistant draft passes before the owner may save it, run in a child process with a time
limit (a draft stuck in a loop, or crashing, cannot take the demo down with it):

1. load: the manifest and the AST static checks (allowed imports only, no open/eval/exec/getattr...),
2. a smoke backtest on synthetic data for each symbol (does it run, does it trade),
3. the future-poisoning test (no lookahead), meaningful only when it signalled before the cut.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

SMOKE_DAYS = 150
CUT_DAYS = 120
TIMEOUT_S = 300.0


def _start_price(symbol: str) -> float:
    if symbol.startswith("XAU"):
        return 2000.0
    if symbol.startswith("XAG"):
        return 25.0
    return 150.0 if symbol.endswith("JPY") else 1.1


def run_checks(directory: Path, root: Path) -> dict[str, Any]:
    """The checks in this process (the child process calls this)."""
    from autotrader.data.instruments import load_instruments  # noqa: PLC0415 - imported in the child only
    from autotrader.data.synthetic import SyntheticSpec, generate  # noqa: PLC0415
    from autotrader.engine.backtest import run_backtest  # noqa: PLC0415
    from autotrader.strategies_api.loader import load_strategy  # noqa: PLC0415
    from autotrader.validation.inputs import prepare  # noqa: PLC0415
    from autotrader.validation.poisoning import future_poisoning_test  # noqa: PLC0415

    out: dict[str, Any] = {"ok": False}
    try:
        ls = load_strategy(directory)
    except Exception as e:  # the draft's own error, shown to the assistant to fix
        out.update(stage="load", error=f"{type(e).__name__}: {e}"[:2000])
        return out
    m = ls.manifest
    out["manifest"] = {"id": m.id, "symbols": list(m.symbols), "timeframes": [t.value for t in m.timeframes]}
    instruments, _ = load_instruments(root / "config" / "instruments.yaml")
    unknown = [s for s in m.symbols if s not in instruments]
    if unknown:
        out.update(stage="symbols", error=f"unknown symbols {unknown}; known: {sorted(instruments)}")
        return out
    try:
        frames = {
            s: generate(
                SyntheticSpec(
                    symbol=s,
                    days=SMOKE_DAYS,
                    seed=7,
                    start_price=_start_price(s),
                    pip_size=float(instruments[s].pip_size),
                    annual_vol=0.15,
                )
            )
            for s in m.symbols
        }
        insts = {s: instruments[s] for s in m.symbols}
        inp = prepare(frames, m, insts, synthetic=True)
        res = run_backtest(ls.cls, inp.m1, inp.series, inp.instruments, inp.cost_model)
        r = [t.r_multiple for t in res.trades]
        out["backtest"] = {
            "days": SMOKE_DAYS,
            "trades": len(r),
            "win_rate": sum(x > 0 for x in r) / len(r) if r else None,
            "avg_r": sum(r) / len(r) if r else None,
            "note": "synthetic prices: shows it runs and trades, says nothing about profit",
        }
        cut = min(f["open_time"][0] for f in frames.values()) + timedelta(days=CUT_DAYS)
        rep = future_poisoning_test(ls.cls, frames, insts, cut)
        out["lookahead"] = {
            "passed": rep.passed,
            "signals_checked": rep.signals_checked,
            "first_difference": str(rep.first_difference) if rep.first_difference else None,
        }
    except Exception as e:  # a crash inside the strategy while running
        out.update(stage="run", error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"[:3000])
        return out
    out["ok"] = bool(rep.passed)
    if not rep.passed:
        out.update(stage="lookahead", error=f"future poisoning changed its signals: {rep.first_difference}")
    return out


def _child(directory: str, root: str, q: Any) -> None:
    q.put(run_checks(Path(directory), Path(root)))


def check_draft(directory: Path, root: Path, timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """run_checks in a fresh child process, killed after `timeout` seconds."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(str(directory), str(root), q), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        return {
            "ok": False,
            "stage": "timeout",
            "error": f"the checks did not finish in {timeout:.0f} s (a loop?)",
        }
    if q.empty():
        return {"ok": False, "stage": "crash", "error": f"the check process died (exit code {p.exitcode})"}
    result: dict[str, Any] = q.get()
    return result
