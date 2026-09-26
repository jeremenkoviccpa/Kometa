"""L3 meta filter on live signals (spec 14.5): a strategy version wrapped with a meta model that passed the
out-of-sample gate. The wrapper is the same code as a new version (origin learning_meta, parent the plain
version); before a signal leaves, it takes the market snapshot the model learned from and drops the signal if
the predicted chance of reaching the target is below the training base rate. It never adds, moves or grows a
signal. Such a version enters as a challenger in shadow; the lifecycle's champion/challenger rule decides.

Model files are pickles (loading one runs code), so a model loads only if its sha256 matches its registry
record. Inference is deterministic.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from autotrader.core.events import BarClosed
from autotrader.core.models import Signal, Timeframe
from autotrader.learning.features import snapshot
from autotrader.learning.journal import KEEP
from autotrader.strategies_api.base import FillView, Request, Strategy, StrategyContext


class ModelIntegrityError(Exception):
    pass


def model_records(models_dir: Path) -> list[dict[str, Any]]:
    path = models_dir / "registry.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_model(models_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """The model object, only if the file is the one its record describes."""
    blob = (models_dir / str(record["file"])).read_bytes()
    if hashlib.sha256(blob).hexdigest() != record["sha256"]:
        raise ModelIntegrityError(f"{record['file']}: sha256 does not match the model registry")
    obj: dict[str, Any] = pickle.loads(blob)  # noqa: S301 - our own artifact, verified by its hash above
    return obj


def meta_filtered(
    cls: type[Strategy], model: Mapping[str, Any], model_id: str, version: str
) -> type[Strategy]:
    """`cls` as a new version whose signals pass through the meta model first."""
    base = cls.manifest
    tfs = tuple(dict.fromkeys([*base.timeframes, *KEEP]))  # the model's timeframes too
    manifest = base.model_copy(
        update={
            "version": version,
            "origin": "learning_meta",
            "timeframes": tfs,
            "description": f"{base.description} Meta filter {model_id}: skips signals it expects to fail.",
        }
    )
    own = set(base.timeframes)
    predictor, names, base_rate = model["model"], list(model["features"]), float(model["base_rate"])

    class MetaFiltered(cls):  # type: ignore[valid-type,misc]
        meta_model_id: ClassVar[str] = model_id

        def warmup(self) -> dict[tuple[str, Timeframe], int]:
            w = dict(super().warmup())
            for s in manifest.symbols:
                for tf, n in KEEP.items():
                    w[(s, tf)] = max(w.get((s, tf), 0), min(n, 60))
            return w

        def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
            if event.timeframe not in own:
                return []  # the model's extra timeframes are not the strategy's business
            out: list[Request] = []
            for req in super().on_bar(ctx, event):
                if not isinstance(req, Signal):
                    out.append(req)
                    continue
                p = self._probability(ctx, req)
                meta = dict(ctx.state.get("meta") or {"kept": 0, "skipped": 0})
                if p >= base_rate:
                    meta["kept"] += 1
                    out.append(
                        req.model_copy(update={"tags": {**req.tags, "meta_p": f"{p:.3f}", "meta": model_id}})
                    )
                else:
                    meta["skipped"] += 1
                ctx.state["meta"] = meta
            return out

        def on_fill(self, ctx: StrategyContext, fill: FillView) -> list[Request]:
            if fill.kind == "exit":  # the rolling win rate the model was trained with (target first = a win)
                last = list(ctx.state.get("meta_last20") or [])[-19:]
                ctx.state["meta_last20"] = [*last, 1 if fill.exit_reason == "target" else 0]
            return list(super().on_fill(ctx, fill))

        def _probability(self, ctx: StrategyContext, sig: Signal) -> float:
            bars = {tf: ctx.market.bars(sig.symbol, tf, n) for tf, n in KEEP.items()}
            m5 = bars[Timeframe.M5]
            quoted = float(m5.ask_c[-1] - m5.bid_c[-1]) if len(m5) else 0.0  # as in the training journal
            row = snapshot(bars, ctx.market.now, spread=quoted).row()
            row["side_buy"] = 1.0 if sig.side == "buy" else 0.0
            last = list(ctx.state.get("meta_last20") or [])
            row["win_rate_20"] = sum(last) / len(last) if last else math.nan
            x = np.array([[row.get(k, math.nan) for k in names]], dtype=np.float64)
            return float(predictor.predict_proba(x)[0, 1])

    MetaFiltered.manifest = manifest
    MetaFiltered.__name__ = f"{cls.__name__}MetaFiltered"
    return MetaFiltered
