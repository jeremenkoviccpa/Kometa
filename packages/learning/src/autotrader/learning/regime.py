"""L4 regime (spec 14.6): each day's market regime, which regimes a strategy does badly in, and a filter.

The regime is rule based and explainable, from closed daily bars only (features.snapshot): the trend from
Kaufman's efficiency ratio over 20 daily closes (trending >= 0.35, ranging <= 0.20, mixed between) and the
volatility from where the daily ATR sits in its last 100 values (high >= 0.80, low <= 0.20, normal between).
A strategy's journaled outcomes are split by regime; a regime is proposed as blocked only with >= 20 signals
in it and >= 20 outside, and a gap of >= 0.3R per signal, worded as the hypothesis it is. The filter is a new
version of the same code that skips signals in the blocked regimes; like any version it must pass full
validation before it may enter as a challenger. SPEC-QUESTION: the spec also names cross-pair correlation;
it waits until the demo trades more than one market (open question 38).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from autotrader.core.events import BarClosed
from autotrader.core.models import Signal, Timeframe
from autotrader.learning.features import FeatureSnapshot, snapshot
from autotrader.learning.journal import JournalEntry
from autotrader.strategies_api.base import Request, Strategy, StrategyContext

TRENDING, RANGING = 0.35, 0.20
HIGH_VOL, LOW_VOL = 0.80, 0.20
MIN_N, MIN_GAP = 20, 0.3
D1_BARS = 120  # enough for the 100-day ATR percentile


def regime(f: FeatureSnapshot) -> tuple[str | None, str | None]:
    """(trend, volatility) or None where the daily history is too short to know."""
    e, v = f.efficiency_d1, f.atr_pct_d1
    trend = None if e is None else "trending" if e >= TRENDING else "ranging" if e <= RANGING else "mixed"
    vol = None if v is None else "high vol" if v >= HIGH_VOL else "low vol" if v <= LOW_VOL else "normal vol"
    return trend, vol


def labels(f: FeatureSnapshot) -> list[str]:
    return [x for x in regime(f) if x is not None]


@dataclass(frozen=True)
class RegimeStat:
    regime: str
    n: int
    win_rate: float
    avg_r: float
    rest_n: int
    rest_avg_r: float


def by_regime(entries: Iterable[JournalEntry]) -> list[RegimeStat]:
    done = [(set(labels(e.features)), e.outcome) for e in entries if e.outcome is not None]
    names = sorted({x for ls, _ in done for x in ls})
    out = []
    for name in names:
        inside = [o for ls, o in done if name in ls and o is not None]
        rest = [o for ls, o in done if name not in ls and o is not None]
        if not inside:
            continue
        out.append(
            RegimeStat(
                regime=name,
                n=len(inside),
                win_rate=sum(o.label == 1 for o in inside) / len(inside),
                avg_r=sum(o.r for o in inside) / len(inside),
                rest_n=len(rest),
                rest_avg_r=sum(o.r for o in rest) / len(rest) if rest else 0.0,
            )
        )
    return out


def blocked(stats: Sequence[RegimeStat]) -> list[RegimeStat]:
    """Regimes the evidence says to skip: enough signals on both sides and a real gap."""
    return [
        s
        for s in stats
        if s.n >= MIN_N and s.rest_n >= MIN_N and s.rest_avg_r - s.avg_r >= MIN_GAP and s.avg_r < 0
    ]


def finding(stats: Sequence[RegimeStat]) -> str:
    bad = blocked(stats)
    if not bad:
        return f"no regime stands out ({len(stats)} compared); no filter proposed"
    parts = [f"{s.regime}: {s.avg_r:+.2f}R over {s.n} signals vs {s.rest_avg_r:+.2f}R otherwise" for s in bad]
    return "worse in " + "; ".join(parts) + f". A hypothesis: {len(stats)} regimes were compared."


def regime_filtered(cls: type[Strategy], skip: Iterable[str], version: str) -> type[Strategy]:
    """`cls` as a new version that skips its signals in the given regimes (known from closed daily bars)."""
    skip_set = frozenset(skip)
    base = cls.manifest
    manifest = base.model_copy(
        update={
            "version": version,
            "origin": "learning_regime",
            "timeframes": tuple(dict.fromkeys([*base.timeframes, Timeframe.D1])),
            "description": f"{base.description} Regime filter: skips {', '.join(sorted(skip_set))}.",
        }
    )
    own = set(base.timeframes)

    class RegimeFiltered(cls):  # type: ignore[valid-type,misc]
        regime_skip: ClassVar[frozenset[str]] = skip_set

        def warmup(self) -> dict[tuple[str, Timeframe], int]:
            w = dict(super().warmup())
            for s in manifest.symbols:
                w[(s, Timeframe.D1)] = max(w.get((s, Timeframe.D1), 0), 30)
            return w

        def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
            if event.timeframe not in own:
                return []
            out: list[Request] = []
            for req in super().on_bar(ctx, event):
                if isinstance(req, Signal):
                    d1 = ctx.market.bars(req.symbol, Timeframe.D1, D1_BARS)
                    now = set(labels(snapshot({Timeframe.D1: d1}, ctx.market.now, spread=0.0)))
                    if now & skip_set:
                        n = dict(ctx.state.get("regime_skipped") or {})
                        for x in now & skip_set:
                            n[x] = n.get(x, 0) + 1
                        ctx.state["regime_skipped"] = n
                        continue
                out.append(req)
            return out

    RegimeFiltered.manifest = manifest
    RegimeFiltered.__name__ = f"{cls.__name__}RegimeFiltered"
    return RegimeFiltered


def summary(entries: Iterable[JournalEntry]) -> dict[str, Any]:
    stats = by_regime(entries)
    return {
        "stats": [s.__dict__ for s in stats],
        "blocked": [s.regime for s in blocked(stats)],
        "finding": finding(stats),
    }
