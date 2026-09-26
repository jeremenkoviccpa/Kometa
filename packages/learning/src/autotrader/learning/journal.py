"""Trade journal (spec 14.2): every signal, taken or not, with the market at signal time and its outcome.

The outcome of every signal is followed on M1 bars whether or not it was traded (the triple barrier: target
first, stop first, or time out), so what learns later sees the signals the risk gate rejected and the
shadow ones too, not only the trades that happened. A signal that became a trade also gets the trade's real
R. Stop and target touched in the same M1 bar count as the stop (the pessimistic reading).

Entries are appended to a JSONL file on every change and the newest line per signal wins on reload.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from autotrader.core.broker import Quote
from autotrader.core.bus import DECISIONS, INTENTS, QUOTES, SIGNALS, TRADES, Handler
from autotrader.core.clock import Clock
from autotrader.core.events import (
    BarClosed,
    OrderIntentCreated,
    PositionClosed,
    QuoteUpdate,
    RiskDecided,
    SignalEmitted,
)
from autotrader.core.indicators import EventIndex
from autotrader.core.models import Bar, Timeframe
from autotrader.core.series import BarsArray, to_ns
from autotrader.engine.live_bars import LiveBarBuilder
from autotrader.learning.features import FeatureSnapshot, snapshot

KEEP = {Timeframe.M5: 400, Timeframe.H1: 400, Timeframe.H4: 300, Timeframe.D1: 120}
# how long an untouched signal is followed before it times out, by the strategy's own timeframe
HORIZON = {
    Timeframe.M1: timedelta(days=1),
    Timeframe.M5: timedelta(days=1),
    Timeframe.M15: timedelta(days=1),
    Timeframe.H1: timedelta(days=5),
    Timeframe.H4: timedelta(days=20),
    Timeframe.D1: timedelta(days=60),
}
Status = Literal["signalled", "approved", "resized", "rejected"]
VERDICT: dict[str, Status] = {"approve": "approved", "resize": "resized", "reject": "rejected"}


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: Literal[1, -1, 0]  # target first, stop first, timed out
    r: float  # R at the barrier, from the price the signal would have got
    mae_r: float  # worst excursion against, in R (positive)
    mfe_r: float  # best excursion in favour, in R
    minutes: int
    resolved_at: datetime


class JournalEntry(BaseModel):
    signal_id: str
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Literal["buy", "sell"]
    timeframe: Timeframe
    created_at: datetime
    entry: float
    stop: float
    target: float | None
    shadow: bool
    status: Status = "signalled"
    reasons: list[str] = []
    strategy_win_rate_20: float | None  # the strategy's last 20 resolved signals, at signal time
    features: FeatureSnapshot
    mae_r: float = 0.0
    mfe_r: float = 0.0
    outcome: Outcome | None = None
    trade_r: float | None = None  # the real trade's R, when it was taken


def _arrays(rows: Sequence[dict[str, float | int]]) -> BarsArray:
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    if not rows:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        return BarsArray(empty_i, empty_i, *([empty_f] * 9))
    return BarsArray(
        cols["open_time"].astype(np.int64),
        cols["close_time"].astype(np.int64),
        *(
            cols[k].astype(np.float64)
            for k in ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "volume")
        ),
    )


def _row(b: Bar) -> dict[str, float | int]:
    return {
        "open_time": to_ns(b.open_time),
        "close_time": to_ns(b.close_time),
        **{
            k: getattr(b, k)
            for k in ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "volume")
        },
    }


def follow(e: JournalEntry, bar: Bar) -> JournalEntry:
    """One closed M1 bar of the signal's life: excursions, and the outcome if a barrier is reached."""
    if e.outcome is not None or bar.open_time < e.created_at:
        return e
    buy = e.side == "buy"
    risk = e.entry - e.stop if buy else e.stop - e.entry
    if not risk > 0:
        return e
    # a buy exits on the bid, a sell on the ask
    worst = bar.bid_l if buy else bar.ask_h
    best = bar.bid_h if buy else bar.ask_l
    mae = max(e.mae_r, (e.entry - worst) / risk if buy else (worst - e.entry) / risk)
    mfe = max(e.mfe_r, (best - e.entry) / risk if buy else (e.entry - best) / risk)
    minutes = int((bar.close_time - e.created_at).total_seconds() // 60)
    stopped = worst <= e.stop if buy else worst >= e.stop
    hit = e.target is not None and (best >= e.target if buy else best <= e.target)
    done: Outcome | None = None
    if stopped:  # first, also when both are in this bar
        done = Outcome(label=-1, r=-1.0, mae_r=mae, mfe_r=mfe, minutes=minutes, resolved_at=bar.close_time)
    elif hit and e.target is not None:
        r = (e.target - e.entry) / risk if buy else (e.entry - e.target) / risk
        done = Outcome(label=1, r=r, mae_r=mae, mfe_r=mfe, minutes=minutes, resolved_at=bar.close_time)
    elif bar.close_time - e.created_at >= HORIZON[e.timeframe]:
        px = bar.bid_c if buy else bar.ask_c
        r = (px - e.entry) / risk if buy else (e.entry - px) / risk
        done = Outcome(label=0, r=r, mae_r=mae, mfe_r=mfe, minutes=minutes, resolved_at=bar.close_time)
    return e.model_copy(update={"mae_r": mae, "mfe_r": mfe, "outcome": done})


class JournalService:
    name = "journal"

    def __init__(
        self,
        clock: Clock,
        symbols: Sequence[str],
        *,
        path: Path | None = None,
        history: Mapping[tuple[str, Timeframe], BarsArray] | None = None,
        events: Callable[[], EventIndex | None] | None = None,
    ) -> None:
        self.clock = clock
        self.path = path
        self.events = events
        subs = [(s, tf) for s in symbols for tf in (Timeframe.M1, *KEEP)]
        self.builder = LiveBarBuilder(subs)
        self.bars: dict[tuple[str, Timeframe], deque[dict[str, float | int]]] = {
            (s, tf): deque(maxlen=n) for s in symbols for tf, n in KEEP.items()
        }
        for (s, tf), buf in self.bars.items():
            h = (history or {}).get((s, tf))
            if h is not None:
                start = max(0, len(h) - buf.maxlen) if buf.maxlen else 0
                for i in range(start, len(h)):
                    buf.append({f: getattr(h, f)[i].item() for f in h.__dataclass_fields__})
        self.quotes: dict[str, Quote] = {}
        self.entries: dict[str, JournalEntry] = {}
        self.open: set[str] = set()
        self.intents: dict[str, str] = {}  # intent id -> signal id
        self._load()

    def handlers(self) -> Mapping[str, Handler]:
        return {
            QUOTES: self._on_quote,
            SIGNALS: self._on_signal,
            INTENTS: self._on_intent,
            DECISIONS: self._on_decision,
            TRADES: self._on_trade,
        }

    # ------------------------------------------------------------ persistence

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if line.strip():
                e = JournalEntry.model_validate_json(line)
                self.entries[e.signal_id] = e
        self.open = {k for k, e in self.entries.items() if e.outcome is None}

    def _save(self, e: JournalEntry) -> None:
        self.entries[e.signal_id] = e
        if e.outcome is None:
            self.open.add(e.signal_id)
        else:
            self.open.discard(e.signal_id)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(e.model_dump_json() + "\n")

    # ------------------------------------------------------------ market

    async def _on_quote(self, msg: Any) -> None:
        if not isinstance(msg, QuoteUpdate):
            return
        q = Quote(symbol=msg.symbol, bid=Decimal(str(msg.bid)), ask=Decimal(str(msg.ask)), time=msg.at)
        self.quotes[q.symbol] = q
        self.on_bars(self.builder.on_quote(q))

    async def on_time(self, now: datetime) -> None:
        self.on_bars(self.builder.on_time(now))

    def on_bars(self, events: Sequence[BarClosed]) -> None:
        for ev in events:
            key = (ev.symbol, ev.timeframe)
            if key in self.bars:
                self.bars[key].append(_row(ev.bar))
            if ev.timeframe == Timeframe.M1:
                for sid in [s for s in self.open if self.entries[s].symbol == ev.symbol]:
                    e = self.entries[sid]
                    n = follow(e, ev.bar)
                    if n.outcome is not None:
                        self._save(n)
                    else:
                        self.entries[sid] = n  # excursions so far: saved with the outcome

    # ------------------------------------------------------------ the pipeline

    def win_rate_20(self, strategy_id: str) -> float | None:
        done = [e for e in self.entries.values() if e.strategy_id == strategy_id and e.outcome is not None]
        done.sort(key=lambda e: e.created_at)
        last = done[-20:]
        return sum(e.outcome.label == 1 for e in last if e.outcome) / len(last) if last else None

    async def _on_signal(self, msg: Any) -> None:
        if not isinstance(msg, SignalEmitted) or str(msg.signal.signal_id) in self.entries:
            return
        s = msg.signal
        q = self.quotes.get(s.symbol)
        spread = float(q.ask - q.bid) if q is not None else 0.0
        entry = s.entry_price
        if entry is None:  # market: what it would really pay
            entry = float(q.ask if s.side == "buy" else q.bid) if q is not None else s.stop_price
        idx = self.events() if self.events is not None else None
        feats = snapshot(
            {tf: _arrays(list(self.bars[(s.symbol, tf)])) for tf in KEEP},
            s.created_at,
            spread=spread,
            currencies=(s.symbol[:3], s.symbol[3:]) if len(s.symbol) == 6 else None,
            events=idx,
        )
        self._save(
            JournalEntry(
                signal_id=str(s.signal_id),
                strategy_id=s.strategy_id,
                strategy_version=s.strategy_version,
                symbol=s.symbol,
                side=s.side,
                timeframe=msg.timeframe,
                created_at=s.created_at,
                entry=float(entry),
                stop=s.stop_price,
                target=s.target_price,
                shadow=msg.shadow,
                strategy_win_rate_20=self.win_rate_20(s.strategy_id),
                features=feats,
            )
        )

    async def _on_intent(self, msg: Any) -> None:
        if isinstance(msg, OrderIntentCreated):
            self.intents[str(msg.intent.intent_id)] = str(msg.intent.signal.signal_id)

    async def _on_decision(self, msg: Any) -> None:
        if not isinstance(msg, RiskDecided):
            return
        d = msg.decision
        e = self.entries.get(self.intents.get(str(d.intent_id), ""))
        if e is not None:
            status = VERDICT[d.verdict]
            self._save(e.model_copy(update={"status": status, "reasons": list(d.reasons)}))

    async def _on_trade(self, msg: Any) -> None:
        if not isinstance(msg, PositionClosed):
            return
        t = msg.trade
        mine = [
            e
            for e in self.entries.values()
            if e.strategy_id == t.strategy_id
            and e.strategy_version == t.strategy_version
            and e.symbol == t.symbol
            and e.side == t.side
            and e.trade_r is None
            and e.status in ("approved", "resized")
            and e.created_at <= t.entry_time
        ]
        if mine:  # the latest signal before the entry is the one that became this trade
            e = max(mine, key=lambda x: x.created_at)
            self._save(e.model_copy(update={"trade_r": t.r_multiple}))

    # ------------------------------------------------------------ the hub

    def view(self, recent: int = 40) -> dict[str, Any]:
        es = sorted(self.entries.values(), key=lambda e: e.created_at)
        by: dict[str, list[JournalEntry]] = {}
        for e in es:
            by.setdefault(f"{e.strategy_id} {e.strategy_version}", []).append(e)
        rows = []
        for k, xs in by.items():
            done = [e.outcome for e in xs if e.outcome is not None]
            rows.append(
                {
                    "strategy": k,
                    "signals": len(xs),
                    "traded": sum(e.status in ("approved", "resized") for e in xs),
                    "rejected": sum(e.status == "rejected" for e in xs),
                    "shadow": sum(e.shadow for e in xs),
                    "resolved": len(done),
                    "target_first": sum(o.label == 1 for o in done),
                    "stop_first": sum(o.label == -1 for o in done),
                    "timed_out": sum(o.label == 0 for o in done),
                    "avg_r": _mean([o.r for o in done]),
                    "avg_mae_r": _mean([o.mae_r for o in done]),
                    "avg_mfe_r": _mean([o.mfe_r for o in done]),
                }
            )
        return {
            "signals": len(es),
            "resolved": sum(e.outcome is not None for e in es),
            "following": len(self.open),
            "by_strategy": rows,
            "recent": [json.loads(e.model_dump_json()) for e in reversed(es[-recent:])],
        }


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None
