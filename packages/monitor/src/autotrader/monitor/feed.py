"""The live event feed for the trading hub: one short line per decision-relevant bus message.

Signals, sizing proposals, risk decisions, orders, fills, closed trades, stage changes, halts, alerts
and config loads, in the order the monitor saw them. Quotes, account updates and heartbeats are state,
not events, and are left out. The feed is for people; the audit log stays the record.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from autotrader.core.events import (
    AlertRaised,
    ConfigChanged,
    DemotionOrder,
    HaltCleared,
    HaltEntered,
    OrderCancelled,
    OrderFilled,
    OrderIntentCreated,
    OrderModified,
    OrderPlaced,
    PositionClosed,
    RiskDecided,
    SignalEmitted,
    StageChanged,
    StrategyRequestEmitted,
)
from autotrader.monitor.journey import ref_of

# pipeline steps, in order, for the hub's flow counters
STEPS = ("signal", "intent", "decision", "order", "fill", "trade")


@dataclass(frozen=True)
class FeedItem:
    seq: int
    at: datetime
    step: str  # one of STEPS, or: stage, halt, alert, config, request
    tone: str  # good | bad | warn | info | muted
    text: str
    strategy: str = ""
    symbol: str = ""
    ref: str = ""  # the trade journey this event belongs to (client order id), if any


def _px(x: Any) -> str:
    return "market" if x is None else f"{float(x):g}"


def describe(m: Any) -> tuple[str, str, str, str, str] | None:
    """(step, tone, text, strategy, symbol) for a message, or None when it is not a feed event."""
    if isinstance(m, SignalEmitted):
        s = m.signal
        who = f"{s.strategy_id} {s.strategy_version}"
        tag = " (shadow, not sized)" if m.shadow else ""
        at = "at market" if s.entry_price is None else f"{s.entry_type} {s.entry_price:g}"
        text = f"{s.side.upper()} {s.symbol} {at}, stop {s.stop_price:g}{tag}"
        return "signal", "muted" if m.shadow else "info", text, who, s.symbol
    if isinstance(m, OrderIntentCreated):
        i, s = m.intent, m.intent.signal
        text = f"allocator proposes {i.proposed_lots} lots of {s.symbol}, risk {i.risk_fraction:.3%}"
        return "intent", "info", text, f"{s.strategy_id} {s.strategy_version}", s.symbol
    if isinstance(m, RiskDecided):
        d = m.decision
        why = f": {'; '.join(d.reasons)}" if d.reasons else ""
        tone = {"approve": "good", "resize": "warn", "reject": "bad"}[d.verdict]
        return "decision", tone, f"risk gate {d.verdict}s, {d.approved_lots} lots{why}", "", ""
    if isinstance(m, OrderPlaced | OrderModified):
        o = m.order
        if isinstance(m, OrderModified) and o.lots == 0:
            text = f"position closed: {o.side} {o.symbol} (entered {o.entry_type} {_px(o.price)})"
            return "order", "muted", text, f"{o.strategy_id} {o.strategy_version}", o.symbol
        verb = "placed" if isinstance(m, OrderPlaced) else o.status.value
        at = "at market" if o.entry_type == "market" else f"{o.entry_type} {_px(o.price)}"
        text = f"order {verb}: {o.side} {o.lots} {o.symbol} {at}, stop {o.sl}"
        tone = "bad" if o.status.value == "rejected" else "info"
        return "order", tone, text, f"{o.strategy_id} {o.strategy_version}", o.symbol
    if isinstance(m, OrderFilled):
        f = m.fill
        text = f"filled {f.side} {f.lots} {f.symbol} at {f.price} ({f.latency_ms:.0f} ms)"
        return "fill", "good", text, "", f.symbol
    if isinstance(m, OrderCancelled):
        return "order", "muted", f"order {m.reason}", "", ""
    if isinstance(m, PositionClosed):
        t = m.trade
        shadow = t.account_id == "shadow"
        money = "" if shadow else f", {t.pnl_net:+.2f}"
        tone = "muted" if shadow else ("good" if t.r_multiple > 0 else "bad")
        text = f"closed {t.side} {t.symbol} {t.r_multiple:+.2f}R{money}{' (shadow)' if shadow else ''}"
        return "trade", tone, text, f"{t.strategy_id} {t.strategy_version}", t.symbol
    if isinstance(m, StageChanged):
        text = f"{m.from_stage.value} -> {m.to_stage.value}: {m.reason}"
        return "stage", "warn", text, f"{m.strategy_id} {m.strategy_version}", ""
    if isinstance(m, DemotionOrder):
        act = "closing positions" if m.close_positions else "cancelling pending entries"
        return (
            "stage",
            "bad",
            f"demotion order, {act}: {m.reason}",
            f"{m.strategy_id} {m.strategy_version}",
            "",
        )
    if isinstance(m, HaltEntered):
        return "halt", "bad", f"HALT {m.state.value}: {m.reason}", "", ""
    if isinstance(m, HaltCleared):
        return "halt", "good", f"{m.previous.value} cleared by {m.actor}", "", ""
    if isinstance(m, AlertRaised):
        a = m.alert
        tone = {"critical": "bad", "warning": "warn"}.get(a.severity.value, "info")
        return "alert", tone, f"{a.kind}: {a.message.splitlines()[0]}", "", ""
    if isinstance(m, ConfigChanged):
        return "config", "muted", f"{m.service} loaded {m.name} ({m.config_hash[:10]})", "", ""
    if isinstance(m, StrategyRequestEmitted):
        r = m.request
        return "request", "info", f"strategy asks: {type(r).__name__.removesuffix('Request').lower()}", "", ""
    return None


@dataclass
class Feed:
    items: deque[FeedItem] = field(default_factory=lambda: deque(maxlen=600))
    counts: Counter[str] = field(default_factory=Counter)
    seq: int = 0

    def add(self, m: Any) -> None:
        d = describe(m)
        if d is None:
            return
        step, tone, text, strategy, symbol = d
        self.seq += 1
        self.counts[step] += 1
        self.items.append(FeedItem(self.seq, m.at, step, tone, text, strategy, symbol, ref_of(m)))

    def after(self, seq: int, limit: int = 200) -> list[FeedItem]:
        return [i for i in self.items if i.seq > seq][-limit:]
