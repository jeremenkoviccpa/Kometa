"""Trade journeys for the hub: every step one signal took, from the strategy to the closed trade.

All ids derive from the signal: intent id = uuid5(signal id), client order id = "at" + intent hex, and a
live trade's id is its client order id (core.broker). So a signal, its sizing proposal, the risk
decision, the order changes, the fill and the closed trade can be joined without any lookup table.
Shadow signals are never sized and have no journey. Also keeps the fills for the execution view.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from autotrader.core.broker import client_order_id, intent_id_for
from autotrader.core.events import (
    OrderCancelled,
    OrderFilled,
    OrderIntentCreated,
    OrderModified,
    OrderPlaced,
    PositionClosed,
    RiskDecided,
    SignalEmitted,
)

MAX_JOURNEYS = 300


def ref_of(m: Any) -> str:
    """The client order id a message belongs to, or "" (shadow signals, trades of other accounts)."""
    if isinstance(m, SignalEmitted):
        return "" if m.shadow else client_order_id(intent_id_for(m.signal.signal_id))
    if isinstance(m, OrderIntentCreated):
        return client_order_id(m.intent.intent_id)
    if isinstance(m, RiskDecided):
        return client_order_id(m.decision.intent_id)
    if isinstance(m, OrderPlaced | OrderModified):
        return m.order.client_order_id
    if isinstance(m, OrderFilled):
        return m.fill.client_order_id
    if isinstance(m, OrderCancelled):
        return m.client_order_id
    if isinstance(m, PositionClosed) and m.trade.account_id != "shadow":
        return m.trade.trade_id
    return ""


def _s(x: Any) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


@dataclass
class Step:
    at: datetime
    step: str
    title: str
    details: dict[str, str]


@dataclass
class Journey:
    ref: str
    strategy: str = ""
    symbol: str = ""
    side: str = ""
    status: str = "signal"  # signal, sized, rejected, sent, pending, open, closed, cancelled
    started: datetime | None = None
    updated: datetime | None = None
    r: float | None = None
    pnl: float | None = None
    steps: list[Step] = field(default_factory=list)


@dataclass
class ExecRow:
    at: datetime
    ref: str
    symbol: str
    side: str
    requested: float | None
    filled: float
    slippage: float | None  # adverse is positive, price units
    modelled: float  # the backtest cost model's slippage for this fill
    spread: float
    latency_ms: float


@dataclass
class Journeys:
    slippage_mult: float = 0.2
    slippage_mult_by_symbol: dict[str, float] = field(default_factory=dict)
    by_ref: OrderedDict[str, Journey] = field(default_factory=OrderedDict)
    fills: deque[ExecRow] = field(default_factory=lambda: deque(maxlen=500))

    def add(self, m: Any) -> None:
        ref = ref_of(m)
        if not ref:
            return
        j = self.by_ref.get(ref)
        if j is None:
            if not isinstance(m, SignalEmitted | OrderIntentCreated):
                return  # an order from before the monitor started: not a journey we can tell
            j = self.by_ref[ref] = Journey(ref=ref, started=m.at)
            while len(self.by_ref) > MAX_JOURNEYS:
                self.by_ref.popitem(last=False)
        j.updated = m.at
        self._step(j, m)

    def _step(self, j: Journey, m: Any) -> None:
        if isinstance(m, SignalEmitted):
            s = m.signal
            j.strategy, j.symbol, j.side = f"{s.strategy_id} {s.strategy_version}", s.symbol, s.side
            j.steps.append(
                Step(
                    m.at,
                    "signal",
                    f"{s.strategy_id} signals {s.side.upper()} {s.symbol}",
                    {
                        "entry": f"{s.entry_type} {_s(s.entry_price) if s.entry_price else ''}".strip(),
                        "stop": _s(s.stop_price),
                        "target": _s(s.target_price),
                        "timeframe": m.timeframe.value,
                        "reason": s.reason,
                        **{f"tag {k}": v for k, v in s.tags.items()},
                    },
                )
            )
        elif isinstance(m, OrderIntentCreated):
            i = m.intent
            s = i.signal
            j.strategy = j.strategy or f"{s.strategy_id} {s.strategy_version}"
            j.symbol, j.side, j.status = s.symbol, s.side, "sized"
            j.steps.append(
                Step(
                    m.at,
                    "intent",
                    f"allocator sizes it at {i.proposed_lots} lots",
                    {"risk per trade": f"{i.risk_fraction:.3%} of equity", "account": i.account_id},
                )
            )
        elif isinstance(m, RiskDecided):
            d = m.decision
            j.status = "rejected" if d.verdict == "reject" else "approved"
            j.steps.append(
                Step(
                    m.at,
                    "decision",
                    f"risk gate {d.verdict}s" + (f" {d.approved_lots} lots" if d.verdict != "reject" else ""),
                    {
                        "verdict": d.verdict,
                        "approved lots": str(d.approved_lots),
                        "reasons": "; ".join(d.reasons) or "all 9 checks passed",
                        "sequence": str(d.sequence),
                        "expires": d.expires_at.isoformat(),
                        "signed": "yes (Ed25519)" if d.signature else "no",
                    },
                )
            )
        elif isinstance(m, OrderPlaced | OrderModified):
            o = m.order
            st = o.status.value
            if isinstance(m, OrderPlaced):
                title, j.status = f"order sent to the broker: {o.side} {o.lots} {o.symbol}", "sent"
            elif o.lots == 0:
                title = "position closed at the broker"
            elif st == "placed":
                title, j.status = "pending order resting at the broker", "pending"
            elif st == "filled":
                title, j.status = "position open, stop confirmed at the broker", "open"
            else:
                title, j.status = f"order {st}", st
            j.steps.append(
                Step(
                    m.at,
                    "order",
                    title,
                    {
                        "client order id": o.client_order_id,
                        "broker order id": _s(o.broker_order_id),
                        "type": o.entry_type,
                        "status": st,
                        "lots": str(o.lots),
                        "price": _s(o.price),
                        "stop": str(o.sl),
                        "target": _s(o.tp),
                    },
                )
            )
        elif isinstance(m, OrderFilled):
            f = m.fill
            slip = None
            if f.requested_price is not None:
                slip = float(f.price - f.requested_price if f.side == "buy" else f.requested_price - f.price)
            model = self.slippage_mult_by_symbol.get(f.symbol, self.slippage_mult) * float(f.spread_at_fill)
            self.fills.append(
                ExecRow(
                    f.filled_at,
                    f.client_order_id,
                    f.symbol,
                    f.side,
                    None if f.requested_price is None else float(f.requested_price),
                    float(f.price),
                    slip,
                    model,
                    float(f.spread_at_fill),
                    f.latency_ms,
                )
            )
            j.steps.append(
                Step(
                    m.at,
                    "fill",
                    f"filled {f.lots} at {f.price}",
                    {
                        "requested": _s(f.requested_price),
                        "slippage": "–"
                        if slip is None
                        else f"{slip:+g} ({'worse' if slip > 0 else 'no worse'})",
                        "cost model expects": f"{model:.5g}",
                        "spread at fill": str(f.spread_at_fill),
                        "commission": str(f.commission),
                        "latency": f"{f.latency_ms:.0f} ms",
                    },
                )
            )
        elif isinstance(m, OrderCancelled):
            j.status = "cancelled"
            j.steps.append(Step(m.at, "order", f"order {m.reason}", {"reason": m.reason}))
        elif isinstance(m, PositionClosed):
            t = m.trade
            j.status, j.r, j.pnl = "closed", t.r_multiple, float(t.pnl_net)
            held = t.exit_time - t.entry_time
            j.steps.append(
                Step(
                    m.at,
                    "trade",
                    f"closed at {t.exit_price}: {t.r_multiple:+.2f}R",
                    {
                        "entry": str(t.entry_price),
                        "exit": str(t.exit_price),
                        "initial stop": str(t.stop_price),
                        "money at risk": f"{t.money_at_risk:.2f}",
                        "gross P&L": f"{t.pnl_gross:+.2f}",
                        "costs": f"{t.costs:.2f}",
                        "net P&L": f"{t.pnl_net:+.2f}",
                        "R multiple": f"{t.r_multiple:+.2f} (net P&L / money at risk)",
                        "held": f"{held.total_seconds() / 3600:.1f} h",
                    },
                )
            )

    def summaries(self, limit: int) -> list[Journey]:
        return list(self.by_ref.values())[-limit:][::-1]


def journey_json(j: Journey, steps: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ref": j.ref,
        "strategy": j.strategy,
        "symbol": j.symbol,
        "side": j.side,
        "status": j.status,
        "started": j.started.isoformat() if j.started else None,
        "updated": j.updated.isoformat() if j.updated else None,
        "r": None if j.r is None else round(j.r, 3),
        "pnl": j.pnl,
        "n_steps": len(j.steps),
    }
    if steps:
        out["steps"] = [
            {"at": s.at.isoformat(), "step": s.step, "title": s.title, "details": s.details} for s in j.steps
        ]
    return out
