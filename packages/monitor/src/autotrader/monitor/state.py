"""The monitor's view of the system (fed by the bus) and the daily summary (spec section 16).

Halts become critical alerts here; alerts raised in other processes arrive as AlertRaised messages.
The API reads this state; it never talks to the broker or the risk gate directly.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from autotrader.core import bus as streams
from autotrader.core.alerts import Alert, Severity
from autotrader.core.bus import Handler
from autotrader.core.clock import Clock
from autotrader.core.events import (
    AccountUpdate,
    AlertRaised,
    DemotionOrder,
    HaltCleared,
    HaltEntered,
    Heartbeat,
    OrderFilled,
    PositionClosed,
    QuoteUpdate,
    RiskDecided,
    SignalEmitted,
    StageChanged,
    StageSnapshot,
)
from autotrader.core.fx import rate_from_quotes
from autotrader.core.models import Fill, HaltState, RiskDecision, Stage, Trade
from autotrader.monitor.alerts import AlertRouter
from autotrader.monitor.candles import Candles
from autotrader.monitor.feed import Feed
from autotrader.monitor.journey import Journeys

EQUITY_POINTS = 2000
PRICE_POINTS = 7500  # per symbol, one per minute: about a trading week


class AccountSnapshot(BaseModel):
    """One row of `account_snapshots` (migration 0006): at most one per account per minute."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    account_id: str
    at: datetime
    currency: str
    balance: Decimal
    equity: Decimal
    peak_equity: Decimal
    open_risk: Decimal | None
    positions: int


@dataclass
class SlippageWatch:
    """Warns when real slippage runs above the backtest cost model (spec section 16).

    The model charges `mult x median spread`, always adverse (engine.costs); the spread at the fill
    stands in for the median. The mean adverse slippage of a symbol's last `window` fills is compared
    with the mean modelled slippage of the same fills. One warning per breach; it re-arms once the
    mean is back under the model. L8 (Phase 9) recalibrates the model; this is the early signal."""

    mult: float = 0.2
    mult_by_symbol: Mapping[str, float] = field(default_factory=dict)
    window: int = 20
    fills: dict[str, deque[tuple[float, float]]] = field(default_factory=dict)  # (real, modelled)
    breached: set[str] = field(default_factory=set)

    def add(self, f: Fill) -> Alert | None:
        """Returns a warning when this fill tips the symbol's rolling mean above the model."""
        if f.requested_price is None:
            return None
        adverse = f.price - f.requested_price if f.side == "buy" else f.requested_price - f.price
        modelled = self.mult_by_symbol.get(f.symbol, self.mult) * float(f.spread_at_fill)
        q = self.fills.setdefault(f.symbol, deque(maxlen=self.window))
        q.append((float(adverse), modelled))
        if len(q) < self.window:
            return None
        real = sum(a for a, _ in q) / len(q)
        model = sum(m for _, m in q) / len(q)
        if real <= model:
            self.breached.discard(f.symbol)
            return None
        if f.symbol in self.breached:
            return None
        self.breached.add(f.symbol)
        return Alert(
            severity=Severity.WARNING,
            kind="slippage_above_model",
            message=f"{f.symbol}: mean slippage {real:.5g} over the last {len(q)} fills, model {model:.5g}",
            at=f.filled_at,
            details={"symbol": f.symbol},
        )


@dataclass
class MonitorState:
    account: AccountUpdate | None = None
    equity_curve: deque[tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=EQUITY_POINTS))
    peak_equity: Decimal | None = None
    day_start: tuple[datetime, Decimal] | None = None
    month_start: tuple[datetime, Decimal] | None = None
    trades: deque[Trade] = field(default_factory=lambda: deque(maxlen=1000))
    shadow_trades: deque[Trade] = field(default_factory=lambda: deque(maxlen=1000))
    decisions: deque[RiskDecision] = field(default_factory=lambda: deque(maxlen=500))
    verdicts: Counter[str] = field(default_factory=Counter)
    signals: Counter[str] = field(default_factory=Counter)  # "money" / "shadow"
    halt: HaltState = HaltState.NORMAL
    halt_reason: str = ""
    halts_today: list[str] = field(default_factory=list)
    heartbeats: dict[str, datetime] = field(default_factory=dict)
    stages: dict[tuple[str, str], Stage] = field(default_factory=dict)
    demotions: deque[DemotionOrder] = field(default_factory=lambda: deque(maxlen=100))
    feed: Feed = field(default_factory=Feed)
    journeys: Journeys = field(default_factory=Journeys)
    candles: Candles = field(default_factory=Candles)
    prices: dict[str, deque[tuple[datetime, float, float]]] = field(default_factory=dict)  # (at, bid, ask)

    def open_risk(self, contract_size: Mapping[str, Decimal], quote_ccy: Mapping[str, str]) -> Decimal | None:
        """Money at risk to the stops of the system's open positions, account currency.
        None when a position has no stop or a rate is missing (unknown is never shown as zero)."""
        acct = self.account
        if acct is None:
            return Decimal(0)
        quotes = {q.symbol: q for q in acct.quotes}
        total = Decimal(0)
        for e in acct.exposures:
            if e.pending or e.external:
                continue
            if e.stop is None or e.symbol not in contract_size:
                return None
            try:
                fx = rate_from_quotes(quotes, quote_ccy[e.symbol], acct.currency)
            except KeyError:
                return None
            per_unit = e.entry - e.stop if e.side == "buy" else e.stop - e.entry
            total += max(per_unit, Decimal(0)) * contract_size[e.symbol] * e.lots * fx
        return total


class MonitorService:
    name = "monitor"

    def __init__(
        self,
        router: AlertRouter,
        clock: Clock,
        state: MonitorState | None = None,
        slippage: SlippageWatch | None = None,
        snapshots: Callable[[AccountSnapshot], None] | None = None,
        contract_size: Mapping[str, Decimal] | None = None,
        quote_ccy: Mapping[str, str] | None = None,
    ) -> None:
        self.router = router
        self.clock = clock
        self.state = state or MonitorState()
        self.slippage = slippage or SlippageWatch()
        self.state.journeys.slippage_mult = self.slippage.mult
        self.state.journeys.slippage_mult_by_symbol = dict(self.slippage.mult_by_symbol)
        self.snapshots = snapshots
        self.contract_size = dict(contract_size or {})
        self.quote_ccy = dict(quote_ccy or {})

    def handlers(self) -> Mapping[str, Handler]:
        own: dict[str, Handler] = {
            streams.ACCOUNT: self._account,
            streams.SIGNALS: self._signal,
            streams.DECISIONS: self._decision,
            streams.HALTS: self._halt,
            streams.TRADES: self._trade,
            streams.ORDERS: self._order,
            streams.STAGES: self._stage,
            streams.HEARTBEATS: self._heartbeat,
            streams.ALERTS: self._alert,
            streams.CONTROL: self._control,
            streams.INTENTS: self._nothing,
            streams.REQUESTS: self._nothing,
            streams.CONFIG: self._nothing,
        }
        return {stream: self._fed(h) for stream, h in own.items()} | {streams.QUOTES: self._quote}

    async def _quote(self, m: Any) -> None:
        if isinstance(m, QuoteUpdate):
            self.state.candles.quote(m.symbol, m.at, m.bid, m.ask)

    def _fed(self, h: Handler) -> Handler:
        async def run(m: Any) -> None:
            self.state.feed.add(m)
            self.state.journeys.add(m)
            await h(m)

        return run

    async def _nothing(self, m: Any) -> None:
        pass

    async def _account(self, m: Any) -> None:
        if not isinstance(m, AccountUpdate):
            return
        s = self.state
        s.account = m
        s.peak_equity = max(s.peak_equity or m.equity, m.equity)
        for q in m.quotes:
            hist = s.prices.setdefault(q.symbol, deque(maxlen=PRICE_POINTS))
            if not hist or m.at - hist[-1][0] >= timedelta(minutes=1):
                hist.append((m.at, float(q.bid), float(q.ask)))
        if not s.equity_curve or m.at - s.equity_curve[-1][0] >= timedelta(minutes=1):
            s.equity_curve.append((m.at, float(m.equity)))
            if self.snapshots is not None:
                self._snapshot(self.snapshots, m, s.peak_equity)
        if s.day_start is None:
            s.day_start = (m.at, m.equity)
        if s.month_start is None or s.month_start[0].month != m.at.month:
            s.month_start = (m.at, m.equity)

    def _snapshot(self, write: Callable[[AccountSnapshot], None], m: AccountUpdate, peak: Decimal) -> None:
        snap = AccountSnapshot(
            tenant_id=m.tenant_id,
            account_id=m.account_id,
            at=m.at,
            currency=m.currency,
            balance=m.balance,
            equity=m.equity,
            peak_equity=peak,
            open_risk=self.state.open_risk(self.contract_size, self.quote_ccy),
            positions=sum(1 for e in m.exposures if not e.pending),
        )
        try:
            write(snap)
        except Exception as e:  # a dashboard row is never worth stopping the monitor
            self.router.send(
                Alert(severity=Severity.WARNING, kind="snapshot_write_failed", message=str(e), at=m.at)
            )

    async def _signal(self, m: Any) -> None:
        if isinstance(m, SignalEmitted):
            self.state.signals["shadow" if m.shadow else "money"] += 1

    async def _decision(self, m: Any) -> None:
        if isinstance(m, RiskDecided):
            self.state.decisions.append(m.decision)
            self.state.verdicts[m.decision.verdict] += 1

    async def _halt(self, m: Any) -> None:
        s = self.state
        if isinstance(m, HaltEntered):
            s.halt, s.halt_reason = m.state, m.reason
            s.halts_today.append(f"{m.state.value}: {m.reason}")
            self.router.send(
                Alert(
                    severity=Severity.CRITICAL, kind="halt", message=f"{m.state.value}: {m.reason}", at=m.at
                )
            )
        elif isinstance(m, HaltCleared):
            s.halt, s.halt_reason = HaltState.NORMAL, ""
            self.router.send(
                Alert(
                    severity=Severity.INFO,
                    kind="halt_cleared",
                    message=f"{m.previous.value} cleared by {m.actor}",
                    at=m.at,
                )
            )

    async def _trade(self, m: Any) -> None:
        if isinstance(m, PositionClosed):
            (self.state.shadow_trades if m.trade.account_id == "shadow" else self.state.trades).append(
                m.trade
            )

    async def _order(self, m: Any) -> None:
        if isinstance(m, OrderFilled):
            a = self.slippage.add(m.fill)
            if a is not None:
                self.router.send(a)

    async def _stage(self, m: Any) -> None:
        if isinstance(m, StageSnapshot):
            self.state.stages = {(a, b): st for a, b, st in m.stages}
        elif isinstance(m, StageChanged):
            self.state.stages[(m.strategy_id, m.strategy_version)] = m.to_stage

    async def _heartbeat(self, m: Any) -> None:
        if isinstance(m, Heartbeat):
            self.state.heartbeats[m.service] = m.at

    async def _alert(self, m: Any) -> None:
        if isinstance(m, AlertRaised):
            self.router.send(m.alert)

    async def _control(self, m: Any) -> None:
        if isinstance(m, DemotionOrder):
            self.state.demotions.append(m)

    # ------------------------------------------------------------ daily summary

    def daily_summary(self, now: datetime) -> str:
        """After the 17:00 New York close: the day in a few lines, sent as an info alert; the day resets."""
        text = self.summary_text(now)
        self.router.send(Alert(severity=Severity.INFO, kind="daily_summary", message=text, at=now))
        s = self.state
        if s.account is not None:
            s.day_start = (now, s.account.equity)
        s.halts_today = []
        s.verdicts.clear()
        return text

    def summary_text(self, now: datetime) -> str:
        s = self.state
        acct = s.account
        lines = [f"Daily summary {now:%Y-%m-%d}"]
        if acct is None:
            lines.append("no account data")
        else:
            eq = acct.equity
            day0 = s.day_start[1] if s.day_start else eq
            month0 = s.month_start[1] if s.month_start else eq
            peak = s.peak_equity or eq
            lines += [
                f"equity {eq:,.2f} {acct.currency}",
                f"day P&L {eq - day0:+,.2f}  month P&L {eq - month0:+,.2f}",
                f"drawdown from peak {float((peak - eq) / peak) if peak else 0.0:.2%}",
                f"open positions {sum(1 for e in acct.exposures if not e.pending)}",
            ]
        since = now - timedelta(days=1)
        today = [t for t in s.trades if t.exit_time >= since]
        by_version: dict[str, list[Trade]] = {}
        for t in today:
            by_version.setdefault(f"{t.strategy_id} {t.strategy_version}", []).append(t)
        for v, ts in sorted(by_version.items()):
            pnl = sum((t.pnl_net for t in ts), Decimal(0))
            r = sum(t.r_multiple for t in ts)
            lines.append(f"  {v}: {len(ts)} trades, {pnl:+,.2f}, {r:+.2f}R")
        lines.append(
            f"decisions: {s.verdicts['approve']} approved, {s.verdicts['resize']} resized, "
            f"{s.verdicts['reject']} rejected"
        )
        lines.append(f"halts: {'; '.join(s.halts_today) or 'none'}")
        return "\n".join(lines)
