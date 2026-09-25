"""ShadowBroker and shadow sessions (spec sections 8 and 10): would-be fills from live quotes.

Shadow versions risk nothing. The ShadowBroker applies the SimBroker's rules to live quotes instead of
M1 bars, so shadow trades measure what the strategy would really have got:
- market entries and closes fill at the first quote after the request (ask for buys, bid for sells);
- limit entries fill when price trades through the limit by at least one tick, at the limit;
- stop entries fill at the triggering quote (gaps included);
- stops and targets are checked on every quote after the entry quote; stops fill at the quote (never
  better than the stop), targets at the target;
- commission per lot per side from the instrument, swap at each rollover (triple on its weekday).
Positions are sized with a nominal equity and risk fraction only so costs scale realistically; the
lifecycle evaluator reads R multiples, which do not depend on that size.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from autotrader.core.broker import Quote
from autotrader.core.clock import Clock
from autotrader.core.models import (
    CancelRequest,
    CloseRequest,
    ModifyStopRequest,
    Signal,
    Timeframe,
    Trade,
)
from autotrader.core.series import from_ns, to_ns
from autotrader.engine.costs import InstrumentCosts, StaticRates, rollover_times, swap_per_night
from autotrader.engine.gate import lots_for_risk
from autotrader.engine.live import LiveRunner
from autotrader.engine.live_bars import LiveBarBuilder
from autotrader.engine.simbroker import Rejection, TradeRecord
from autotrader.strategies_api.base import FillView, PendingView, PositionView, Request, Strategy
from autotrader.strategies_api.manifest import ParamValue


@dataclass
class _Pending:
    signal: Signal
    lots: float
    expires_ns: int | None


@dataclass
class _Position:
    signal: Signal
    lots: float
    entry: float
    entry_ns: int
    stop: float
    target: float | None
    money_at_risk: float
    commission: float
    spread_cost: float
    worst: float
    best: float
    swap: float = 0.0
    entry_quote_ns: int = 0
    close_requested: bool = False


@dataclass
class ShadowBroker:
    instruments: Mapping[str, InstrumentCosts]
    rates: StaticRates
    account_ccy: str = "USD"
    nominal_equity: float = 100_000.0
    risk_fraction: float = 0.005
    min_stop_spreads: float = 1.5
    quotes: dict[str, Quote] = field(default_factory=dict)
    pending: dict[str, _Pending] = field(default_factory=dict)
    positions: dict[str, _Position] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)

    def _fx(self, symbol: str) -> float:
        return self.rates.rate(self.instruments[symbol].quote, self.account_ccy)

    # ------------------------------------------------------------ requests

    def submit(self, s: Signal, tf: Timeframe) -> None:
        q = self.quotes.get(s.symbol)
        now_ns = to_ns(s.created_at)
        if q is None:
            self.rejections.append(Rejection(now_ns, f"signal {s.signal_id}", "no price yet"))
            return
        bid, ask = float(q.bid), float(q.ask)
        ref = (ask if s.side == "buy" else bid) if s.entry_type == "market" else float(s.entry_price or 0.0)
        wrong = (s.side == "buy" and s.stop_price >= ref) or (s.side == "sell" and s.stop_price <= ref)
        if wrong or abs(ref - s.stop_price) < self.min_stop_spreads * (ask - bid):
            self.rejections.append(Rejection(now_ns, f"signal {s.signal_id}", "stop invalid or too close"))
            return
        c = self.instruments[s.symbol]
        lots = lots_for_risk(
            self.nominal_equity, self.risk_fraction, ref, s.stop_price, c, self._fx(s.symbol)
        )
        if lots <= 0:
            self.rejections.append(Rejection(now_ns, f"signal {s.signal_id}", "size below min lot"))
            return
        expires = None
        if s.entry_type != "market" and s.expiry_bars:
            expires = now_ns + int(timedelta(minutes=s.expiry_bars * tf.minutes).total_seconds() * 1e9)
        self.pending[str(s.signal_id)] = _Pending(s, lots, expires)

    def handle(self, r: Request, tf: Timeframe) -> None:
        if isinstance(r, Signal):
            self.submit(r, tf)
        elif isinstance(r, CancelRequest):
            self.pending.pop(str(r.signal_id), None)
        elif isinstance(r, CloseRequest):
            p = self.positions.get(r.position_id)
            if p is not None and p.signal.strategy_id == r.strategy_id:
                p.close_requested = True
        elif isinstance(r, ModifyStopRequest):
            p = self.positions.get(r.position_id)
            if p is not None and p.signal.strategy_id == r.strategy_id:
                tighter = r.new_stop > p.stop if p.signal.side == "buy" else r.new_stop < p.stop
                if tighter:
                    p.stop = r.new_stop

    # ------------------------------------------------------------ quotes

    def on_quote(self, q: Quote) -> list[FillView]:
        self.quotes[q.symbol] = q
        t = to_ns(q.time)
        bid, ask = float(q.bid), float(q.ask)
        tick = self.instruments[q.symbol].tick_size
        fills: list[FillView] = []
        for sid, o in list(self.pending.items()):
            s = o.signal
            if s.symbol != q.symbol or t <= to_ns(s.created_at):
                continue
            if o.expires_ns is not None and t >= o.expires_ns:
                del self.pending[sid]
                continue
            px: float | None = None
            if s.entry_type == "market":
                px = ask if s.side == "buy" else bid
            elif s.entry_type == "limit":
                lim = float(s.entry_price or 0.0)
                if (s.side == "buy" and ask <= lim - tick) or (s.side == "sell" and bid >= lim + tick):
                    px = lim
            else:
                lvl = float(s.entry_price or 0.0)
                if s.side == "buy" and ask >= lvl:
                    px = ask
                elif s.side == "sell" and bid <= lvl:
                    px = bid
            if px is not None:
                del self.pending[sid]
                fill = self._open(o, px, t, ask - bid)
                if fill is not None:
                    fills.append(fill)
        for pid, p in list(self.positions.items()):
            if p.signal.symbol != q.symbol or t <= p.entry_quote_ns:
                continue
            buy = p.signal.side == "buy"
            mark = bid if buy else ask
            p.worst = min(p.worst, mark) if buy else max(p.worst, mark)
            p.best = max(p.best, mark) if buy else min(p.best, mark)
            if p.close_requested:
                fills.append(self._close(pid, p, mark, t, "close_request", ask - bid))
            elif (buy and bid <= p.stop) or (not buy and ask >= p.stop):
                fills.append(self._close(pid, p, mark, t, "stop", ask - bid))
            elif p.target is not None and ((buy and bid >= p.target) or (not buy and ask <= p.target)):
                fills.append(self._close(pid, p, p.target, t, "target", ask - bid))
        return fills

    def _open(self, o: _Pending, px: float, t: int, spread: float) -> FillView | None:
        s = o.signal
        c = self.instruments[s.symbol]
        rpu = abs(px - s.stop_price)
        if (s.side == "buy" and s.stop_price >= px) or (s.side == "sell" and s.stop_price <= px) or rpu == 0:
            self.rejections.append(Rejection(t, f"entry {s.signal_id}", "fill price beyond stop"))
            return None
        fx = self._fx(s.symbol)
        pid = str(s.signal_id)
        self.positions[pid] = _Position(
            signal=s,
            lots=o.lots,
            entry=px,
            entry_ns=t,
            stop=s.stop_price,
            target=s.target_price,
            money_at_risk=rpu * c.contract_size * o.lots * fx,
            commission=c.commission_per_lot_side * o.lots,
            spread_cost=spread * c.contract_size * o.lots * fx if s.side == "buy" else 0.0,
            worst=px,
            best=px,
            entry_quote_ns=t,
        )
        return FillView(pid, pid, s.symbol, s.side, "entry", px, o.lots, from_ns(t))

    def _close(self, pid: str, p: _Position, px: float, t: int, reason: str, spread: float) -> FillView:
        del self.positions[pid]
        s = p.signal
        c = self.instruments[s.symbol]
        fx = self._fx(s.symbol)
        sign = 1.0 if s.side == "buy" else -1.0
        gross = sign * (px - p.entry) * c.contract_size * p.lots * fx
        commission = p.commission + c.commission_per_lot_side * p.lots
        net = gross - commission + p.swap
        rpu = abs(p.entry - s.stop_price)
        self.trades.append(
            TradeRecord(
                trade_id=pid,
                signal_id=pid,
                strategy_id=s.strategy_id,
                strategy_version=s.strategy_version,
                symbol=s.symbol,
                side=s.side,
                lots=p.lots,
                entry_time_ns=p.entry_ns,
                entry_price=p.entry,
                stop_price=s.stop_price,
                exit_time_ns=t,
                exit_price=px,
                exit_reason=reason,
                pnl_gross=gross,
                commission=commission,
                swap=p.swap,
                spread_cost=p.spread_cost
                + (spread * c.contract_size * p.lots * fx if s.side == "sell" else 0.0),
                slippage_cost=0.0,
                pnl_net=net,
                money_at_risk=p.money_at_risk,
                r_multiple=net / p.money_at_risk,
                mae_r=max(0.0, sign * (p.entry - p.worst) / rpu),
                mfe_r=max(0.0, sign * (p.best - p.entry) / rpu),
                bars_held=max(1, (t - p.entry_ns) // 60_000_000_000),
                tags=tuple(sorted(s.tags.items())),
            )
        )
        return FillView(pid, pid, s.symbol, s.side, "exit", px, p.lots, from_ns(t), reason)

    def charge_swaps(self, weekday: int) -> None:
        for p in self.positions.values():
            c = self.instruments[p.signal.symbol]
            q = self.quotes[p.signal.symbol]
            nights = 3 if weekday == c.triple_swap_weekday else 1
            p.swap += nights * swap_per_night(
                c, p.signal.side, p.lots, float(q.bid), self._fx(p.signal.symbol)
            )

    # ------------------------------------------------------------ strategy views

    def position_views(self, strategy_id: str, symbol: str | None) -> list[PositionView]:
        return [
            PositionView(
                pid,
                pid,
                p.signal.symbol,
                p.signal.side,
                p.lots,
                p.entry,
                p.stop,
                p.target,
                from_ns(p.entry_ns),
            )
            for pid, p in self.positions.items()
            if p.signal.strategy_id == strategy_id and (symbol is None or p.signal.symbol == symbol)
        ]

    def pending_views(self, strategy_id: str, symbol: str | None) -> list[PendingView]:
        return [
            PendingView(
                sid,
                o.signal.symbol,
                o.signal.side,
                o.signal.entry_type,
                o.signal.entry_price,
                o.signal.stop_price,
                o.signal.target_price,
                o.signal.created_at,
            )
            for sid, o in self.pending.items()
            if o.signal.strategy_id == strategy_id and (symbol is None or o.signal.symbol == symbol)
        ]


def to_trade(t: TradeRecord, account_id: str) -> Trade:
    """Engine trade record -> the core Trade model the lifecycle and journal read."""

    def d(x: float) -> Decimal:
        return Decimal(str(round(x, 8)))

    return Trade(
        trade_id=t.trade_id,
        account_id=account_id,
        strategy_id=t.strategy_id,
        strategy_version=t.strategy_version,
        symbol=t.symbol,
        side=t.side,
        lots=d(t.lots),
        entry_time=from_ns(t.entry_time_ns),
        entry_price=d(t.entry_price),
        stop_price=d(t.stop_price),
        exit_time=from_ns(t.exit_time_ns),
        exit_price=d(t.exit_price),
        pnl_gross=d(t.pnl_gross),
        costs=d(t.commission - t.swap + t.slippage_cost),
        pnl_net=d(t.pnl_net),
        money_at_risk=d(t.money_at_risk),
        r_multiple=t.r_multiple,
        mae=t.mae_r,
        mfe=t.mfe_r,
    )


class ShadowSession:
    """One strategy version in shadow: quotes -> bars -> strategy -> ShadowBroker -> trades."""

    def __init__(
        self,
        strategy_cls: type[Strategy],
        instruments: Mapping[str, InstrumentCosts],
        *,
        rates: StaticRates | None = None,
        account_ccy: str = "USD",
        params: Mapping[str, ParamValue] | None = None,
        on_signal: Callable[[Signal], None] | None = None,
        on_trade: Callable[[Trade], None] | None = None,
        grace: timedelta = timedelta(seconds=1),
    ) -> None:
        m = strategy_cls.manifest
        self.broker = ShadowBroker(
            {s: instruments[s] for s in m.symbols}, rates or StaticRates(), account_ccy
        )
        self.runner = LiveRunner(
            strategy_cls,
            spread_fn=self._spread,
            positions_fn=self.broker.position_views,
            pending_fn=self.broker.pending_views,
            params=params,
        )
        self.builder = LiveBarBuilder(self.runner.subs, grace=grace)
        self.tf = min(m.timeframes, key=lambda t: t.minutes)
        self.on_signal = on_signal
        self.on_trade = on_trade
        self.signals: list[Signal] = []
        self._last_ns: int | None = None
        self._n_trades = 0

    def _spread(self, symbol: str, _ns: int) -> float:
        q = self.broker.quotes.get(symbol)
        return float(q.ask - q.bid) if q is not None else 0.0

    def _route(self, reqs: list[Request]) -> None:
        for r in reqs:
            if isinstance(r, Signal):
                self.signals.append(r)
                if self.on_signal is not None:
                    self.on_signal(r)
            self.broker.handle(r, self.tf)

    def _rollovers(self, now: datetime) -> None:
        if self._last_ns is not None:
            for _, weekday in rollover_times(from_ns(self._last_ns + 1), now):
                self.broker.charge_swaps(weekday)
        self._last_ns = to_ns(now)

    def on_time(self, now: datetime) -> None:
        self._route(self.runner.on_bars(self.builder.on_time(now)))
        self._rollovers(now)
        self._emit_trades()

    def on_quote(self, q: Quote) -> None:
        self._route(self.runner.on_bars(self.builder.on_quote(q)))
        self._rollovers(q.time)
        for fill in self.broker.on_quote(q):
            self._route(self.runner.on_fill(fill))
        self._emit_trades()

    def _emit_trades(self) -> None:
        new = self.broker.trades[self._n_trades :]
        self._n_trades = len(self.broker.trades)
        if self.on_trade is not None:
            for t in new:
                self.on_trade(to_trade(t, account_id="shadow"))


async def run_shadow(
    quotes: AsyncIterable[Quote],
    sessions: Sequence[ShadowSession],
    clock: Clock,
    *,
    tick_seconds: float = 1.0,
    stop: asyncio.Event | None = None,
) -> None:
    """engine-live, shadow part: feed every quote to every session; a ticker closes bars on time
    even when a market goes quiet. Stops when the quote stream ends or `stop` is set."""
    stop = stop or asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            await asyncio.sleep(tick_seconds)
            for s in sessions:
                s.on_time(clock.now())

    task = asyncio.create_task(ticker())
    try:
        async for q in quotes:
            for s in sessions:
                s.on_quote(q)
            if stop.is_set():
                break
    finally:
        stop.set()
        task.cancel()
