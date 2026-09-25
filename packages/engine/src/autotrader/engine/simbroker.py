"""Fill simulator at M1 resolution (spec section 8).

Rules
- Market orders (entries and closes) fill at the next M1 bar open: ask for buys,
  bid for sells, plus adverse slippage.
- Limit entries fill only when price trades through the limit by >= 1 tick, at
  the limit price.
- Stop entries fill at the stop plus slippage, or at the open plus slippage if
  the bar gaps through.
- Stop loss and take profit are checked on every M1 bar (long on bid, short on
  ask). If both are inside the same M1 bar, the stop is assumed hit first.
- A pending entry filled inside a bar may be stopped out on that same bar, but
  may NOT take profit on it (the order of events inside the bar is unknown).
- Swap at each Mon-Fri rollover, triple on the instrument's weekday.
- Commission is charged half at entry and half at exit.
- No partial fills (docs/decisions.md). Hedging account: every signal opens its
  own position.

The broker jumps from event to event: for each open item it finds the first M1
bar in the remaining interval that triggers it with numpy, processes the
earliest, and repeats. Idle stretches cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from autotrader.core.models import Side, Signal
from autotrader.core.series import NS_PER_MINUTE, BarsArray, from_ns
from autotrader.engine.costs import (
    CostModel,
    InstrumentCosts,
    StaticRates,
    hour_of_week_array,
    swap_per_night,
)
from autotrader.strategies_api.base import FillView, PendingView, PositionView

F64 = npt.NDArray[np.float64]
ExitReason = Literal["stop", "target", "close_request", "end_of_data", "forced"]


@dataclass
class SymbolData:
    costs: InstrumentCosts
    bars: BarsArray
    spread: F64  # modelled spread per M1 bar
    slip: F64  # slippage per M1 bar (price units)
    ask_o: F64
    ask_h: F64
    ask_l: F64
    ask_c: F64
    to_account: float  # quote currency -> account currency


@dataclass
class PendingOrder:
    signal: Signal
    lots: float
    active_from_ns: int
    expires_ns: int | None
    seq: int


@dataclass
class MarketOrder:
    kind: Literal["entry", "close"]
    seq: int
    signal: Signal | None = None
    lots: float = 0.0
    position_id: str | None = None
    reason: ExitReason = "close_request"


@dataclass
class OpenPosition:
    position_id: str
    signal: Signal
    side: Side
    lots: float
    entry_price: float
    entry_idx: int
    entry_ns: int
    initial_stop: float
    stop: float
    target: float | None
    money_at_risk: float
    commission: float
    swap: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    seq: int = 0


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    signal_id: str
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Side
    lots: float
    entry_time_ns: int
    entry_price: float
    stop_price: float
    exit_time_ns: int
    exit_price: float
    exit_reason: str
    pnl_gross: float
    commission: float
    swap: float
    spread_cost: float
    slippage_cost: float
    pnl_net: float
    money_at_risk: float
    r_multiple: float
    mae_r: float
    mfe_r: float
    bars_held: int
    tags: tuple[tuple[str, str], ...] = ()


@dataclass
class Rejection:
    time_ns: int
    what: str
    reason: str


@dataclass
class SimBroker:
    symbols: dict[str, SymbolData]
    balance: float
    pending: dict[str, list[PendingOrder]] = field(default_factory=dict)
    market: dict[str, list[MarketOrder]] = field(default_factory=dict)
    positions: dict[str, list[OpenPosition]] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    fills_out: list[tuple[str, FillView]] = field(default_factory=list)  # (strategy_id, fill)
    rejections: list[Rejection] = field(default_factory=list)
    last_idx: dict[str, int] = field(default_factory=dict)
    _seq: int = 0

    @staticmethod
    def build(
        bars: dict[str, BarsArray],
        instruments: dict[str, InstrumentCosts],
        cost_model: CostModel,
        rates: StaticRates,
        account_ccy: str,
        balance: float,
    ) -> SimBroker:
        data: dict[str, SymbolData] = {}
        for sym, b in bars.items():
            inst = instruments[sym]
            spr = cost_model.spreads.spread_array(sym, b.open_time)
            med = np.asarray(cost_model.spreads.median_by_hour[sym], dtype=np.float64)
            mult = cost_model.slippage_mult_by_symbol.get(sym, cost_model.slippage_mult)
            slip = mult * med[hour_of_week_array(b.open_time)]
            data[sym] = SymbolData(
                costs=inst,
                bars=b,
                spread=spr,
                slip=slip,
                ask_o=b.bid_o + spr,
                ask_h=b.bid_h + spr,
                ask_l=b.bid_l + spr,
                ask_c=b.bid_c + spr,
                to_account=rates.rate(inst.quote, account_ccy),
            )
        return SimBroker(symbols=data, balance=balance)

    # ---- order intake (called by the engine at time T, after risk approval) ----

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def submit_entry(self, signal: Signal, lots: float, now_ns: int, tf_minutes: int) -> None:
        sym = signal.symbol
        if signal.entry_type == "market":
            self.market.setdefault(sym, []).append(
                MarketOrder("entry", self._next_seq(), signal=signal, lots=lots)
            )
            return
        expires = (
            None if signal.expiry_bars is None else now_ns + signal.expiry_bars * tf_minutes * NS_PER_MINUTE
        )
        self.pending.setdefault(sym, []).append(PendingOrder(signal, lots, now_ns, expires, self._next_seq()))

    def cancel(self, strategy_id: str, signal_id: str, now_ns: int) -> bool:
        for orders in self.pending.values():
            for o in orders:
                if str(o.signal.signal_id) == signal_id and o.signal.strategy_id == strategy_id:
                    orders.remove(o)
                    return True
        self.rejections.append(Rejection(now_ns, f"cancel {signal_id}", "no such pending order"))
        return False

    def _find_position(self, strategy_id: str, position_id: str) -> OpenPosition | None:
        for ps in self.positions.values():
            for p in ps:
                if p.position_id == position_id and p.signal.strategy_id == strategy_id:
                    return p
        return None

    def request_close(self, strategy_id: str, position_id: str, now_ns: int, reason: ExitReason) -> bool:
        p = self._find_position(strategy_id, position_id)
        if p is None:
            self.rejections.append(Rejection(now_ns, f"close {position_id}", "no such position"))
            return False
        orders = self.market.setdefault(p.signal.symbol, [])
        if not any(o.kind == "close" and o.position_id == position_id for o in orders):
            orders.append(MarketOrder("close", self._next_seq(), position_id=position_id, reason=reason))
        return True

    def modify_stop(self, strategy_id: str, position_id: str, new_stop: float, now_ns: int) -> bool:
        p = self._find_position(strategy_id, position_id)
        if p is None:
            self.rejections.append(Rejection(now_ns, f"modify {position_id}", "no such position"))
            return False
        d = self.symbols[p.signal.symbol]
        i = self.last_idx.get(p.signal.symbol, -1)
        if i < 0:
            self.rejections.append(Rejection(now_ns, f"modify {position_id}", "no price yet"))
            return False
        if p.side == "buy":
            ok = p.stop < new_stop < float(d.bars.bid_c[i])
        else:
            ok = float(d.ask_c[i]) < new_stop < p.stop
        if not ok:
            self.rejections.append(
                Rejection(now_ns, f"modify {position_id}", "only tightening stops is allowed")
            )
            return False
        p.stop = new_stop
        return True

    # ---- views ----

    def position_views(self, strategy_id: str, symbol: str | None) -> list[PositionView]:
        out = []
        for sym, ps in self.positions.items():
            if symbol is not None and sym != symbol:
                continue
            for p in ps:
                if p.signal.strategy_id == strategy_id:
                    out.append(
                        PositionView(
                            p.position_id,
                            str(p.signal.signal_id),
                            sym,
                            p.side,
                            p.lots,
                            p.entry_price,
                            p.stop,
                            p.target,
                            from_ns(p.entry_ns),
                        )
                    )
        return out

    def pending_views(self, strategy_id: str, symbol: str | None) -> list[PendingView]:
        out = []
        for sym in sorted(set(self.pending) | set(self.market)):
            if symbol is not None and sym != symbol:
                continue
            sigs = [o.signal for o in self.pending.get(sym, [])]
            sigs += [o.signal for o in self.market.get(sym, []) if o.kind == "entry" and o.signal is not None]
            for s in sigs:
                if s.strategy_id == strategy_id:
                    out.append(
                        PendingView(
                            str(s.signal_id),
                            sym,
                            s.side,
                            s.entry_type,
                            s.entry_price,
                            s.stop_price,
                            s.target_price,
                            s.created_at,
                        )
                    )
        return out

    def has_activity(self, sym: str) -> bool:
        return bool(self.pending.get(sym) or self.market.get(sym) or self.positions.get(sym))

    # ---- simulation ----

    def advance(self, sym: str, lo: int, hi: int) -> None:
        """Process M1 bars [lo, hi) of `sym`."""
        if hi <= lo:
            return
        self.last_idx[sym] = hi - 1
        cur = lo
        d = self.symbols[sym]
        while cur < hi and self.has_activity(sym):
            i = self._earliest_event(d, sym, cur, hi)
            if i is None:
                return
            self._process_bar(d, sym, i)
            cur = i + 1

    def _first(self, mask: npt.NDArray[np.bool_], offset: int) -> int | None:
        k = int(np.argmax(mask)) if mask.size else 0
        return offset + k if mask.size and mask[k] else None

    def _pending_trigger_mask(
        self, d: SymbolData, o: PendingOrder, lo: int, hi: int
    ) -> npt.NDArray[np.bool_]:
        s, tick = o.signal, d.costs.tick_size
        px = float(s.entry_price or 0.0)
        if s.entry_type == "limit":
            m = d.ask_l[lo:hi] <= px - tick if s.side == "buy" else d.bars.bid_h[lo:hi] >= px + tick
        else:
            m = d.ask_h[lo:hi] >= px if s.side == "buy" else d.bars.bid_l[lo:hi] <= px
        m = m & (d.bars.open_time[lo:hi] >= o.active_from_ns)
        return np.asarray(m, dtype=np.bool_)

    def _exit_mask(self, d: SymbolData, p: OpenPosition, lo: int, hi: int) -> npt.NDArray[np.bool_]:
        if p.side == "buy":
            m = d.bars.bid_l[lo:hi] <= p.stop
            if p.target is not None:
                m = m | (d.bars.bid_h[lo:hi] >= p.target)
        else:
            m = d.ask_h[lo:hi] >= p.stop
            if p.target is not None:
                m = m | (d.ask_l[lo:hi] <= p.target)
        return np.asarray(m, dtype=np.bool_)

    def _earliest_event(self, d: SymbolData, sym: str, lo: int, hi: int) -> int | None:
        best: int | None = lo if self.market.get(sym) else None
        ot = d.bars.open_time
        for o in self.pending.get(sym, []):
            if o.expires_ns is not None:
                e = int(np.searchsorted(ot[lo:hi], o.expires_ns)) + lo
                if e < hi and (best is None or e < best):
                    best = e
            f = self._first(self._pending_trigger_mask(d, o, lo, hi if best is None else best + 1), lo)
            if f is not None and (best is None or f < best):
                best = f
        for p in self.positions.get(sym, []):
            f = self._first(self._exit_mask(d, p, lo, hi if best is None else best + 1), lo)
            if f is not None and (best is None or f < best):
                best = f
        return best

    def _process_bar(self, d: SymbolData, sym: str, i: int) -> None:
        t_open = int(d.bars.open_time[i])
        # 1. market orders at the bar open
        for mo in sorted(self.market.pop(sym, []), key=lambda o: o.seq):
            if mo.kind == "entry" and mo.signal is not None:
                buy = mo.signal.side == "buy"
                slip = float(d.slip[i])
                px = float(d.ask_o[i]) + slip if buy else float(d.bars.bid_o[i]) - slip
                self._open(d, mo.signal, mo.lots, px, i, slip, float(d.spread[i]))
            elif mo.kind == "close" and mo.position_id is not None:
                p = next((q for q in self.positions.get(sym, []) if q.position_id == mo.position_id), None)
                if p is not None:
                    slip = float(d.slip[i])
                    px = float(d.bars.bid_o[i]) - slip if p.side == "buy" else float(d.ask_o[i]) + slip
                    self._close(d, p, px, i, mo.reason, slip)
        # 2. expiries (checked before triggers on the expiry bar)
        orders = self.pending.get(sym, [])
        for o in list(orders):
            if o.expires_ns is not None and t_open >= o.expires_ns:
                orders.remove(o)
        # 3. exits of positions (including ones opened at this bar's open)
        for p in sorted(list(self.positions.get(sym, [])), key=lambda q: q.seq):
            self._check_exit(d, p, i, allow_target=True)
        # 4. pending triggers; new positions may be stopped out on this bar but not take profit
        for o in sorted(list(orders), key=lambda q: q.seq):
            if not self._pending_trigger_mask(d, o, i, i + 1)[0]:
                continue
            orders.remove(o)
            s = o.signal
            px_entry = float(s.entry_price or 0.0)
            slip = 0.0
            if s.entry_type == "stop":
                slip = float(d.slip[i])
                if s.side == "buy":
                    px = max(px_entry, float(d.ask_o[i])) + slip
                else:
                    px = min(px_entry, float(d.bars.bid_o[i])) - slip
            else:
                px = px_entry
            p = self._open(d, s, o.lots, px, i, slip, float(d.spread[i]))
            if p is not None:
                self._check_exit(d, p, i, allow_target=False)

    def _open(
        self, d: SymbolData, s: Signal, lots: float, px: float, i: int, slip: float, spread: float
    ) -> OpenPosition | None:
        c = d.costs
        risk_per_unit = abs(px - s.stop_price)
        wrong_side = (s.side == "buy" and s.stop_price >= px) or (s.side == "sell" and s.stop_price <= px)
        if wrong_side or risk_per_unit == 0.0:
            # the fill landed beyond the stop (gap); record and do not open
            self.rejections.append(
                Rejection(int(d.bars.open_time[i]), f"entry {s.signal_id}", "fill price beyond stop")
            )
            return None
        mar = risk_per_unit * c.contract_size * lots * d.to_account
        commission = c.commission_per_lot_side * lots
        self.balance -= commission
        p = OpenPosition(
            position_id=str(s.signal_id),
            signal=s,
            side=s.side,
            lots=lots,
            entry_price=px,
            entry_idx=i,
            entry_ns=int(d.bars.open_time[i]),
            initial_stop=s.stop_price,
            stop=s.stop_price,
            target=s.target_price,
            money_at_risk=mar,
            commission=commission,
            spread_cost=(spread * c.contract_size * lots * d.to_account) if s.side == "buy" else 0.0,
            slippage_cost=slip * c.contract_size * lots * d.to_account,
            seq=self._next_seq(),
        )
        self.positions.setdefault(s.symbol, []).append(p)
        self.fills_out.append(
            (
                s.strategy_id,
                FillView(
                    str(s.signal_id), p.position_id, s.symbol, s.side, "entry", px, lots, from_ns(p.entry_ns)
                ),
            )
        )
        return p

    def _check_exit(self, d: SymbolData, p: OpenPosition, i: int, *, allow_target: bool) -> None:
        if p not in self.positions.get(p.signal.symbol, []):
            return
        slip = float(d.slip[i])
        if p.side == "buy":
            if float(d.bars.bid_l[i]) <= p.stop:
                gap = float(d.bars.bid_o[i]) <= p.stop and i != p.entry_idx
                px = (float(d.bars.bid_o[i]) if gap else p.stop) - slip
                self._close(d, p, px, i, "stop", slip)
            elif allow_target and p.target is not None and float(d.bars.bid_h[i]) >= p.target:
                self._close(d, p, p.target, i, "target", 0.0)
        elif float(d.ask_h[i]) >= p.stop:
            gap = float(d.ask_o[i]) >= p.stop and i != p.entry_idx
            px = (float(d.ask_o[i]) if gap else p.stop) + slip
            self._close(d, p, px, i, "stop", slip)
        elif allow_target and p.target is not None and float(d.ask_l[i]) <= p.target:
            self._close(d, p, p.target, i, "target", 0.0)

    def _close(
        self, d: SymbolData, p: OpenPosition, px: float, i: int, reason: ExitReason, slip: float
    ) -> None:
        c = d.costs
        self.positions[p.signal.symbol].remove(p)
        sign = 1.0 if p.side == "buy" else -1.0
        gross = sign * (px - p.entry_price) * c.contract_size * p.lots * d.to_account
        exit_commission = c.commission_per_lot_side * p.lots
        self.balance += gross - exit_commission
        commission = p.commission + exit_commission
        net = gross - commission + p.swap
        spread_cost = p.spread_cost + (
            float(d.spread[i]) * c.contract_size * p.lots * d.to_account if p.side == "sell" else 0.0
        )
        slippage_cost = p.slippage_cost + slip * c.contract_size * p.lots * d.to_account
        # excursions over the holding bars, in R
        lo, hi = p.entry_idx, i + 1
        rpu = abs(p.entry_price - p.initial_stop)
        if p.side == "buy":
            mae = (p.entry_price - float(d.bars.bid_l[lo:hi].min())) / rpu
            mfe = (float(d.bars.bid_h[lo:hi].max()) - p.entry_price) / rpu
        else:
            mae = (float(d.ask_h[lo:hi].max()) - p.entry_price) / rpu
            mfe = (p.entry_price - float(d.ask_l[lo:hi].min())) / rpu
        exit_ns = int(d.bars.open_time[i]) if reason == "close_request" else int(d.bars.close_time[i])
        s = p.signal
        self.trades.append(
            TradeRecord(
                trade_id=p.position_id,
                signal_id=str(s.signal_id),
                strategy_id=s.strategy_id,
                strategy_version=s.strategy_version,
                symbol=s.symbol,
                side=p.side,
                lots=p.lots,
                entry_time_ns=p.entry_ns,
                entry_price=p.entry_price,
                stop_price=p.initial_stop,
                exit_time_ns=exit_ns,
                exit_price=px,
                exit_reason=reason,
                pnl_gross=gross,
                commission=commission,
                swap=p.swap,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                pnl_net=net,
                money_at_risk=p.money_at_risk,
                r_multiple=net / p.money_at_risk,
                mae_r=max(0.0, mae),
                mfe_r=max(0.0, mfe),
                bars_held=i - p.entry_idx + 1,
                tags=tuple(sorted(s.tags.items())),
            )
        )
        self.fills_out.append(
            (
                s.strategy_id,
                FillView(
                    str(s.signal_id),
                    p.position_id,
                    s.symbol,
                    p.side,
                    "exit",
                    px,
                    p.lots,
                    from_ns(exit_ns),
                    reason,
                ),
            )
        )

    # ---- account ----

    def charge_swaps(self, weekday: int) -> None:
        for sym, ps in self.positions.items():
            d = self.symbols[sym]
            i = self.last_idx.get(sym, -1)
            if i < 0:
                continue
            nights = 3 if weekday == d.costs.triple_swap_weekday else 1
            for p in ps:
                amt = nights * swap_per_night(d.costs, p.side, p.lots, float(d.bars.bid_c[i]), d.to_account)
                p.swap += amt
                self.balance += amt

    def floating(self) -> float:
        total = 0.0
        for sym, ps in self.positions.items():
            d = self.symbols[sym]
            i = self.last_idx.get(sym, -1)
            if i < 0:
                continue
            for p in ps:
                if p.side == "buy":
                    total += (
                        (float(d.bars.bid_c[i]) - p.entry_price)
                        * d.costs.contract_size
                        * p.lots
                        * d.to_account
                    )
                else:
                    total += (
                        (p.entry_price - float(d.ask_c[i])) * d.costs.contract_size * p.lots * d.to_account
                    )
        return total

    def equity(self) -> float:
        return self.balance + self.floating()

    def close_all(self, reason: ExitReason) -> None:
        """Close everything at the last processed bar close (end of data, forced exits)."""
        for sym in sorted(self.positions):
            d = self.symbols[sym]
            i = self.last_idx.get(sym, -1)
            if i < 0:
                continue
            for p in sorted(list(self.positions[sym]), key=lambda q: q.seq):
                px = float(d.bars.bid_c[i]) if p.side == "buy" else float(d.ask_c[i])
                self._close(d, p, px, i, reason, 0.0)
        self.pending.clear()
        self.market.clear()
