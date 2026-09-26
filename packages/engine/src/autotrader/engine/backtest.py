"""Backtest event loop (spec section 8): SimClock + historical data + SimBroker.

Per distinct event time T (a bar close of any subscribed series or a rollover):
  1. the broker simulates every M1 bar that closed at or before T,
  2. fills are delivered to the strategy (`on_fill`),
  3. at rollovers swaps are charged and daily equity is recorded,
  4. every subscribed series bar closing at T becomes visible, then `on_bar`
     is called for each of them in a fixed order (symbol order of the
     manifest, then shorter timeframe first).
Orders created at T can only fill on M1 bars that open at or after T.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autotrader.core.clock import SimClock
from autotrader.core.events import BarClosed
from autotrader.core.hashing import hash_obj
from autotrader.core.models import Bar, CancelRequest, CloseRequest, ModifyStopRequest, Signal, Timeframe
from autotrader.core.series import BarsArray, from_ns
from autotrader.engine.context import EngineContext
from autotrader.engine.costs import CostModel, InstrumentCosts, StaticRates, rollover_times
from autotrader.engine.gate import BacktestGuard, GateRequest, OrderGate
from autotrader.engine.market import EngineMarketView, SeriesBuffer, news_from_times
from autotrader.engine.metrics import Metrics, compute
from autotrader.engine.requests import StrategyError, check_requests
from autotrader.engine.simbroker import Rejection, SimBroker, TradeRecord
from autotrader.strategies_api.base import Request, Strategy
from autotrader.strategies_api.manifest import ParamValue


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    account_ccy: str = "USD"
    risk_fraction: float = 0.005
    seed: int = 0
    open_risk_total_max: float = 0.03


@dataclass
class BacktestResult:
    strategy_id: str
    strategy_version: str
    params: dict[str, ParamValue]
    signals: list[Signal]
    trades: list[TradeRecord]
    equity_curve: list[tuple[int, float]]  # (ns, equity) at each rollover and at the end
    rejections: list[Rejection]
    metrics: Metrics
    meta: dict[str, Any] = field(default_factory=dict)

    def result_hash(self) -> str:
        """Exact fingerprint used by golden tests: any change in engine math changes it."""
        return hash_obj(
            {
                "signals": [s.model_dump(mode="json") for s in self.signals],
                "trades": [t.__dict__ for t in self.trades],
                "equity": self.equity_curve,
            }
        )


def _bar_from(buf_row: Mapping[str, float | int], symbol: str, tf: Timeframe) -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=tf,
        open_time=from_ns(int(buf_row["open_time"])),
        close_time=from_ns(int(buf_row["close_time"])),
        bid_o=float(buf_row["bid_o"]),
        bid_h=float(buf_row["bid_h"]),
        bid_l=float(buf_row["bid_l"]),
        bid_c=float(buf_row["bid_c"]),
        ask_o=float(buf_row["ask_o"]),
        ask_h=float(buf_row["ask_h"]),
        ask_l=float(buf_row["ask_l"]),
        ask_c=float(buf_row["ask_c"]),
        volume=float(buf_row["volume"]),
    )


def run_backtest(
    strategy_cls: type[Strategy],
    m1: Mapping[str, BarsArray],
    series: Mapping[tuple[str, Timeframe], BarsArray],
    instruments: Mapping[str, InstrumentCosts],
    cost_model: CostModel,
    *,
    params: Mapping[str, ParamValue] | None = None,
    config: BacktestConfig | None = None,
    rates: StaticRates | None = None,
    gate: OrderGate | None = None,
    meta: Mapping[str, Any] | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    manifest = strategy_cls.manifest
    if manifest.demo_only and meta and meta.get("purpose") == "promotion":
        raise StrategyError("demo_only strategies can never be used for promotion")
    pvals = manifest.param_values(dict(params or {}))
    symbols = list(manifest.symbols)
    tfs = sorted(manifest.timeframes, key=lambda t: t.minutes)
    subs = [(s, tf) for s in symbols for tf in tfs]
    missing = [k for k in subs if k not in series and not (k[1] == Timeframe.M1 and k[0] in m1)]
    if missing or any(s not in m1 for s in symbols):
        raise ValueError(f"missing data for {missing or [s for s in symbols if s not in m1]}")

    broker = SimBroker.build(
        {s: m1[s] for s in symbols},
        dict(instruments),
        cost_model,
        rates or StaticRates(),
        cfg.account_ccy,
        cfg.initial_equity,
    )
    market = EngineMarketView(cost_model.spreads.spread, news_from_times(cost_model.spreads.news_ns))
    for s, tf in subs:
        market.add_series(s, tf, SeriesBuffer(series.get((s, tf), m1[s] if tf == Timeframe.M1 else None)))

    strategy = strategy_cls()
    ctx = EngineContext(
        manifest.id,
        manifest.version,
        market,
        pvals,
        broker.position_views,
        broker.pending_views,
        seed=cfg.seed,
    )
    guard: OrderGate = gate or BacktestGuard(
        "backtest", cfg.risk_fraction, open_risk_total_max=cfg.open_risk_total_max
    )
    warmup = strategy.warmup()

    # ---- timeline ----
    start_ns = min(int(m1[s].open_time[0]) for s in symbols)
    end_ns = max(int(m1[s].close_time[-1]) for s in symbols)
    rolls = rollover_times(from_ns(start_ns), from_ns(end_ns))
    roll_weekday = dict(rolls)
    closes = [market.series(s, tf) for s, tf in subs]
    times = np.unique(
        np.concatenate(
            [np.asarray([t for t, _ in rolls], dtype=np.int64)] + [b.close_times() for b in closes]
        )
    )
    clock = SimClock(from_ns(start_ns))
    ptr = dict.fromkeys(symbols, 0)
    next_close = [0] * len(subs)
    signals: list[Signal] = []
    seen_ids: set[str] = set()
    equity_curve: list[tuple[int, float]] = []
    tf_of_strategy = tfs[0]

    def open_risk() -> float:
        total = 0.0
        for sym, ps in broker.positions.items():
            d = broker.symbols[sym]
            for p in ps:
                per_unit = p.entry_price - p.stop if p.side == "buy" else p.stop - p.entry_price
                total += max(0.0, per_unit) * d.costs.contract_size * p.lots * d.to_account
        return total

    def handle(reqs: Sequence[Request], now_ns: int, tf: Timeframe) -> None:
        for r in check_requests(reqs, manifest, market.now, seen_ids):
            if isinstance(r, Signal):
                sid = str(r.signal_id)
                signals.append(r)
                d = broker.symbols[r.symbol]
                i = broker.last_idx.get(r.symbol, -1)
                if i < 0:
                    broker.rejections.append(Rejection(now_ns, f"signal {sid}", "no price yet"))
                    continue
                if r.entry_type == "market":
                    ref = float(d.ask_c[i]) if r.side == "buy" else float(d.bars.bid_c[i])
                else:
                    ref = float(r.entry_price or 0.0)
                res = guard.decide(
                    GateRequest(
                        r, ref, float(d.spread[i]), broker.equity(), open_risk(), d.costs, d.to_account
                    )
                )
                if res.lots <= 0:
                    broker.rejections.append(Rejection(now_ns, f"signal {sid}", "; ".join(res.reasons)))
                    continue
                broker.submit_entry(r, res.lots, now_ns, tf.minutes)
            elif isinstance(r, CancelRequest):
                broker.cancel(manifest.id, str(r.signal_id), now_ns)
            elif isinstance(r, CloseRequest):
                broker.request_close(manifest.id, r.position_id, now_ns, "close_request")
            elif isinstance(r, ModifyStopRequest):
                broker.modify_stop(manifest.id, r.position_id, r.new_stop, now_ns)
            else:
                raise StrategyError(f"unsupported request type {type(r).__name__}")

    def warmed() -> bool:
        return all(market.series(s, tf).visible >= n for (s, tf), n in warmup.items())

    for t_np in times:
        t = int(t_np)
        for s in symbols:
            ct = m1[s].close_time
            hi = int(np.searchsorted(ct, t, side="right"))
            if hi > ptr[s]:
                if broker.has_activity(s):
                    broker.advance(s, ptr[s], hi)
                else:
                    broker.last_idx[s] = hi - 1
                ptr[s] = hi
        clock.advance_to(from_ns(t))
        market.set_now(t)
        fills, broker.fills_out = broker.fills_out, []
        for _, fill in fills:
            handle(strategy.on_fill(ctx, fill), t, tf_of_strategy)
        if t in roll_weekday:
            broker.charge_swaps(roll_weekday[t])
            equity_curve.append((t, broker.equity()))
        due = []
        for k, (s, tf) in enumerate(subs):
            buf = closes[k]
            j = next_close[k]
            if j < buf.size and buf.close_time(j) == t:
                buf.advance_to(j + 1)
                next_close[k] = j + 1
                due.append((s, tf, buf))
            elif j < buf.size and buf.close_time(j) < t:
                raise RuntimeError("series bar skipped; timeline is inconsistent")
        if due and warmed():
            for s, tf, buf in due:
                bar = _bar_from(buf.row(buf.visible - 1), s, tf)
                event = BarClosed(at=bar.close_time, symbol=s, timeframe=tf, bar=bar)
                handle(strategy.on_bar(ctx, event), t, tf)

    for s in symbols:
        n = len(m1[s])
        if n > ptr[s]:
            broker.advance(s, ptr[s], n)
            ptr[s] = n
    broker.close_all("end_of_data")
    broker.fills_out = []
    equity_curve.append((end_ns, broker.equity()))
    eq = [cfg.initial_equity] + [e for _, e in equity_curve]
    return BacktestResult(
        strategy_id=manifest.id,
        strategy_version=manifest.version,
        params=dict(pvals),
        signals=signals,
        trades=broker.trades,
        equity_curve=equity_curve,
        rejections=broker.rejections,
        metrics=compute(broker.trades, eq),
        meta=dict(meta or {}),
    )
