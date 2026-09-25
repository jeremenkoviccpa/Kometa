"""demo_ma_cross: EMA crossover. NOT FOR LIVE TRADING; capped at Shadow forever (demo_only).

Buys when the fast EMA of H1 bid closes crosses above the slow EMA, sells on
the opposite cross. Stop at `stop_atr` x ATR, target at `reward_risk` x stop
distance. One position per symbol; an opposite cross closes it.
"""

from __future__ import annotations

from autotrader.core.events import BarClosed
from autotrader.core.indicators import atr, ema
from autotrader.core.models import Timeframe
from autotrader.strategies_api import Request, Strategy, StrategyContext


class DemoMaCross(Strategy):
    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        slow = int(self.manifest.params["slow"].value)
        return {(s, Timeframe.H1): slow + 2 for s in self.manifest.symbols}

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        fast_n, slow_n = int(ctx.params["fast"]), int(ctx.params["slow"])
        if fast_n >= slow_n:
            return []
        bars = ctx.market.bars(event.symbol, Timeframe.H1, slow_n * 4)
        if len(bars) < slow_n + 2:
            return []
        c = bars.bid_c
        f, s = ema(c, fast_n), ema(c, slow_n)
        crossed_up = f[-2] <= s[-2] and f[-1] > s[-1]
        crossed_down = f[-2] >= s[-2] and f[-1] < s[-1]
        if not (crossed_up or crossed_down):
            return []
        out: list[Request] = [
            ctx.close(p.position_id, "opposite cross") for p in ctx.my_positions(event.symbol)
        ]
        a = float(atr(bars.bid_h, bars.bid_l, c, int(ctx.params["atr_len"]))[-1])
        if not a > 0:  # also false for NaN
            return out
        dist = float(ctx.params["stop_atr"]) * a
        rr = float(ctx.params["reward_risk"])
        if crossed_up:
            entry = float(bars.ask_c[-1])
            out.append(
                ctx.signal(
                    event.symbol, "buy", entry - dist, target_price=entry + rr * dist, reason="ema cross up"
                )
            )
        else:
            entry = float(c[-1])
            out.append(
                ctx.signal(
                    event.symbol,
                    "sell",
                    entry + dist,
                    target_price=entry - rr * dist,
                    reason="ema cross down",
                )
            )
        return out
