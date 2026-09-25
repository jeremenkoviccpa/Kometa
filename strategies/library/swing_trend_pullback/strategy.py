"""swing_trend_pullback: swing trading with the daily trend, entering on a pullback. A CANDIDATE: it has
proven nothing until validation, shadow and micro say so.

Rules (H4 execution, D1 trend):
1. Trend: the D1 close is above its EMA(trend_len) and that EMA is higher than 5 days ago (up), or the mirror
   (down). No trend, no trade.
2. Pullback: the last H4 bar reached back to within `pullback_atr` x ATR of the H4 EMA(pullback_ema) and
   closed back on the trend side of it.
3. Trigger: that bar is a reversal candle in the trend direction (pin bar, engulfing bar or morning/evening
   star).
4. Stop beyond the extreme of the last two H4 bars plus `stop_buffer_atr` x ATR, at least 1 ATR away.
   Target `reward_risk` x the risk.
5. Management: at +`be_at_r` R the stop moves to entry (a trade that ran well cannot turn into a full loss);
   the position is closed if the D1 trend turns against it. One position per symbol.
"""

from __future__ import annotations

import math

import numpy as np

from autotrader.core.events import BarClosed
from autotrader.core.indicators import atr, ema, patterns, reversals
from autotrader.core.models import Timeframe
from autotrader.core.series import BarsArray
from autotrader.strategies_api import Request, Strategy, StrategyContext


def trend_of(close: np.ndarray, n: int) -> int:
    c = ema(close, n)
    last = float(close[-1])
    if math.isnan(c[-1]) or math.isnan(c[-6]):  # not enough history
        return 0
    if last > c[-1] and c[-1] > c[-6]:
        return 1
    if last < c[-1] and c[-1] < c[-6]:
        return -1
    return 0


class SwingTrendPullback(Strategy):
    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        trend = int(self.manifest.params["trend_len"].max or 200)
        return {
            **{(s, Timeframe.D1): trend + 10 for s in self.manifest.symbols},
            **{(s, Timeframe.H4): 150 for s in self.manifest.symbols},
        }

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        if event.timeframe != Timeframe.H4:
            return []
        p = ctx.params
        sym = event.symbol
        trend_len = int(p["trend_len"])
        d1 = ctx.market.bars(sym, Timeframe.D1, trend_len * 2 + 10)
        h4 = ctx.market.bars(sym, Timeframe.H4, 150)
        if len(d1) < trend_len + 10 or len(h4) < 60:
            return []
        trend = trend_of(d1.bid_c, trend_len)
        a = float(atr(h4.bid_h, h4.bid_l, h4.bid_c, int(p["atr_len"]))[-1])
        if not a > 0:
            return []
        out = self.manage(ctx, sym, h4, trend)
        if trend == 0 or ctx.my_positions(sym) or ctx.my_pending(sym):
            return out
        e = float(ema(h4.bid_c, int(p["pullback_ema"]))[-1])
        zone = float(p["pullback_atr"]) * a
        rev = reversals(h4.bid_o, h4.bid_h, h4.bid_l, h4.bid_c)
        pat = patterns(h4.bid_o, h4.bid_h, h4.bid_l, h4.bid_c)
        buf = float(p["stop_buffer_atr"]) * a
        rr = float(p["reward_risk"])
        close = float(h4.bid_c[-1])
        if trend > 0:
            pulled = float(h4.bid_l[-1]) <= e + zone and close > e
            trigger = bool(rev.bull_pin[-1] or pat.bull_engulfing[-1] or rev.morning_star[-1])
            if pulled and trigger:
                entry = float(h4.ask_c[-1])
                stop = min(float(h4.bid_l[-1]), float(h4.bid_l[-2])) - buf
                stop = min(stop, entry - a)
                out.append(
                    ctx.signal(
                        sym,
                        "buy",
                        stop,
                        target_price=entry + rr * (entry - stop),
                        reason="pullback in uptrend",
                    )
                )
        else:
            pulled = float(h4.ask_h[-1]) >= e - zone and close < e
            trigger = bool(rev.bear_pin[-1] or pat.bear_engulfing[-1] or rev.evening_star[-1])
            if pulled and trigger:
                entry = close
                stop = max(float(h4.ask_h[-1]), float(h4.ask_h[-2])) + buf
                stop = max(stop, entry + a)
                out.append(
                    ctx.signal(
                        sym,
                        "sell",
                        stop,
                        target_price=entry - rr * (stop - entry),
                        reason="pullback in downtrend",
                    )
                )
        return out

    def manage(self, ctx: StrategyContext, sym: str, h4: BarsArray, trend: int) -> list[Request]:
        out: list[Request] = []
        be_at = float(ctx.params["be_at_r"])
        for pos in ctx.my_positions(sym):
            risk = ctx.state.setdefault(f"risk:{pos.position_id}", abs(pos.entry_price - pos.stop_price))
            if (pos.side == "buy" and trend < 0) or (pos.side == "sell" and trend > 0):
                out.append(ctx.close(pos.position_id, "daily trend turned"))
                continue
            if not risk > 0:
                continue
            if pos.side == "buy":
                gain = float(h4.bid_c[-1]) - pos.entry_price
                if gain >= be_at * risk and pos.stop_price < pos.entry_price:
                    out.append(ctx.modify_stop(pos.position_id, pos.entry_price, "breakeven"))
            else:
                gain = pos.entry_price - float(h4.ask_c[-1])
                if gain >= be_at * risk and pos.stop_price > pos.entry_price:
                    out.append(ctx.modify_stop(pos.position_id, pos.entry_price, "breakeven"))
        return out
