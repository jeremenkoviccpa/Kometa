"""scalp_session_breakout: intraday scalp of the London open breaking the Asian range. A CANDIDATE: it has
proven nothing until validation, shadow and micro say so. Short trades live or die on costs: the spread
filter below and the backtest's cost model are what decide whether anything is left.

Rules (M5 execution, H1 volatility), all times UTC:
1. Range: high and low of today's M5 bars from 00:00 to `range_end_h`.
2. Width filter: the range must be between 1 and `max_width_atr` x the H1 ATR (a tiny range is noise, a
   huge one has already spent the day's move).
3. Trigger, from `range_end_h` for 4 hours: an M5 bar closes beyond the range by `buffer` x its width.
4. Stop at the middle of the range; target `reward_risk` x the risk. Skipped if the spread is more than
   `max_spread_r` of the risk.
5. At most one trade a day; anything still open at `exit_h` is closed at market.
"""

from __future__ import annotations

from autotrader.core.events import BarClosed
from autotrader.core.indicators import atr
from autotrader.core.models import Timeframe
from autotrader.strategies_api import Request, Strategy, StrategyContext

NS_HOUR = 3_600_000_000_000
NS_DAY = 24 * NS_HOUR


class ScalpSessionBreakout(Strategy):
    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        return {
            **{(s, Timeframe.M5): 300 for s in self.manifest.symbols},
            **{(s, Timeframe.H1): 60 for s in self.manifest.symbols},
        }

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        if event.timeframe != Timeframe.M5:
            return []
        p = ctx.params
        sym = event.symbol
        m5 = ctx.market.bars(sym, Timeframe.M5, 300)
        if len(m5) < 100:
            return []
        t = int(m5.close_time[-1])
        day, hour = t // NS_DAY, (t % NS_DAY) / NS_HOUR  # hour of the bar's close, fractional
        range_end, exit_h = float(p["range_end_h"]), float(p["exit_h"])
        out: list[Request] = []
        if hour >= exit_h:
            out += [ctx.close(pos.position_id, "session over") for pos in ctx.my_positions(sym)]
            return out
        if not (range_end <= hour - 5 / 60 < range_end + 4):  # the bar opened inside the trade window
            return out
        if ctx.state.get(f"traded:{sym}") == day or ctx.my_positions(sym) or ctx.my_pending(sym):
            return out
        opens = m5.open_time
        today = (opens // NS_DAY == day) & ((opens % NS_DAY) < int(range_end * NS_HOUR))
        if int(today.sum()) < 12:  # less than an hour of range data (holiday, late open)
            return out
        r_hi, r_lo = float(m5.bid_h[today].max()), float(m5.bid_l[today].min())
        width = r_hi - r_lo
        h1 = ctx.market.bars(sym, Timeframe.H1, 60)
        a = float(atr(h1.bid_h, h1.bid_l, h1.bid_c, 14)[-1]) if len(h1) >= 20 else float("nan")
        if not (a > 0 and a <= width <= float(p["max_width_atr"]) * a):
            return out
        close, buf = float(m5.bid_c[-1]), float(p["buffer"]) * width
        mid, rr = (r_hi + r_lo) / 2, float(p["reward_risk"])
        spread = ctx.market.spread(sym)
        if close > r_hi + buf:
            entry = float(m5.ask_c[-1])
            risk = entry - mid
            if risk > 0 and spread <= float(p["max_spread_r"]) * risk:
                ctx.state[f"traded:{sym}"] = day
                out.append(
                    ctx.signal(
                        sym, "buy", mid, target_price=entry + rr * risk, reason="asian range breakout up"
                    )
                )
        elif close < r_lo - buf:
            entry = close
            risk = mid - entry
            if risk > 0 and spread <= float(p["max_spread_r"]) * risk:
                ctx.state[f"traded:{sym}"] = day
                out.append(
                    ctx.signal(
                        sym, "sell", mid, target_price=entry - rr * risk, reason="asian range breakout down"
                    )
                )
        return out
