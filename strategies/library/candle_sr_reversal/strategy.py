"""candle_sr_reversal: price action at support and resistance (candlestick reversals at tested levels). A
CANDIDATE: it has proven nothing until validation, shadow and micro say so.

Rules (H1 execution, H4 levels):
1. Levels: support and resistance from clustered swing points of the last 300 H4 bars
   (core.indicators.levels), recomputed when an H4 bar closes; a level exists only once its swings are
   confirmed. Only levels touched at least `min_touches` times and not broken count.
2. Setup: the last H1 bar reached into a level's zone (`zone_atr` x H4 ATR) and closed back outside it,
   on the side the level defends.
3. Trigger: that bar completes a reversal pattern away from the level, from the full classic set
   (core.indicators.candlestick, ported from cm45t3r/candlestick): at support a hammer shape, bullish
   engulfing, bullish harami, piercing line, tweezers bottom, morning star or bullish kicker; at resistance
   an inverted-hammer shape (a shooting star without the gap), hanging man, bearish engulfing, bearish harami,
   dark cloud cover, tweezers top, evening star or bearish kicker. The pattern is recorded on the signal.
4. Trend: with `with_trend` on, buys at support need the H4 EMA(50) rising and sells at resistance need it
   falling (fading a strong trend at every level is how these methods usually die).
5. Stop beyond the candle's extreme plus `stop_buffer_atr` x H1 ATR. Target: just before the next opposite
   level, but the trade is skipped if that gives less than `min_rr` R, and capped at `max_rr` R.
6. Management: stop to entry at +1R. One position per symbol.
"""

from __future__ import annotations

from autotrader.core.events import BarClosed
from autotrader.core.indicators import atr, candlestick_patterns, ema, levels
from autotrader.core.models import Timeframe
from autotrader.core.series import BarsArray
from autotrader.strategies_api import Request, Strategy, StrategyContext

H4_WINDOW = 300
AT_SUPPORT = (
    "hammer",
    "bullish_engulfing",
    "bullish_harami",
    "piercing_line",
    "tweezers_bottom",
    "morning_star",
    "bullish_kicker",
)
AT_RESISTANCE = (
    "inverted_hammer",
    "hanging_man",
    "bearish_engulfing",
    "bearish_harami",
    "dark_cloud_cover",
    "tweezers_top",
    "evening_star",
    "bearish_kicker",
)


class CandleSrReversal(Strategy):
    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        return {
            **{(s, Timeframe.H4): H4_WINDOW for s in self.manifest.symbols},
            **{(s, Timeframe.H1): 100 for s in self.manifest.symbols},
        }

    def levels(self, ctx: StrategyContext, sym: str) -> list[tuple[float, str, int]]:
        """(price, kind, touches) of intact levels after the last closed H4 bar, cached per H4 bar
        (strategy state must stay JSON: plain lists only)."""
        h4 = ctx.market.bars(sym, Timeframe.H4, H4_WINDOW)
        if len(h4) < 50:
            return []
        stamp = int(h4.close_time[-1])
        cached = ctx.state.get(f"lv:{sym}")
        if cached is not None and cached[0] == stamp:
            return [(float(x), str(k), int(n)) for x, k, n in cached[1]]
        snap = levels(h4.bid_h, h4.bid_l, h4.bid_c, k=float(ctx.params["level_k"]))[-1]
        out = [(lv.price, lv.kind, lv.touches) for lv in snap if not lv.broken]
        ctx.state[f"lv:{sym}"] = [stamp, [list(x) for x in out]]
        return out

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        if event.timeframe != Timeframe.H1:
            return []
        p = ctx.params
        sym = event.symbol
        lvls = self.levels(ctx, sym)
        h1 = ctx.market.bars(sym, Timeframe.H1, 100)
        h4 = ctx.market.bars(sym, Timeframe.H4, 120)
        if len(h1) < 30 or len(h4) < 60:
            return []
        a1 = float(atr(h1.bid_h, h1.bid_l, h1.bid_c, 14)[-1])
        a4 = float(atr(h4.bid_h, h4.bid_l, h4.bid_c, 14)[-1])
        if not (a1 > 0 and a4 > 0):
            return []
        out = self.breakeven(ctx, sym, h1)
        if ctx.my_positions(sym) or ctx.my_pending(sym):
            return out
        e50 = ema(h4.bid_c, 50)
        slope = float(e50[-1] - e50[-6])
        with_trend = bool(p["with_trend"])
        zone = float(p["zone_atr"]) * a4
        min_touch = int(p["min_touches"])
        n = 3  # the longest pattern; only the newest bar's patterns matter
        pats = candlestick_patterns(h1.bid_o[-n:], h1.bid_h[-n:], h1.bid_l[-n:], h1.bid_c[-n:])
        lo, hi, close = float(h1.bid_l[-1]), float(h1.ask_h[-1]), float(h1.bid_c[-1])
        buf = float(p["stop_buffer_atr"]) * a1
        min_rr, max_rr = float(p["min_rr"]), float(p["max_rr"])
        supports = [x for x, k, n in lvls if k == "support" and n >= min_touch]
        resist = [x for x, k, n in lvls if k == "resistance" and n >= min_touch]
        bull_hit = [name for name in AT_SUPPORT if bool(pats[name][-1])]
        bear_hit = [name for name in AT_RESISTANCE if bool(pats[name][-1])]
        bull, bear = bool(bull_hit), bool(bear_hit)
        if bull and (not with_trend or slope >= 0):
            level = next((x for x in supports if lo <= x + zone and close > x), None)
            if level is not None:
                entry = float(h1.ask_c[-1])
                stop = lo - buf
                risk = entry - stop
                above = [x for x in resist if x > entry]
                room = (min(above) - 0.1 * a1 - entry) if above else max_rr * risk
                if risk > 0 and room >= min_rr * risk:
                    target = entry + min(room, max_rr * risk)
                    out.append(
                        ctx.signal(
                            sym,
                            "buy",
                            stop,
                            target_price=target,
                            reason=f"{bull_hit[0]} at support {level:.2f}",
                            tags={"pattern": ",".join(bull_hit), "level": f"{level:.2f}"},
                        )
                    )
        elif bear and (not with_trend or slope <= 0):
            level = next((x for x in reversed(resist) if hi >= x - zone and close < x), None)
            if level is not None:
                entry = close
                stop = hi + buf
                risk = stop - entry
                below = [x for x in supports if x < entry]
                room = (entry - max(below) - 0.1 * a1) if below else max_rr * risk
                if risk > 0 and room >= min_rr * risk:
                    target = entry - min(room, max_rr * risk)
                    out.append(
                        ctx.signal(
                            sym,
                            "sell",
                            stop,
                            target_price=target,
                            reason=f"{bear_hit[0]} at resistance {level:.2f}",
                            tags={"pattern": ",".join(bear_hit), "level": f"{level:.2f}"},
                        )
                    )
        return out

    def breakeven(self, ctx: StrategyContext, sym: str, h1: BarsArray) -> list[Request]:
        out: list[Request] = []
        for pos in ctx.my_positions(sym):
            risk = ctx.state.setdefault(f"risk:{pos.position_id}", abs(pos.entry_price - pos.stop_price))
            if not risk > 0:
                continue
            gain = (
                float(h1.bid_c[-1]) - pos.entry_price
                if pos.side == "buy"
                else pos.entry_price - float(h1.ask_c[-1])
            )
            below_entry = (
                pos.stop_price < pos.entry_price if pos.side == "buy" else pos.stop_price > pos.entry_price
            )
            if below_entry and gain >= risk:
                out.append(ctx.modify_stop(pos.position_id, pos.entry_price, "breakeven"))
        return out
