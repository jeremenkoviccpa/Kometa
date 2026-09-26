"""smc_sniper: the owner's SMC/ICT top-down "1M sniper entry" method as fixed rules. A CANDIDATE: it has
proven nothing until validation, shadow and micro say so. The method is discretionary; every step below is
the mechanical reading of it, and "not enough confluence" always means NO TRADE (no signal).

Rules. Shorts mirror longs exactly (the code flips prices, so both sides run the same lines).
1. HTF bias (D1 -> H4): D1 and H4 structure (higher highs + higher lows / lower highs + lower lows,
   core.indicators.trend_state) must agree; otherwise NO TRADE.
2. Dealing range (H4 external structure): the last confirmed H4 swing high and swing low. The 50% level
   is equilibrium: longs only from discount (below it), shorts only from premium (above it).
3. POI (H1 zone, refined on M15): an order block, meaning the last opposite candle before a displacement
   candle (body >= `disp_atr` x ATR) that leaves a fair value gap. Only unmitigated blocks count: no
   close through the block since it formed. It must lie wholly in discount (premium for shorts). The
   newest H1 block is the zone; the newest M15 block inside it refines it. Recomputed on every M15 close.
4. M5 reaction: an M5 bar trades into the POI and closes in the bias direction. This opens a hunt window
   of `window_m` minutes for the M1 sequence.
5. M1 sequence, in this order, within the window:
   a. liquidity sweep: the window's extreme low trades below the lowest low of the `liq_lookback` bars
      before it (sell-side liquidity, equal lows) and reaches the POI;
   b. rejection: that bar or the next closes back above the swept level;
   c. CHOCH/MSS: the first close above the last M1 lower high (swing high) before the sweep, made by a
      displacement candle (body >= `disp_atr` x M1 ATR). A wick through it, or a weak close, is no CHOCH;
   d. displacement leaves a fair value gap (the bar after it stays above the bar before it);
   e. BOS: a later close above the high of the move from the sweep through the gap;
   f. pullback into the fair value gap without a close below it;
   g. confirmation: the bar that ends the pullback is a bullish pin bar or a bullish engulfing.
6. Entry at market on the confirmation close (never chasing: the setup is only valid on that bar).
7. Stop beyond the sweep's extreme plus `stop_buffer_atr` x M1 ATR (the logical invalidation). If that
   stop is wider than `max_sl_atr` x H1 ATR: NO TRADE.
8. Target: the nearest real liquidity beyond the entry: an untaken H1 swing high, the previous day's
   high, or the dealing range's high. The target is chosen first; if it gives less than `min_rr` (3R):
   NO TRADE.
9. One position or pending order per symbol. Size per trade is not the strategy's call: the allocator
   and the signed risk limits decide it, and can only make it smaller.

Strictness switches (not tunable; the defaults are the method as written): `require_d1` (rule 1; off: H4
bias with D1 not against it), `m15_poi` (rule 3; on: an M15 block in discount is a POI without an H1 zone),
`require_m5` (rule 4; off: the M1 hunt runs whenever there is a POI), `require_bos` (rule 5e; off: the
displacement's close through the lower high is the break). strategies/library/smc_sniper_active is this
same file (a test keeps them identical) with all four relaxed, for more trades to judge.
"""

from __future__ import annotations

import math

import numpy as np

from autotrader.core.events import BarClosed
from autotrader.core.indicators import atr, reversals, swings, trend_state
from autotrader.core.models import Timeframe
from autotrader.core.series import BarsArray
from autotrader.strategies_api import Request, Strategy, StrategyContext

NS_MIN = 60_000_000_000
Candles = tuple[list[float], list[float], list[float], list[float]]


def _candles(b: BarsArray, sell: bool) -> Candles:
    """Bid OHLC as float lists; for shorts negated with high and low swapped, so a short is a long."""
    if not sell:
        return b.bid_o.tolist(), b.bid_h.tolist(), b.bid_l.tolist(), b.bid_c.tolist()
    return (-b.bid_o).tolist(), (-b.bid_l).tolist(), (-b.bid_h).tolist(), (-b.bid_c).tolist()


def _atr(x: Candles) -> list[float]:
    _, h, lo, c = x
    return [float(v) for v in atr(np.array(h), np.array(lo), np.array(c), 14)]


def order_blocks(x: Candles, k: float) -> list[tuple[float, float, int]]:
    """Unmitigated bullish order blocks (low, high, index of the bar that completes the gap)."""
    o, h, lo, c = x
    a = _atr(x)
    n = len(c)
    floor = [math.inf] * (n + 1)  # lowest close from index i on
    for i in range(n - 1, -1, -1):
        floor[i] = min(c[i], floor[i + 1])
    out = []
    for i in range(2, n):
        d = i - 1
        if not (lo[i] > h[i - 2] and c[d] - o[d] >= k * a[d] > 0):
            continue
        j = next((y for y in range(d - 1, max(d - 4, -1), -1) if c[y] < o[y]), d - 1)
        if floor[i + 1] >= lo[j]:
            out.append((lo[j], h[j], i))
    return out


def liquidity_above(x: Candles) -> list[float]:
    """Confirmed swing highs no later bar has traded through yet: resting buy-side liquidity."""
    _, h, lo, _ = x
    s = swings(np.array(h), np.array(lo)).high.tolist()
    n = len(h)
    top = [-math.inf] * (n + 1)  # highest high from index i on
    for i in range(n - 1, -1, -1):
        top[i] = max(h[i], top[i + 1])
    # a swing confirmed at t is the high of bar t-2 (left = right = 2)
    return [v for t, v in enumerate(s) if not math.isnan(v) and top[t - 1] <= v]


def _last(v: np.ndarray) -> float:
    ok = v[~np.isnan(v)]
    return float(ok[-1]) if ok.size else math.nan


def _pivot_high(h: list[float], k: int) -> bool:
    return h[k] > h[k - 1] and h[k] > h[k - 2] and h[k] >= h[k + 1] and h[k] >= h[k + 2]


def m1_sequence(
    x: Candles, window: int, lookback: int, disp: float, require_bos: bool = True
) -> dict[str, float] | None:
    """The M1 sequence (rules 5a-5g) ending on the last bar, or None. Levels are in the long frame.
    Without `require_bos` the displacement's close through the lower high is the break (5e skipped), and the
    pullback must come after the gap bar."""
    o, h, lo, c = x
    a = _atr(x)
    n = len(c)
    w0 = max(n - window, lookback + 3)
    if n - 5 < w0:
        return None
    s = min(range(w0, n - 4), key=lambda i: (lo[i], i))  # the sweep: the window's extreme
    liq = min(lo[s - lookback : s])
    if not (lo[s] < liq and max(c[s], c[s + 1]) > liq):  # 5a sweep, 5b rejection
        return None
    choch = next((h[k] for k in range(s - 2, s - lookback, -1) if _pivot_high(h, k)), None)
    if choch is None:
        return None
    x_ = next((i for i in range(s + 1, n) if c[i] > choch), None)  # 5c: first close through
    if x_ is None or x_ + 1 >= n or not c[x_] - o[x_] >= disp * a[x_] > 0:
        return None
    fvg_lo, fvg_hi = h[x_ - 1], lo[x_ + 1]
    if not fvg_hi > fvg_lo:  # 5d
        return None
    leg_hi = max(h[s : x_ + 2])
    b = next((i for i in range(x_ + 2, n - 1) if c[i] > leg_hi), None) if require_bos else x_ + 1  # 5e
    if b is None or min(c[x_ + 1 :]) < fvg_lo:  # 5f: no close through the gap
        return None
    e = n - 1
    first = e - 1 if require_bos else max(e - 1, x_ + 2)
    if first > e or not min(lo[first:]) <= fvg_hi:  # 5f: the pullback reaches the gap on this bar or the last
        return None
    pin = bool(
        reversals(np.array(o[-3:]), np.array(h[-3:]), np.array(lo[-3:]), np.array(c[-3:])).bull_pin[-1]
    )
    engulf = c[e - 1] < o[e - 1] and c[e] > o[e] and o[e] <= c[e - 1] and c[e] >= o[e - 1]
    if not (pin or engulf):  # 5g
        return None
    return {
        "sweep": lo[s],
        "swept": liq,
        "choch": choch,
        "fvg_lo": fvg_lo,
        "fvg_hi": fvg_hi,
        "atr": a[e],
        "sweep_bar": float(s - n),
        "confirm": 1.0 if pin else 2.0,
    }


class SmcSniper(Strategy):
    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        w = {Timeframe.M1: 400, Timeframe.M5: 50, Timeframe.M15: 200, Timeframe.H1: 200, Timeframe.H4: 300}
        return {(s, tf): n for s in self.manifest.symbols for tf, n in {**w, Timeframe.D1: 60}.items()}

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        sym = event.symbol
        if event.timeframe == Timeframe.H1:
            self._structure(ctx, sym)
            self._refine(ctx, sym)
        elif event.timeframe == Timeframe.M15:
            self._refine(ctx, sym)
        elif event.timeframe == Timeframe.M5:
            self._reaction(ctx, sym)
        elif event.timeframe == Timeframe.M1:
            return self._hunt(ctx, sym)
        return []

    def _structure(self, ctx: StrategyContext, sym: str) -> None:
        """Rules 1-3 on the higher timeframes and the liquidity map, rebuilt on every H1 close."""
        ctx.state[f"htf:{sym}"] = None
        m = ctx.market
        d1, h4, h1 = (
            m.bars(sym, Timeframe.D1, 60),
            m.bars(sym, Timeframe.H4, 300),
            m.bars(sym, Timeframe.H1, 200),
        )
        if min(len(d1), len(h4)) < 30 or len(h1) < 60:
            return
        d1_bias = float(trend_state(d1.bid_h, d1.bid_l)[-1])
        bias = float(trend_state(h4.bid_h, h4.bid_l)[-1])
        if bias == 0 or (d1_bias != bias if ctx.params["require_d1"] else d1_bias == -bias):
            return  # 1: no agreed bias (or, without require_d1, D1 against H4)
        sell = bias < 0
        sw = swings(h4.bid_h, h4.bid_l)
        r_hi, r_lo = _last(sw.high), _last(sw.low)
        if not r_hi > r_lo:
            return
        sign = -1.0 if sell else 1.0
        eq = sign * (r_hi + r_lo) / 2
        top = -r_lo if sell else r_hi  # the range's far side in the long frame
        h1c = _candles(h1, sell)
        zones = [z for z in order_blocks(h1c, float(ctx.params["disp_atr"])) if z[1] <= eq]
        if not zones and not ctx.params["m15_poi"]:
            return  # 3: no POI in discount
        liq = [*liquidity_above(h1c), _candles(d1, sell)[1][-1], top]
        ctx.state[f"htf:{sym}"] = {
            "sell": sell,
            "zone": [zones[-1][0], zones[-1][1]] if zones else None,
            "eq": eq,
            "liq": liq,
            "h1_atr": _atr(h1c)[-1],
        }

    def _refine(self, ctx: StrategyContext, sym: str) -> None:
        """Rule 3, M15 part: the newest M15 block inside the H1 zone narrows the POI. With `m15_poi` and no H1
        zone, the newest M15 block in discount is the POI on its own."""
        htf = ctx.state.get(f"htf:{sym}")
        if not htf:
            ctx.state[f"ctx:{sym}"] = None
            return
        m15 = ctx.market.bars(sym, Timeframe.M15, 200)
        blocks = [
            z
            for z in order_blocks(_candles(m15, bool(htf["sell"])), float(ctx.params["disp_atr"]))
            if z[1] <= htf["eq"]
        ]
        if htf["zone"] is None:
            if not blocks:
                ctx.state[f"ctx:{sym}"] = None
                return
            z_lo, z_hi = blocks[-1][0], blocks[-1][1]
        else:
            z_lo, z_hi = htf["zone"]
            fine = [z for z in blocks if z[0] <= z_hi and z[1] >= z_lo]
            if fine:
                z_lo, z_hi = max(fine[-1][0], z_lo), min(fine[-1][1], z_hi)
        ctx.state[f"ctx:{sym}"] = {**htf, "poi": [z_lo, z_hi]}

    def _reaction(self, ctx: StrategyContext, sym: str) -> None:
        """Rule 4: the M5 reaction at the POI opens the M1 hunt window."""
        c = ctx.state.get(f"ctx:{sym}")
        if not c or not ctx.params["require_m5"]:
            return
        m5 = ctx.market.bars(sym, Timeframe.M5, 2)
        if not len(m5):
            return
        o, _, lo, cl = _candles(m5, bool(c["sell"]))
        z_lo, z_hi = c["poi"]
        if lo[-1] <= z_hi and cl[-1] >= z_lo and cl[-1] > o[-1]:
            ctx.state[f"hunt:{sym}"] = int(m5.close_time[-1]) + int(ctx.params["window_m"]) * NS_MIN

    def _hunt(self, ctx: StrategyContext, sym: str) -> list[Request]:
        """Rules 5-9 on each M1 close inside the hunt window."""
        p = ctx.params
        until, c = ctx.state.get(f"hunt:{sym}"), ctx.state.get(f"ctx:{sym}")
        if (p["require_m5"] and not until) or not c or ctx.my_positions(sym) or ctx.my_pending(sym):
            return []
        window, lookback = int(p["window_m"]), int(p["liq_lookback"])
        m1 = ctx.market.bars(sym, Timeframe.M1, window + lookback + 20)
        if not len(m1) or (until is not None and p["require_m5"] and int(m1.close_time[-1]) > int(until)):
            return []
        sell = bool(c["sell"])
        seq = m1_sequence(_candles(m1, sell), window, lookback, float(p["disp_atr"]), bool(p["require_bos"]))
        if seq is None:
            return []
        z_lo, z_hi = c["poi"]
        if not seq["sweep"] <= z_hi:  # 5a: the sweep reaches the POI
            return []
        sweep_at = int(m1.open_time[len(m1) + int(seq["sweep_bar"])])
        if ctx.state.get(f"used:{sym}") == sweep_at:
            return []
        spread = ctx.market.spread(sym)
        entry = -float(m1.bid_c[-1]) if sell else float(m1.ask_c[-1])
        stop = seq["sweep"] - float(p["stop_buffer_atr"]) * seq["atr"] - (spread if sell else 0.0)
        risk = entry - stop
        if not 0 < risk <= float(p["max_sl_atr"]) * float(c["h1_atr"]):
            return []  # 7: stop too wide
        above = [v for v in c["liq"] if v > entry]
        if not above:
            return []
        target = min(above)  # 8: the nearest real liquidity, chosen before the RR
        rr = (target - entry) / risk
        if rr < float(p["min_rr"]):
            return []
        ctx.state[f"used:{sym}"] = sweep_at
        ctx.state[f"hunt:{sym}"] = None
        sign = -1.0 if sell else 1.0
        return [
            ctx.signal(
                sym,
                "sell" if sell else "buy",
                sign * stop,
                target_price=sign * target,
                reason=(
                    f"SMC {'bearish' if sell else 'bullish'} bias, POI {sign * z_lo:.2f}-{sign * z_hi:.2f}, "
                    f"sweep {sign * seq['sweep']:.2f}, CHOCH {sign * seq['choch']:.2f}, "
                    f"{'BOS, ' if p['require_bos'] else ''}FVG retest, "
                    f"{'pin bar' if seq['confirm'] == 1.0 else 'engulfing'}, target liquidity "
                    f"{sign * target:.2f} ({rr:.1f}R)"
                ),
                tags={"setup": self.manifest.id, "rr": f"{rr:.2f}", "sweep": f"{sign * seq['sweep']:.5f}"},
            )
        ]
