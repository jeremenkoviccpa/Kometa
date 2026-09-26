"""smc_sniper 2.0: the owner's XAUUSD sniper method (docs/decisions.md 2026-09-26). A CANDIDATE: it has proven
nothing until validation, shadow and micro say so. Principle: WAIT -> LIQUIDITY -> REACTION -> STRUCTURE
SHIFT -> RETEST -> SNIPER ENTRY -> 1:2+. Any missing key element is NO TRADE; the rest is scored.

Shorts mirror longs exactly (prices negated with high and low swapped, so both run the same lines).

Hard rules (each one missing = NO TRADE):
1. BIAS (each H1 close): D1, H4 and H1 structure, each bull (higher highs and lows), bear, or neutral. The
   dominant bias is H4's, or D1's when H4 is neutral and H1 agrees with D1; D1 against H4 = NO TRADE.
2. LOCATION: longs only from discount (below 50% of the H4 dealing range, its last confirmed swing high and
   low), shorts only from premium. Never mid-range.
3. ZONE: demand below price in discount (supply for shorts): an unmitigated H1 order block that left a fair
   value gap, a tested H4 support (>= 2 touches, unbroken, +- `zone_atr` x H4 ATR), or the range's low.
4. 15M CONFIRMATION (each M15 close): in the last `window_m` minutes the M15 low swept the liquidity below
   (the lowest low of `liq_lookback` bars before it) at a zone, closed back above it (rejection), and a
   candle with a body of at least `disp_atr` x ATR closed through the last M15 swing high before the sweep
   (displacement + BOS/CHoCH) on this bar. That arms the 5M trigger for `window_m` minutes with the context
   as it was then (bias, range, ATR), so the rally that follows a real shift does not cancel it.
5. 5M TRIGGER (each M5 close after the M15 break): the M5 sweep, rejection, displacement and break are the
   inside of that M15 move; the trigger is the FIRST M5 pullback to the breakout area (the level the M15 BOS
   broke, within 0.1 M15 ATR) that closes up above it (the rejection), while no M5 close since the break
   has gone more than that below it. Never the second retest, never chasing.
6. RISK: stop below the relevant swing low, the retest's pullback low (the higher low whose loss would undo
   the shift), + `stop_buffer_atr` x M5 ATR, and at most `max_sl_atr` x H1 ATR away (never widened to
   survive). Target: the nearest liquidity above, before any obstacle; RR >= `min_rr` (2). Spread at most
   `max_spread_r` of the risk. Size is the risk gate's (the owner's cap: 1%).
7. NEWS: no entry within `news_min` minutes of a scheduled high-impact event in either currency.

SNIPER SCORE (the owner's weights, which sum to 95 as written; the owner has been told): D1 bias 5, H4 bias
10, H1 bias 10; clear structure 5, BOS/CHoCH 5, HTF confluence 5; liquidity target 5, confirmed sweep
10; supply/demand 5, support/resistance 5, location quality 5; displacement 5, retest 5, rejection 5,
EMA 9/20 5; RR >= 2 5. Below 70 NO TRADE, 70-79 WATCH
(recorded, not traded), 80-89 VALID, 90+ A+. Outside the London and New York sessions a trade needs
`offsession_score` (90). The score and checklist travel on the signal (and WATCH setups in the state).
SPEC-QUESTION: the owner's macro engine (DXY, yields) waits for those data series (open question 36).
"""

from __future__ import annotations

import math

import numpy as np

from autotrader.core.events import BarClosed
from autotrader.core.indicators import atr, ema, levels, swings, trend_state
from autotrader.core.models import Timeframe
from autotrader.core.series import BarsArray
from autotrader.strategies_api import Request, Strategy, StrategyContext

NS_MIN = 60_000_000_000
NS_DAY = 1440 * NS_MIN
Candles = tuple[list[float], list[float], list[float], list[float]]
SESSIONS = ((0, 7, "Asia"), (7, 12, "London"), (12, 17, "New York"))  # UTC hours
WATCH_KEEP = 5
RETEST_TOL = 0.1  # x M15 ATR: how close to the breakout level a pullback counts as its retest


def _candles(b: BarsArray, sell: bool) -> Candles:
    """Bid OHLC as float lists; for shorts negated with high and low swapped, so a short is a long."""
    if not sell:
        return b.bid_o.tolist(), b.bid_h.tolist(), b.bid_l.tolist(), b.bid_c.tolist()
    return (-b.bid_o).tolist(), (-b.bid_l).tolist(), (-b.bid_h).tolist(), (-b.bid_c).tolist()


def _atr(x: Candles) -> list[float]:
    _, h, lo, c = x
    return [float(v) for v in atr(np.array(h), np.array(lo), np.array(c), 14)]


def _last(v: np.ndarray) -> float:
    ok = v[~np.isnan(v)]
    return float(ok[-1]) if ok.size else math.nan


# ---------------------------------------------------------------- zones and liquidity


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


def support_zones(x: Candles, width_atr: float) -> list[tuple[float, float]]:
    """Tested, unbroken supports (>= 2 touches) below price as zones of +- width_atr x ATR (long frame)."""
    _, h, lo, c = x
    if len(c) < 30:
        return []
    a = _atr(x)[-1]
    if not a > 0:
        return []
    px = c[-1]
    book = levels(np.array(h), np.array(lo), np.array(c))[-1]
    return [
        (lv.price - width_atr * a, lv.price + width_atr * a)
        for lv in book
        if not lv.broken and lv.touches >= 2 and lv.price < px
    ]


def liquidity_above(x: Candles) -> list[float]:
    """Resting buy-side liquidity: untaken confirmed swing highs, plus equal highs as one pool."""
    _, h, lo, _ = x
    s = swings(np.array(h), np.array(lo)).high.tolist()
    n = len(h)
    top = [-math.inf] * (n + 1)  # highest high from index i on
    for i in range(n - 1, -1, -1):
        top[i] = max(h[i], top[i + 1])
    # a swing confirmed at t is the high of bar t-2 (left = right = 2)
    untaken = [v for t, v in enumerate(s) if not math.isnan(v) and top[t - 1] <= v]
    a = _atr(x)[-1] if n >= 15 else math.nan
    pools: list[float] = []
    if a > 0:  # equal highs: two untaken swing highs within 0.1 ATR rest as one pool at the higher
        srt = sorted(untaken)
        pools = [max(p, q) for p, q in zip(srt, srt[1:], strict=False) if q - p <= 0.1 * a]  # noqa: RUF007
    return untaken + pools


def calendar_levels(d1: Candles, d1_open_ns: list[int]) -> list[float]:
    """Previous day high and previous (ISO) week high, in the long frame."""
    _, h, _, _ = d1
    if not h:
        return []
    out = [h[-1]]
    weeks = [
        (t // NS_DAY + 3) // 7 for t in d1_open_ns
    ]  # Monday-based week number (1970-01-01 was a Thursday)
    prev = sorted(set(weeks))[-2] if len(set(weeks)) >= 2 else None
    if prev is not None and weeks[-1] != prev:
        out.append(max(v for v, w in zip(h, weeks, strict=True) if w == prev))
    return out


def session_highs(m15: Candles, open_ns: list[int]) -> list[float]:
    """Highs of today's finished sessions (Asia, London, New York by UTC hour), in the long frame."""
    _, h, _, _ = m15
    if not h:
        return []
    today = open_ns[-1] // NS_DAY
    hour_now = (open_ns[-1] % NS_DAY) / (60 * NS_MIN)
    out = []
    for start, end, _ in SESSIONS:
        if hour_now < end:
            continue
        xs = [
            v
            for v, t in zip(h, open_ns, strict=True)
            if t // NS_DAY == today and start <= (t % NS_DAY) / (60 * NS_MIN) < end
        ]
        if xs:
            out.append(max(xs))
    return out


# ---------------------------------------------------------------- the structure-shift sequence


def _pivot_high(h: list[float], k: int) -> bool:
    return h[k] > h[k - 1] and h[k] > h[k - 2] and h[k] >= h[k + 1] and h[k] >= h[k + 2]


def shift(x: Candles, start: int, lookback: int, disp: float) -> dict[str, float] | None:
    """Sweep -> rejection -> displacement -> BOS of the last swing high, the sweep at or after `start`: the
    sweep takes the lowest low of `lookback` bars and a close gets back above it within 3 bars; a candle
    from the sweep to the break has a body of at least `disp` x ATR; the break is the first close above the
    last swing high before the sweep. Returns the levels (long frame) and the break index, or None."""
    o, h, lo, c = x
    a = _atr(x)
    n = len(c)
    start = max(start, lookback + 3)
    if n - 2 < start:
        return None
    s = min(range(start, n - 1), key=lambda i: (lo[i], i))  # the sweep: the extreme since start
    swept = min(lo[s - lookback : s])
    if not (lo[s] < swept and max(c[s : s + 3]) > swept):  # sweep, rejection within 3 bars (one M15 on M5)
        return None
    level = next((h[k] for k in range(s - 2, s - lookback, -1) if _pivot_high(h, k)), None)
    if level is None:
        return None
    b = next((i for i in range(s + 1, n) if c[i] > level), None)  # BOS: the first close through
    if b is None:
        return None
    body = max(((c[i] - o[i]) / a[i] for i in range(s, b + 1) if a[i] > 0), default=0.0)
    if not body >= disp:  # displacement: a strong candle between the sweep and the break
        return None
    return {
        "sweep": lo[s],
        "swept": swept,
        "level": level,
        "break": float(b),
        "sweep_i": float(s),
        "atr": a[-1],
        "body_atr": body,
    }


def first_retest(x: Candles, level: float, brk: int, tol: float = 0.0) -> bool:
    """The last bar is the first one after bar `brk` to come back to the level (low within `tol` above it),
    and it closes up above the level (the rejection). No close since the break may be `tol` below it."""
    o, _, lo, c = x
    e = len(c) - 1
    if e <= brk or min(c[brk + 1 :]) < level - tol:  # the level must hold on closes
        return False
    if any(lo[i] <= level + tol for i in range(brk + 1, e)):  # an earlier retest: no chasing
        return False
    return lo[e] <= level + tol and c[e] > o[e] and c[e] > level


def score_setup(parts: dict[str, bool]) -> int:
    weights = {
        "d1_bias": 5, "h4_bias": 10, "h1_bias": 10,
        "clear_structure": 5, "bos_choch": 5, "htf_confluence": 5,
        "liquidity_target": 5, "sweep": 10,
        "supply_demand": 5, "support_resistance": 5, "location": 5,
        "displacement": 5, "retest": 5, "rejection": 5, "ema": 5,
        "rr": 5,
    }  # fmt: skip
    return sum(w for k, w in weights.items() if parts.get(k))


def quality(score: int) -> str:
    return (
        "A+ SNIPER"
        if score >= 90
        else "VALID SNIPER"
        if score >= 80
        else "WATCH"
        if score >= 70
        else "NO TRADE"
    )


def session_of(close_ns: int) -> str | None:
    hour = (close_ns % NS_DAY) / (60 * NS_MIN)
    return next((name for start, end, name in SESSIONS if start <= hour < end), None)


class SmcSniper(Strategy):
    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        w = {Timeframe.M5: 300, Timeframe.M15: 200, Timeframe.H1: 200, Timeframe.H4: 300, Timeframe.D1: 60}
        return {(s, tf): n for s in self.manifest.symbols for tf, n in w.items()}

    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        sym = event.symbol
        if event.timeframe == Timeframe.H1:
            self._htf(ctx, sym)
        elif event.timeframe == Timeframe.M15:
            self._m15(ctx, sym)
        elif event.timeframe == Timeframe.M5:
            return self._m5(ctx, sym)
        return []

    # ------------------------------------------------------------ H1: bias, location, zones, liquidity

    def _htf(self, ctx: StrategyContext, sym: str) -> None:
        ctx.state[f"ctx:{sym}"] = None
        m = ctx.market
        d1, h4, h1 = (
            m.bars(sym, Timeframe.D1, 60),
            m.bars(sym, Timeframe.H4, 300),
            m.bars(sym, Timeframe.H1, 200),
        )
        if min(len(d1), len(h4)) < 30 or len(h1) < 60:
            return
        b_d1, b_h4, b_h1 = (float(trend_state(x.bid_h, x.bid_l)[-1]) for x in (d1, h4, h1))
        if b_d1 != 0 and b_d1 == -b_h4:
            return  # 1: D1 against H4
        bias = b_h4 if b_h4 != 0 else (b_d1 if b_d1 != 0 and b_h1 == b_d1 else 0.0)
        if bias == 0:
            return  # 1: neutral
        sell = bias < 0
        sw = swings(h4.bid_h, h4.bid_l)
        r_hi, r_lo = _last(sw.high), _last(sw.low)
        if not r_hi > r_lo:
            return
        sign = -1.0 if sell else 1.0
        eq = sign * (r_hi + r_lo) / 2
        top, bottom = (-r_lo, -r_hi) if sell else (r_hi, r_lo)  # the range in the long frame
        h1c, h4c, d1c = _candles(h1, sell), _candles(h4, sell), _candles(d1, sell)
        px = h1c[3][-1]
        if not px < eq:
            return  # 2: not in discount (premium for shorts)
        p = ctx.params
        a4 = _atr(h4c)[-1]
        zones: list[list[float | str]] = [
            [z[0], z[1], "demand"] for z in order_blocks(h1c, float(p["disp_atr"]))
        ]
        zones += [[z[0], z[1], "support"] for z in support_zones(h4c, float(p["zone_atr"]))]
        if a4 > 0:
            zones.append(
                [bottom - float(p["zone_atr"]) * a4, bottom + float(p["zone_atr"]) * a4, "range low"]
            )
        zones = [z for z in zones if float(z[1]) <= eq and float(z[0]) < px]
        if not zones:
            return  # 3: no zone
        liq = [*liquidity_above(h1c), *calendar_levels(d1c, d1.open_time.tolist()), top]
        ctx.state[f"ctx:{sym}"] = {
            "sell": sell,
            "bias": {"d1": b_d1 * sign, "h4": b_h4 * sign, "h1": b_h1 * sign},  # +1 agrees, -1 against
            "eq": eq,
            "zones": zones,
            "poi": [max(float(z[0]) for z in zones), max(float(z[1]) for z in zones)],  # nearest, for the hub
            "liq": liq,
            "h1_atr": _atr(h1c)[-1],
        }

    # ------------------------------------------------------------ M15: the structure shift

    def _m15(self, ctx: StrategyContext, sym: str) -> None:
        c = ctx.state.get(f"ctx:{sym}")
        if not c:
            return
        p = ctx.params
        lookback, window = int(p["liq_lookback"]), int(p["window_m"]) // 15
        m15 = ctx.market.bars(sym, Timeframe.M15, window + lookback + 10)
        if len(m15) < lookback + 10:
            return
        x = _candles(m15, bool(c["sell"]))
        sh = shift(x, len(m15) - window, lookback, float(p["disp_atr"]))
        if sh is None or int(sh["break"]) != len(m15) - 1:
            return  # 4: no fresh M15 structure shift on this bar
        at_zone = [
            z for z in c["zones"] if float(z[0]) - float(c["h1_atr"]) * 0.25 <= sh["sweep"] <= float(z[1])
        ]
        if not at_zone:
            return  # 4: the sweep did not happen at a zone
        sweep_ns = int(m15.open_time[int(sh["sweep_i"])])
        if ctx.state.get(f"used:{sym}") == sweep_ns:
            return
        liq_extra = session_highs(x, m15.open_time.tolist())
        ctx.state[f"hunt:{sym}"] = {
            "until": int(m15.close_time[-1]) + int(p["window_m"]) * NS_MIN,
            "sweep_ns": sweep_ns,
            "sweep": sh["sweep"],
            "kinds": sorted({str(z[2]) for z in at_zone}),
            "liq": [*c["liq"], *liq_extra],
            "body_atr": sh["body_atr"],
            "level": sh["level"],  # the M15 level the BOS broke: the breakout area the M5 retest returns to
            "break_close_ns": int(m15.close_time[-1]),
            "tol": RETEST_TOL * float(sh["atr"]),
            # the context as it was at the shift: a rally out of discount after it must not cancel the trigger
            "sell": c["sell"],
            "bias": c["bias"],
            "eq": c["eq"],
            "h1_atr": c["h1_atr"],
        }

    # ------------------------------------------------------------ M5: trigger, risk, score

    def _m5(self, ctx: StrategyContext, sym: str) -> list[Request]:
        hunt = ctx.state.get(f"hunt:{sym}")
        if not hunt or ctx.my_positions(sym) or ctx.my_pending(sym):
            return []
        c = hunt  # the context as it was at the M15 shift
        p = ctx.params
        lookback = int(p["liq_lookback"])
        m5 = ctx.market.bars(sym, Timeframe.M5, int(p["window_m"]) // 5 + 3 * lookback + 20)
        if not len(m5) or int(m5.close_time[-1]) > int(hunt["until"]):
            return []
        sell = bool(c["sell"])
        x = _candles(m5, sell)
        brk = (
            int(np.searchsorted(m5.open_time, int(hunt["break_close_ns"]))) - 1
        )  # the last M5 bar of the break
        level = float(hunt["level"])
        if not first_retest(x, level, brk, float(hunt["tol"])):
            return []  # 5: no first retest of the breakout area (yet)
        nxt, last = ctx.market.minutes_to_news(sym)
        news_min = float(p["news_min"])
        if (nxt is not None and nxt <= news_min) or (last is not None and last <= news_min):
            return []  # 7: news near
        spread = ctx.market.spread(sym)
        entry = -float(m5.bid_c[-1]) if sell else float(m5.ask_c[-1])
        sweep = float(hunt["sweep"])
        atr5 = _atr(x)[-1]
        pullback_low = min(x[2][brk + 1 :])  # the higher low of the retest: the structure's invalidation
        stop = pullback_low - float(p["stop_buffer_atr"]) * atr5 - (spread if sell else 0.0)
        risk = entry - stop
        if not 0 < risk <= float(p["max_sl_atr"]) * float(c["h1_atr"]):
            return []  # 6: stop too wide (never widened to survive)
        if spread > float(p["max_spread_r"]) * risk:
            return []  # 6: spread eats the trade
        above = [float(v) for v in hunt["liq"] if float(v) > entry]
        if not above:
            return []
        target = min(above)  # 6: the nearest liquidity: no obstacle before it
        rr = (target - entry) / risk
        if rr < float(p["min_rr"]):
            return []  # 6
        closes = np.array(x[3])
        e9, e20 = float(ema(closes, 9)[-1]), float(ema(closes, 20)[-1])
        momentum = e9 > e20 or (len(closes) > 3 and closes[-1] > closes[-2] > closes[-3])
        kinds = set(hunt["kinds"])
        b = c["bias"]
        parts = {
            "d1_bias": b["d1"] > 0,
            "h4_bias": b["h4"] > 0,
            "h1_bias": b["h1"] > 0,
            "clear_structure": b["h4"] > 0 and b["h1"] > 0,
            "bos_choch": True,
            "htf_confluence": len(kinds) >= 2 or (b["d1"] > 0 and b["h4"] > 0),
            "liquidity_target": True,
            "sweep": True,
            "supply_demand": "demand" in kinds,
            "support_resistance": bool(kinds & {"support", "range low"}),
            "location": entry < float(c["eq"]),
            "displacement": float(hunt["body_atr"]) >= 1.5 * float(p["disp_atr"]),
            "retest": True,
            "rejection": True,
            "ema": momentum,
            "rr": True,
        }
        score = score_setup(parts)
        session = session_of(int(m5.close_time[-1]))
        need = float(p["min_score"]) if session in ("London", "New York") else float(p["offsession_score"])
        sign = -1.0 if sell else 1.0
        summary = {
            "direction": "SHORT" if sell else "LONG",
            "score": str(score),
            "quality": quality(score),
            "bias": f"D1 {b['d1'] * sign:+.0f} H4 {b['h4'] * sign:+.0f} H1 {b['h1'] * sign:+.0f}",
            "zone": ",".join(sorted(kinds)),
            "session": session or "off-session",
            "rr": f"{rr:.2f}",
            "sweep": f"{sign * sweep:.5f}",
            "missing": ",".join(k for k, v in parts.items() if not v) or "none",
        }
        ctx.state[f"used:{sym}"] = hunt["sweep_ns"]
        ctx.state[f"hunt:{sym}"] = None
        if score < need:
            if score >= 70:  # WATCH: recorded for the owner, not traded
                w = list(ctx.state.get(f"watch:{sym}") or [])[-(WATCH_KEEP - 1) :]
                ctx.state[f"watch:{sym}"] = [*w, {**summary, "at": int(m5.close_time[-1])}]
            return []
        return [
            ctx.signal(
                sym,
                "sell" if sell else "buy",
                sign * stop,
                target_price=sign * target,
                reason=(
                    f"{quality(score)} {summary['direction']} score {score}: bias {summary['bias']}, "
                    f"{summary['zone']} zone, sweep {sign * sweep:.2f}, M15 shift, M5 retest of "
                    f"{sign * level:.2f}, target liquidity {sign * target:.2f} ({rr:.1f}R), "
                    f"{summary['session']}, no news near"
                ),
                tags={"setup": "smc_sniper", **summary},
            )
        ]
