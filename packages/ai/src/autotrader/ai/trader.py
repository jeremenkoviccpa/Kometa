"""ai-trader: Claude trades the owner's SMC method in two tracks, each switched on and off in the hub.

- claude_smc_judge: a private copy of the coded rules (smc_sniper) runs on the live bars; every setup it
  finds goes to Claude, who takes or skips it (and may tighten the stop or choose the target).
- claude_smc_free: on every M15 close Claude reads D1..M1 and may open a trade of its own.

A track calls Claude only while the owner has it trading (paper), with no position open, at most one call
in flight, the free track at most once per `free_every_s` of wall time, and all tracks together within
`max_calls_per_day`. Claude's answer is checked in code (decision.check) at the price the order would
really get; what passes goes to the allocator and the risk gate like any strategy's signal. Every decision
(skips, refusals and errors too) is kept for the hub and appended to a JSONL log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from autotrader.ai.claude import Decider
from autotrader.ai.decision import AiDecision, check
from autotrader.ai.prompts import FREE, JUDGE, METHOD
from autotrader.core.broker import Quote
from autotrader.core.bus import ACCOUNT, QUOTES, SIGNALS, STAGES, Bus, Handler
from autotrader.core.clock import Clock
from autotrader.core.events import (
    AccountUpdate,
    BarClosed,
    QuoteUpdate,
    SignalEmitted,
    StageChanged,
    StageSnapshot,
)
from autotrader.core.indicators import atr
from autotrader.core.models import Signal, Stage, Timeframe
from autotrader.core.series import BarsArray
from autotrader.engine.context import signal_uuid
from autotrader.engine.live import LiveRunner
from autotrader.engine.live_bars import LiveBarBuilder
from autotrader.engine.market import NewsFn
from autotrader.engine.service import MONEY
from autotrader.strategies_api.base import PositionView, Strategy

log = logging.getLogger(__name__)

JUDGE_ID, FREE_ID = "claude_smc_judge", "claude_smc_free"
CANDLES = {
    Timeframe.D1: 30,
    Timeframe.H4: 40,
    Timeframe.H1: 48,
    Timeframe.M15: 48,
    Timeframe.M5: 36,
    Timeframe.M1: 60,
}


@dataclass(frozen=True)
class AiConfig:
    model: str
    max_calls_per_day: int = 200
    free_every_s: float = 900.0  # wall clock: the simulated market runs faster than real time
    max_age: timedelta = timedelta(minutes=15)  # market time from the question to the order


class AiTraderService:
    name = "ai-trader"

    def __init__(
        self,
        bus: Bus,
        clock: Clock,
        *,
        decider: Decider | None,
        versions: Mapping[str, str],
        rules: type[Strategy],
        config: AiConfig,
        history: Mapping[tuple[str, Timeframe], BarsArray] | None = None,
        log_path: Path | None = None,
        env: str = "dev",
        wall: Callable[[], float] = time.monotonic,
        disabled_reason: str = "",
        news_fn: NewsFn | None = None,
    ) -> None:
        self.bus, self.clock, self.cfg = bus, clock, config
        self.versions = dict(versions)
        if env == "live":  # the tracks are for the demo account; this is not a switch the hub can flip
            decider, disabled_reason = None, "Claude never trades real money (AT_ENV=live)"
        self.decider = decider
        self.disabled_reason = disabled_reason if decider is None else ""
        self.wall = wall
        self.log_path = log_path
        self.runner = LiveRunner(
            rules,
            spread_fn=self._spread,
            positions_fn=lambda _sid, sym: self._positions(JUDGE_ID, sym),
            pending_fn=lambda _sid, _sym: [],
            history=history,
            news_fn=news_fn,
        )
        self.builder = LiveBarBuilder(self.runner.subs)
        self.symbols = list(dict.fromkeys(s for s, _ in self.runner.subs))
        self.stages: dict[tuple[str, str], Stage] = {}
        self.account: AccountUpdate | None = None
        self.quotes: dict[str, Quote] = {}
        self.busy: set[str] = set()
        self.calls: deque[float] = deque()
        self.last_free = -math.inf
        self.seq = 0
        self.decisions: deque[dict[str, Any]] = deque(maxlen=60)
        self.tasks: set[asyncio.Task[None]] = set()

    def handlers(self) -> Mapping[str, Handler]:
        return {STAGES: self._on_stage, ACCOUNT: self._on_account, QUOTES: self._on_quote}

    # ------------------------------------------------------------ state

    def trading(self, track: str) -> bool:
        return self.stages.get((track, self.versions[track])) in MONEY

    def _spread(self, symbol: str, _ns: int) -> float:
        q = self.quotes.get(symbol)
        return float(q.ask - q.bid) if q is not None else 0.0

    def _positions(self, track: str, symbol: str | None) -> list[PositionView]:
        if self.account is None:
            return []
        return [
            PositionView(
                e.position_id or "",
                "",
                e.symbol,
                e.side,
                float(e.lots),
                float(e.entry),
                float(e.stop) if e.stop is not None else 0.0,
                None,
                self.account.at,
            )
            for e in self.account.exposures
            if e.strategy_id == track and (symbol is None or e.symbol == symbol)
        ]

    async def _on_stage(self, msg: Any) -> None:
        if isinstance(msg, StageSnapshot):
            self.stages = {(sid, v): st for sid, v, st in msg.stages}
        elif isinstance(msg, StageChanged):
            self.stages[(msg.strategy_id, msg.strategy_version)] = msg.to_stage

    async def _on_account(self, msg: Any) -> None:
        if isinstance(msg, AccountUpdate):
            self.account = msg

    async def _on_quote(self, msg: Any) -> None:
        if not isinstance(msg, QuoteUpdate):
            return
        q = Quote(symbol=msg.symbol, bid=Decimal(str(msg.bid)), ask=Decimal(str(msg.ask)), time=msg.at)
        self.quotes[q.symbol] = q
        self.on_bars(self.builder.on_quote(q))

    async def on_time(self, now: datetime) -> None:
        self.on_bars(self.builder.on_time(now))

    def on_bars(self, events: list[BarClosed]) -> None:
        for req in self.runner.on_bars(events):
            if isinstance(req, Signal):
                self.ask(JUDGE_ID, req.symbol, req)
        for e in events:
            if e.timeframe == Timeframe.M15:
                self.ask(FREE_ID, e.symbol, None)

    # ------------------------------------------------------------ asking Claude

    async def settle(self) -> None:
        """Wait for every question in flight (the simulated demo holds its clock meanwhile)."""
        while self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)

    def blocked(self, track: str, symbol: str) -> str | None:
        """Why this track may not ask Claude now, or None."""
        if not self.trading(track):
            return "switched off"
        if self.decider is None:
            return self.disabled_reason or "no Anthropic API key (AT_ANTHROPIC_API_KEY)"
        if track in self.busy:
            return "still thinking about the last one"
        if self._positions(track, symbol):
            return "a position is open"
        now = self.wall()
        while self.calls and now - self.calls[0] > 86_400:
            self.calls.popleft()
        if len(self.calls) >= self.cfg.max_calls_per_day:
            return f"daily budget of {self.cfg.max_calls_per_day} calls used"
        if track == FREE_ID and now - self.last_free < self.cfg.free_every_s:
            return "waiting for the next read"
        return None

    def ask(self, track: str, symbol: str, candidate: Signal | None) -> None:
        why = self.blocked(track, symbol)
        if why is not None:
            if candidate is not None and why != "switched off":
                self._record(track, "not asked", candidate.side, detail=why)
            return
        now = self.wall()
        self.calls.append(now)
        if track == FREE_ID:
            self.last_free = now
        self.busy.add(track)
        task = asyncio.get_running_loop().create_task(
            self.decide(track, symbol, candidate, self.context(track, symbol, candidate), self.clock.now())
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def decide(
        self, track: str, symbol: str, candidate: Signal | None, context: str, asked_at: datetime
    ) -> None:
        decider = self.decider
        try:
            try:
                if decider is None:
                    raise RuntimeError("no Claude client")
                d = AiDecision.model_validate(
                    await decider(METHOD + "\n" + (JUDGE if candidate else FREE), context)
                )
            except Exception as e:  # the API, the network or a malformed answer: no trade
                log.warning("ai-trader %s: %s", track, e)
                self._record(track, "error", None, detail=f"{type(e).__name__}: {e}"[:300])
                return
            await self.act(track, symbol, candidate, d, asked_at)
        finally:
            self.busy.discard(track)

    async def act(
        self, track: str, symbol: str, candidate: Signal | None, d: AiDecision, asked_at: datetime
    ) -> None:
        if d.action == "skip":
            self._record(track, "skip", d.side, d=d)
            return
        now = self.clock.now()
        if not self.trading(track):
            self._record(track, "refused", d.side, d=d, detail="switched off while Claude was thinking")
            return
        if now - asked_at > self.cfg.max_age:
            self._record(track, "refused", d.side, d=d, detail="the answer came too late for this price")
            return
        q = self.quotes.get(symbol)
        limits = self.limits(symbol)
        if q is None or limits is None:
            self._record(track, "refused", d.side, d=d, detail="no price")
            return
        sweep = candidate.tags.get("sweep") if candidate is not None else None
        res = check(
            d,
            float(q.bid),
            float(q.ask),
            min_rr=limits[0],
            max_stop=limits[1],
            side=candidate.side if candidate is not None else None,
            sweep=float(sweep) if sweep is not None else None,
        )
        if isinstance(res, str):
            self._record(track, "refused", d.side, d=d, detail=res)
            return
        self.seq += 1
        version = self.versions[track]
        sig = Signal(
            signal_id=signal_uuid(track, version, int(now.timestamp() * 1e9), self.seq),
            strategy_id=track,
            strategy_version=version,
            symbol=symbol,
            side=res.side,
            entry_type="market",
            entry_price=None,
            stop_price=res.stop,
            target_price=res.target,
            created_at=now,
            reason=f"Claude ({res.rr:.1f}R, confidence {d.confidence:.0%}): {d.reasoning}"[:1000],
            tags={"ai": self.cfg.model, "track": track, "confidence": f"{d.confidence:.2f}"},
        )
        await self.bus.publish(SIGNALS, SignalEmitted(at=now, signal=sig, timeframe=Timeframe.M1))
        self._record(track, "take", res.side, d=d, rr=res.rr, detail=f"entry about {res.entry:.2f}")

    def limits(self, symbol: str) -> tuple[float, float] | None:
        """(minimum RR, widest stop in price) from the coded method's own parameters and the H1 ATR."""
        h1 = self.runner.market.bars(symbol, Timeframe.H1, 60)
        if len(h1) < 20:
            return None
        a = float(atr(h1.bid_h, h1.bid_l, h1.bid_c, 14)[-1])
        p = self.runner.params
        return float(p["min_rr"]), float(p["max_sl_atr"]) * a

    # ------------------------------------------------------------ what Claude sees

    def context(self, track: str, symbol: str, candidate: Signal | None) -> str:
        m = self.runner.market
        q = self.quotes.get(symbol)
        limits = self.limits(symbol)
        payload: dict[str, Any] = {
            "time_utc": self.clock.now().isoformat(),
            "symbol": symbol,
            "bid": float(q.bid) if q else None,
            "ask": float(q.ask) if q else None,
            "hard_limits": {"min_rr": limits[0], "max_stop_distance": round(limits[1], 2)}
            if limits
            else None,
            "rules_engine": self.rules_view(symbol),
            "open_positions": [
                p.__dict__ | {"opened_at": str(p.opened_at)} for p in self._positions(track, symbol)
            ],
        }
        if candidate is not None:
            payload["candidate"] = {
                "side": candidate.side,
                "stop": candidate.stop_price,
                "target": candidate.target_price,
                "why": candidate.reason,
                "tags": candidate.tags,
            }
        payload["candles_bid_ohlc"] = {
            tf.value: [
                [
                    datetime.fromtimestamp(int(b.open_time[i]) / 1e9, UTC).strftime("%m-%d %H:%M"),
                    *(round(float(x[i]), 2) for x in (b.bid_o, b.bid_h, b.bid_l, b.bid_c)),
                ]
                for i in range(len(b))
            ]
            for tf, n in CANDLES.items()
            if (symbol, tf) in self.runner.subs  # the timeframes the rules use
            for b in [m.bars(symbol, tf, n)]
        }
        return json.dumps(payload, separators=(",", ":"), default=str)

    def rules_view(self, symbol: str) -> dict[str, Any] | str:
        c = self.runner.ctx.state.get(f"ctx:{symbol}")
        if not c:
            return "no agreed D1/H4 bias with a POI in discount/premium right now"
        s = -1.0 if c["sell"] else 1.0  # the rules keep levels in a long-only frame
        return {
            "bias": "bearish" if c["sell"] else "bullish",
            "equilibrium_50pct": round(s * c["eq"], 2),
            "poi": sorted(round(s * v, 2) for v in c["poi"]),
            "liquidity_targets": sorted(round(s * v, 2) for v in c["liq"]),
            "h1_atr": round(c["h1_atr"], 2),
        }

    # ------------------------------------------------------------ the record

    def _record(
        self,
        track: str,
        action: str,
        side: str | None,
        *,
        d: AiDecision | None = None,
        rr: float | None = None,
        detail: str = "",
    ) -> None:
        row = {
            "at": self.clock.now().isoformat(),
            "track": track,
            "action": action,
            "side": side,
            "stop": d.stop if d else None,
            "target": d.target if d else None,
            "rr": round(rr, 2) if rr is not None else None,
            "confidence": d.confidence if d else None,
            "reasoning": d.reasoning if d else "",
            "checklist": d.checklist if d else {},
            "detail": detail,
        }
        self.decisions.append(row)
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def view(self) -> dict[str, Any]:
        return {
            "enabled": self.decider is not None,
            "disabled_reason": "" if self.decider is not None else (self.disabled_reason or "no API key"),
            "model": self.cfg.model,
            "calls_last_24h": len(self.calls),
            "max_calls_per_day": self.cfg.max_calls_per_day,
            "tracks": {t: self.trading(t) for t in self.versions},
            "thinking": sorted(self.busy),
            "decisions": list(reversed(self.decisions)),
        }
