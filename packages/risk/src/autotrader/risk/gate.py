"""The risk gate (spec section 11): the only path from an order intent to the broker.

It never increases size: every check can only keep, reduce or reject. It sizes
independently (Decimal arithmetic, own risk fraction) and never trusts the
allocator's lot number. Every decision is signed (Ed25519) with a sequence
number and a short expiry; execution verifies all three.

Checks, in order:
 1. halt state allows entries
 2. stop present, on the correct side, at least min_stop_spreads x spread away
 3. independent position size
 4. per-trade, per-strategy and total open risk
 5. same-currency same-direction exposure
 6. leverage (total notional / equity) and per-symbol lot caps
 7. margin (free margin after the trade >= ratio x required)
 8. news blackout and weekend cutoff
 9. strategy stage allows trading; risk fraction within the stage limit
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from autotrader.core.hashing import canonical_json
from autotrader.core.indicators.sessions import EventIndex
from autotrader.core.models import (
    HaltCommand,
    HaltState,
    Instrument,
    OrderIntent,
    RiskDecision,
    Side,
    Stage,
)
from autotrader.core.signing import sign_decision, verify_bytes
from autotrader.core.timeutil import ensure_utc
from autotrader.risk.config import RiskLimits
from autotrader.risk.state import SEVERITY, RiskState, StateStore, escalate

TRADING_STAGES = {Stage.MICRO, Stage.LIVE, Stage.SCALED}
ZERO = Decimal(0)


@dataclass(frozen=True)
class Exposure:
    """An open position or a pending entry order, as the gate sees it."""

    symbol: str
    side: Side
    lots: Decimal
    entry: Decimal
    stop: Decimal | None
    strategy_id: str | None
    pending: bool = False
    external: bool = False


@dataclass(frozen=True)
class Snapshot:
    now: datetime
    equity: Decimal
    free_margin: Decimal
    bid: Decimal
    ask: Decimal
    exposures: Sequence[Exposure]
    instruments: Mapping[str, Instrument]
    to_account: Callable[[str], Decimal]  # quote currency -> account currency rate
    margin_per_lot: Mapping[str, Decimal]  # account currency, from the broker
    stages: Mapping[tuple[str, str], Stage]
    events: EventIndex | None = None


HaltActions = HaltCommand  # the bus message execution acts on (core.models)


@dataclass
class _Work:
    lots: Decimal
    reasons: list[str] = field(default_factory=list)
    rejected: bool = False

    def reject(self, why: str) -> None:
        self.rejected = True
        self.lots = ZERO
        self.reasons.append(why)

    def cap(self, limit: Decimal, why: str, step: Decimal, min_lot: Decimal) -> None:
        if self.rejected or self.lots <= limit:
            return
        capped = floor_step(max(limit, ZERO), step)
        if capped < min_lot:
            self.reject(f"{why}: no room")
        else:
            self.lots = capped
            self.reasons.append(f"resized: {why}")


def floor_step(x: Decimal, step: Decimal) -> Decimal:
    return (x / step).to_integral_value(rounding=ROUND_FLOOR) * step


class RiskGate:
    def __init__(
        self,
        limits: RiskLimits,
        limits_hash: str,
        decision_key: Ed25519PrivateKey,
        owner_key: Ed25519PublicKey,
        store: StateStore,
        on_decision: Callable[[RiskDecision, OrderIntent], None] | None = None,
    ) -> None:
        self.limits = limits
        self.limits_hash = limits_hash
        self._key = decision_key
        self._owner = owner_key
        self._store = store
        self.state: RiskState = store.load()
        self._on_decision = on_decision

    # ------------------------------------------------------------ exposure math

    def _risk_money(self, e: Exposure, snap: Snapshot) -> Decimal:
        inst = snap.instruments[e.symbol]
        fx = snap.to_account(inst.quote)
        if e.stop is None:
            # SPEC-QUESTION: risk of a stop-less (external) position; assume the larger of a 2% adverse
            # move and the per-trade maximum, so unknown risk is never counted as zero
            notional = e.lots * inst.contract_size * e.entry * fx
            return max(notional * Decimal("0.02"), snap.equity * self.limits.risk_per_trade_max)
        per_unit = (e.entry - e.stop) if e.side == "buy" else (e.stop - e.entry)
        return max(per_unit, ZERO) * inst.contract_size * e.lots * fx

    def _notional(self, symbol: str, lots: Decimal, price: Decimal, snap: Snapshot) -> Decimal:
        inst = snap.instruments[symbol]
        return lots * inst.contract_size * price * snap.to_account(inst.quote)

    # ------------------------------------------------------------ the decision

    def decide(self, intent: OrderIntent, snap: Snapshot) -> RiskDecision:
        now = ensure_utc(snap.now)
        sig = intent.signal
        lim = self.limits
        w = _Work(lots=intent.proposed_lots)
        inst = snap.instruments.get(sig.symbol)
        if inst is None:
            w.reject(f"unknown instrument {sig.symbol}")
            return self._finish(intent, w, now)
        step, min_lot = inst.lot_step, inst.min_lot
        fx = snap.to_account(inst.quote)
        stop = Decimal(str(sig.stop_price))
        if sig.entry_type == "market":
            entry = snap.ask if sig.side == "buy" else snap.bid
        else:
            entry = Decimal(str(sig.entry_price))
        spread = snap.ask - snap.bid

        # 1. halt state
        if self.state.halt != HaltState.NORMAL:
            w.reject(f"halted: {self.state.halt.value}")
        # 2. stop
        if not w.rejected:
            wrong = (sig.side == "buy" and stop >= entry) or (sig.side == "sell" and stop <= entry)
            if wrong:
                w.reject("stop on the wrong side of entry")
            elif abs(entry - stop) < lim.min_stop_spreads * spread:
                w.reject(f"stop closer than {lim.min_stop_spreads} x spread")
            elif snap.equity <= 0:
                w.reject("non-positive equity")
        dist = abs(entry - stop)
        stage = snap.stages.get((sig.strategy_id, sig.strategy_version))
        stage_limit = lim.stage_risk_limits.get(stage.value, ZERO) if stage else ZERO
        # 3. independent size: own risk fraction, never the allocator's lots
        if not w.rejected:
            rf = min(Decimal(str(intent.risk_fraction)), lim.risk_per_trade_max)
            if stage_limit > 0:
                rf = min(rf, stage_limit)
            own = floor_step(snap.equity * rf / (dist * inst.contract_size * fx), step)
            if own < min_lot:
                w.reject("size below min lot (never rounded up)")
            else:
                w.cap(own, "independent sizing", step, min_lot)
        per_lot_risk = dist * inst.contract_size * fx
        # 4. open risk: per trade, per strategy, total (pending entries count)
        if not w.rejected:
            w.cap(
                floor_step(snap.equity * lim.risk_per_trade_max / per_lot_risk, step),
                "per-trade risk",
                step,
                min_lot,
            )
            strat = sum(
                (
                    self._risk_money(e, snap)
                    for e in snap.exposures
                    if e.strategy_id == sig.strategy_id and not e.external
                ),
                ZERO,
            )
            room = snap.equity * lim.open_risk_per_strategy_max - strat
            w.cap(floor_step(room / per_lot_risk, step), "strategy open risk", step, min_lot)
            total = sum((self._risk_money(e, snap) for e in snap.exposures), ZERO)
            room = snap.equity * lim.open_risk_total_max - total
            w.cap(floor_step(room / per_lot_risk, step), "total open risk", step, min_lot)
        # 5. same currency, same direction
        if not w.rejected:
            new_dirs = _currency_dirs(inst, sig.side)
            for ccy, direction in new_dirs.items():
                n = 0
                for e in snap.exposures:
                    other = snap.instruments.get(e.symbol)
                    if other is not None and _currency_dirs(other, e.side).get(ccy) == direction:
                        n += 1
                if n >= lim.same_currency_same_direction_max_positions:
                    w.reject(f"{n} positions already {'long' if direction > 0 else 'short'} {ccy}")
                    break
        # 6. leverage and per-symbol lot cap
        if not w.rejected:
            notional = sum((self._notional(e.symbol, e.lots, e.entry, snap) for e in snap.exposures), ZERO)
            room = snap.equity * lim.leverage_notional_max - notional
            w.cap(floor_step(room / (inst.contract_size * entry * fx), step), "leverage", step, min_lot)
            on_symbol = sum((e.lots for e in snap.exposures if e.symbol == sig.symbol), ZERO)
            w.cap(lim.max_lots(sig.symbol) - on_symbol, "symbol lot cap", step, min_lot)
            w.cap(inst.max_lot, "broker max lot", step, min_lot)
        # 7. margin
        if not w.rejected:
            mpl = snap.margin_per_lot.get(sig.symbol)
            if mpl is None or mpl <= 0:
                w.reject("no margin data for symbol")
            else:
                # free_margin - req >= ratio * req  <=>  req <= free_margin / (ratio + 1)
                w.cap(
                    floor_step(snap.free_margin / ((lim.min_free_margin_ratio + 1) * mpl), step),
                    "margin",
                    step,
                    min_lot,
                )
        # 8. news blackout and weekend cutoff
        if not w.rejected:
            if snap.events is not None and snap.events.in_blackout(
                now, (inst.base, inst.quote), lim.news_blackout_minutes
            ):
                w.reject("news blackout")
            elif self._after_weekly_cutoff(now):
                w.reject("no new entries after the weekly cutoff")
        # 9. stage (demo_only trades only if the signed config grants it a limit, and at minimum size)
        if not w.rejected:
            trading = TRADING_STAGES | ({Stage.DEMO_ONLY} if stage_limit > 0 else set())
            if stage == Stage.DEMO_ONLY:
                w.cap(min_lot, "demo_only: minimum size", step, min_lot)
            if stage not in trading:
                w.reject(f"stage {stage.value if stage else 'unknown'} does not trade")
            elif Decimal(str(intent.risk_fraction)) > stage_limit:
                w.reasons.append(f"risk fraction capped to stage limit {stage_limit}")
        return self._finish(intent, w, now)

    def reject(self, intent: OrderIntent, reason: str, now: datetime) -> RiskDecision:
        """A signed rejection for an intent the gate cannot evaluate (fail closed)."""
        w = _Work(lots=ZERO)
        w.reject(reason)
        return self._finish(intent, w, ensure_utc(now))

    def _after_weekly_cutoff(self, now: datetime) -> bool:
        c = self.limits.no_new_entries_after
        local = now.astimezone(ZoneInfo(c.tz))
        return local.weekday() > c.weekday or (local.weekday() == c.weekday and local.time() >= c.at)

    def _finish(self, intent: OrderIntent, w: _Work, now: datetime) -> RiskDecision:
        lots = ZERO if w.rejected else min(w.lots, intent.proposed_lots)  # can only reduce
        if lots <= 0:
            verdict = "reject"
            lots = ZERO
        else:
            verdict = "approve" if lots == intent.proposed_lots else "resize"
        self.state.sequence += 1
        self._store.save(self.state)
        d = RiskDecision(
            intent_id=intent.intent_id,
            verdict=verdict,
            approved_lots=lots,
            reasons=tuple(w.reasons),
            limits_snapshot_hash=self.limits_hash,
            decided_at=now,
            expires_at=now + timedelta(seconds=self.limits.decision_ttl_seconds),
            sequence=self.state.sequence,
        )
        signed = sign_decision(self._key, d)
        if self._on_decision:
            self._on_decision(signed, intent)
        return signed

    # ------------------------------------------------------------ halts

    def _enter(self, new: HaltState, reason: str, now: datetime) -> HaltActions | None:
        if not escalate(self.state, new, reason, now):
            return None
        self._store.save(self.state)
        return HaltCommand.for_state(new, reason, now)

    def losses(self, equity: Decimal) -> dict[str, Decimal | None]:
        """Loss from each stored reference as a fraction of it (None before the first reference).
        The one formula for the halts below and for anything that displays how close they are."""
        s = self.state

        def frac(ref: Decimal | None) -> Decimal | None:
            return (ref - equity) / ref if ref else None

        return {
            "peak": frac(s.peak_eod_equity),
            "week": frac(s.week_ref_equity),
            "day": frac(s.day_ref_equity),
        }

    def usage(self, equity: Decimal) -> dict[str, Any]:
        def _f(x: Decimal | None) -> float | None:
            return None if x is None else float(x)

        """Read-only view for the hub: each loss halt's limit and the current loss against it."""
        lim, loss = self.limits, self.losses(equity)
        return {
            "halts": [
                {
                    "name": name,
                    "halt": state,
                    "limit": float(limit),
                    "loss": _f(loss[k]),
                }
                for k, name, state, limit in (
                    ("day", "Daily loss", "DAILY_HALT", lim.daily_loss_halt),
                    ("week", "Weekly loss", "WEEKLY_HALT", lim.weekly_loss_halt),
                    ("peak", "Drawdown from peak", "FULL_HALT", lim.peak_drawdown_full_halt),
                )
            ],
            "limits": json.loads(lim.model_dump_json()),
            "limits_hash": self.limits_hash,
            "sequence": self.state.sequence,
        }

    def on_account(self, equity: Decimal, now: datetime) -> HaltActions | None:
        """Call on every account update. Checks loss limits against the stored references."""
        lim, loss = self.limits, self.losses(equity)
        now = ensure_utc(now)
        if loss["peak"] is not None and loss["peak"] >= lim.peak_drawdown_full_halt:
            return self._enter(HaltState.FULL_HALT, "peak drawdown limit", now)
        if loss["week"] is not None and loss["week"] >= lim.weekly_loss_halt:
            return self._enter(HaltState.WEEKLY_HALT, "weekly loss limit", now)
        if loss["day"] is not None and loss["day"] >= lim.daily_loss_halt:
            return self._enter(HaltState.DAILY_HALT, "daily loss limit", now)
        return None

    def roll_day(self, equity: Decimal, now: datetime, *, new_week: bool) -> None:
        """At the 17:00 New York rollover: new loss references; expired daily/weekly halts clear."""
        s = self.state
        s.peak_eod_equity = max(s.peak_eod_equity or equity, equity)
        s.day_ref_equity = equity
        if s.halt == HaltState.DAILY_HALT:
            s.halt, s.halt_reason = HaltState.NORMAL, ""
        if new_week:
            s.week_ref_equity = equity
            if s.halt == HaltState.WEEKLY_HALT:
                s.halt, s.halt_reason = HaltState.NORMAL, ""
        self._store.save(s)

    def enter_recon_halt(self, reason: str, now: datetime) -> HaltActions | None:
        return self._enter(HaltState.RECON_HALT, reason, ensure_utc(now))

    def clear_recon_halt(self) -> bool:
        if self.state.halt != HaltState.RECON_HALT:
            return False
        self.state.halt, self.state.halt_reason = HaltState.NORMAL, ""
        self._store.save(self.state)
        return True

    def enter_full_halt(self, reason: str, now: datetime) -> HaltActions | None:
        return self._enter(HaltState.FULL_HALT, reason, ensure_utc(now))

    def resume(self, token: Mapping[str, str], signature_hex: str, now: datetime) -> bool:
        """Clear FULL_HALT with an owner-signed, unexpired, unused resume token."""
        if self.state.halt != HaltState.FULL_HALT:
            return False
        if token.get("action") != "resume_full_halt":
            return False
        if not verify_bytes(self._owner, canonical_json(dict(token)).encode(), signature_hex):
            return False
        if ensure_utc(datetime.fromisoformat(token["expires_at"])) < ensure_utc(now):
            return False
        nonce = token["nonce"]
        if nonce in self.state.used_nonces:
            return False
        self.state.used_nonces.append(nonce)
        self.state.halt, self.state.halt_reason = HaltState.NORMAL, ""
        self._store.save(self.state)
        return True

    @property
    def halted(self) -> bool:
        return SEVERITY[self.state.halt] > 0


def _currency_dirs(inst: Instrument, side: Side) -> dict[str, int]:
    """+1 long / -1 short per currency: buying EURUSD is long EUR and short USD."""
    s = 1 if side == "buy" else -1
    return {inst.base: s, inst.quote: -s}
