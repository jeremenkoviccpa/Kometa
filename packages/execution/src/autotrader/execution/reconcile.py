"""Reconciliation (spec section 12): broker truth vs execution's journal, every 60 seconds.

Any mismatch enters RECON_HALT and alerts. One automatic resync follows. It resolves a mismatch only
when the broker's own records explain it:
- an order we sent (its client order id is in the journal) is adopted at the broker;
- a sent order that never reached the broker, and has no deal, is marked failed;
- a position gone from the broker with closing deals for all its lots is marked closed;
- a pending order gone with an entry deal becomes open; gone after its expiry becomes expired;
- a missing stop is set again, or the position is closed.
Anything else stays mismatched: the halt stays until the owner runs an owner resync, which adopts the
broker's state as it is. An unreachable broker also halts, and clears on the next clean check.

Positions and orders that do not carry a system client order id are external: flagged, alerted
once, left alone, and still counted in exposure by the risk gate.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal, Protocol

from autotrader.core.alerts import Severity
from autotrader.core.broker import BrokerDeal, BrokerOrder, BrokerPosition, is_system_comment
from autotrader.core.models import Frozen, HaltCommand, UtcDatetime
from autotrader.execution.adapter import BrokerUnavailableError
from autotrader.execution.journal import ReconEpisode, TrackedOrder
from autotrader.execution.order_manager import OrderManager, stop_protects

MismatchKind = Literal[
    "broker_unreachable",
    "unconfirmed_send",
    "missing_position",
    "missing_order",
    "untracked_position",
    "untracked_order",
    "lots",
    "missing_stop",
    "balance",
]
ZERO = Decimal(0)


class RiskControl(Protocol):
    """What reconciliation needs from the risk gate (a bus client in production, the gate in tests)."""

    def enter_recon_halt(self, reason: str, now: datetime) -> HaltCommand | None: ...

    def clear_recon_halt(self) -> bool: ...


class Mismatch(Frozen):
    kind: MismatchKind
    ref: str
    detail: str


class ReconReport(Frozen):
    at: UtcDatetime
    mismatches: tuple[Mismatch, ...]
    external: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches


@dataclass
class _BrokerView:
    positions: list[BrokerPosition]
    orders: list[BrokerOrder]
    balance: Decimal


class Reconciler:
    def __init__(self, om: OrderManager, risk: RiskControl) -> None:
        self.om = om
        self.risk = risk

    @property
    def episode(self) -> ReconEpisode:
        return self.om.state.recon_episode

    @episode.setter
    def episode(self, value: ReconEpisode) -> None:
        if value != self.om.state.recon_episode:
            self.om.state.recon_episode = value
            self.om.save()

    # ------------------------------------------------------------ comparison

    async def _view(self) -> _BrokerView:
        acct = await self.om.adapter.account()
        return _BrokerView(
            await self.om.adapter.open_positions(), await self.om.adapter.pending_orders(), acct.balance
        )

    async def check(self) -> ReconReport:
        om = self.om
        now = om.clock.now()
        try:
            await om.sync_deals()
            view = await self._view()
        except BrokerUnavailableError as e:
            return ReconReport(
                at=now, mismatches=(Mismatch(kind="broker_unreachable", ref="", detail=str(e)),), external=()
            )
        out: list[Mismatch] = []
        positions = {p.position_id: p for p in view.positions}
        orders = {o.broker_order_id: o for o in view.orders}
        grace = timedelta(seconds=om.cfg.send_grace_seconds)
        for t in om.state.active():
            ref = t.client_order_id
            if t.state == "sending":
                if now - t.created_at > grace:
                    out.append(Mismatch(kind="unconfirmed_send", ref=ref, detail="sent, never confirmed"))
            elif t.state == "pending":
                if t.broker_order_id not in orders:
                    out.append(Mismatch(kind="missing_order", ref=ref, detail=f"order {t.broker_order_id}"))
            elif t.position_id not in positions:
                out.append(Mismatch(kind="missing_position", ref=ref, detail=f"position {t.position_id}"))
            else:
                p = positions[t.position_id]
                if p.lots != t.lots:
                    out.append(Mismatch(kind="lots", ref=ref, detail=f"broker {p.lots} vs journal {t.lots}"))
                if not stop_protects(t.side, p.sl, t.sl):
                    out.append(
                        Mismatch(kind="missing_stop", ref=ref, detail=f"broker sl {p.sl}, want {t.sl}")
                    )
        external: list[str] = []
        for p in view.positions:
            if not is_system_comment(p.comment):
                external.append(f"position:{p.position_id}")
            elif om.state.by_position(p.position_id) is None:
                out.append(
                    Mismatch(kind="untracked_position", ref=p.comment, detail=f"position {p.position_id}")
                )
        for o in view.orders:
            if not is_system_comment(o.comment):
                external.append(f"order:{o.broker_order_id}")
            elif om.state.by_broker_order(o.broker_order_id) is None:
                out.append(
                    Mismatch(kind="untracked_order", ref=o.comment, detail=f"order {o.broker_order_id}")
                )
        expected = om.state.expected_balance
        if expected is not None and abs(view.balance - expected) > om.cfg.balance_tolerance:
            out.append(
                Mismatch(kind="balance", ref="", detail=f"broker {view.balance} vs expected {expected}")
            )
        self._flag_external(external)
        return ReconReport(at=now, mismatches=tuple(out), external=tuple(external))

    def _flag_external(self, external: list[str]) -> None:
        new = [e for e in external if e not in self.om.state.alerted_external]
        for e in new:
            self.om.alert(Severity.CRITICAL, "external_position", f"opened outside the system: {e}", ref=e)
            self.om.state.alerted_external.append(e)
        if new:
            self.om.save()

    # ------------------------------------------------------------ the 60 second job

    async def run_once(self) -> ReconReport:
        """The episode is persisted before the halt is entered (write-ahead), so after a crash or
        restart a clean check still knows it must clear the halt it caused."""
        om = self.om
        report = await self.check()
        now = om.clock.now()
        if report.ok:
            if self.episode in ("unreachable", "resyncing"):
                if self.risk.clear_recon_halt():
                    om.alert(Severity.INFO, "recon_resolved", "broker reachable and state matches")
                self.episode = "clean"
            return report
        if self.episode == "stuck":
            return report  # already halted and alerted; waits for the owner
        summary = "; ".join(f"{m.kind} {m.ref}".strip() for m in report.mismatches)
        unreachable = all(m.kind == "broker_unreachable" for m in report.mismatches)
        previous = self.episode
        self.episode = "unreachable" if unreachable else "resyncing"
        halt = self.risk.enter_recon_halt(f"reconciliation: {summary}", now)
        if halt is not None:
            await self._apply(halt)
        if unreachable:
            if previous != "unreachable":
                om.alert(Severity.CRITICAL, "recon_mismatch", f"broker unreachable: {summary}")
            return report
        om.alert(Severity.CRITICAL, "recon_mismatch", summary)
        with contextlib.suppress(BrokerUnavailableError):  # the re-check below reports it
            await self.resync(report)
        again = await self.check()
        if again.ok:
            self.risk.clear_recon_halt()
            self.episode = "clean"
            om.alert(Severity.INFO, "recon_resolved", "resync explained every mismatch")
        elif all(m.kind == "broker_unreachable" for m in again.mismatches):
            self.episode = "unreachable"  # lost the broker mid-resync: retry when it is back
        else:
            self.episode = "stuck"
            om.alert(
                Severity.CRITICAL,
                "recon_stuck",
                "still mismatched after resync; staying halted until an owner resync: "
                + "; ".join(f"{m.kind} {m.ref}".strip() for m in again.mismatches),
            )
        return again

    async def _apply(self, halt: HaltCommand) -> None:
        try:
            await self.om.apply_halt(halt)
        except BrokerUnavailableError as e:
            self.om.alert(Severity.CRITICAL, "halt_close_failed", f"could not act on halt: {e}")

    async def owner_resync(self) -> ReconReport:
        """Owner command: adopt the broker's state as it is, then clear the halt if everything matches."""
        report = await self.check()
        await self.resync(report, adopt_all=True)
        again = await self.check()
        if again.ok:
            self.risk.clear_recon_halt()
            self.episode = "clean"
            self.om.alert(Severity.INFO, "recon_resolved", "owner resync")
        return again

    # ------------------------------------------------------------ resync

    async def resync(self, report: ReconReport, *, adopt_all: bool = False) -> None:
        om = self.om
        if report.ok:
            return
        view = await self._view()
        oldest = min((t.created_at for t in om.state.active()), default=om.clock.now())
        deals = await om.adapter.deals(oldest - timedelta(seconds=om.cfg.deal_lookback_seconds))
        positions = {p.position_id: p for p in view.positions}
        by_comment_pos = {p.comment: p for p in view.positions if is_system_comment(p.comment)}
        by_comment_ord = {o.comment: o for o in view.orders if is_system_comment(o.comment)}
        for m in report.mismatches:
            t = om.state.orders.get(m.ref)
            if m.kind in ("unconfirmed_send", "untracked_position", "untracked_order"):
                if t is None:
                    if adopt_all:
                        self._adopt_unknown(by_comment_pos.get(m.ref), by_comment_ord.get(m.ref))
                    continue
                pos, order = by_comment_pos.get(m.ref), by_comment_ord.get(m.ref)
                if pos is not None:
                    t.state, t.position_id, t.lots = "open", pos.position_id, pos.lots
                elif order is not None:
                    t.state, t.broker_order_id = "pending", order.broker_order_id
                elif not any(d.comment == m.ref for d in deals):
                    t.state, t.note = "failed", "never reached the broker"
                elif adopt_all:
                    t.state, t.note = "closed", "owner resync"
            elif t is None:
                continue
            elif m.kind == "missing_position":
                out = sum(
                    (d.lots for d in deals if d.position_id == t.position_id and d.entry == "out"), ZERO
                )
                if out >= t.lots or adopt_all:
                    t.state, t.lots = "closed", ZERO
            elif m.kind == "missing_order":
                fill = next((d for d in deals if d.comment == t.client_order_id and d.entry == "in"), None)
                if fill is not None:
                    t.state, t.position_id, t.lots = "open", fill.position_id, fill.lots
                elif (t.expires_at is not None and t.expires_at <= om.clock.now()) or adopt_all:
                    t.state = "expired" if t.expires_at and t.expires_at <= om.clock.now() else "cancelled"
            elif m.kind == "lots":
                p = positions.get(t.position_id or "")
                if p is not None and (adopt_all or _deals_explain(deals, t, p.lots)):
                    t.lots = p.lots
            elif m.kind == "missing_stop":
                await om.confirm_stop(t)
        if adopt_all:
            om.state.expected_balance = view.balance
        om.save()

    def _adopt_unknown(self, pos: BrokerPosition | None, order: BrokerOrder | None) -> None:
        """Owner resync of a system-tagged item the journal never saw (e.g. the journal was lost)."""
        src = pos or order
        if src is None:
            return
        om = self.om
        coid = src.comment
        om.state.orders[coid] = TrackedOrder(
            client_order_id=coid,
            intent_id=None,
            account_id=om.account_id,
            strategy_id=f"adopted-magic-{src.magic}",
            strategy_version="unknown",
            symbol=src.symbol,
            side=src.side,
            order_type="market" if pos is not None else (order.order_type if order else "market"),
            lots=src.lots,
            requested_price=None,
            sl=src.sl if src.sl is not None else ZERO,
            tp=src.tp,
            magic=src.magic,
            state="open" if pos is not None else "pending",
            position_id=pos.position_id if pos else None,
            broker_order_id=order.broker_order_id if order else None,
            created_at=pos.opened_at if pos else (order.created_at if order else om.clock.now()),
            note="adopted by owner resync",
        )


def _deals_explain(deals: list[BrokerDeal], t: TrackedOrder, broker_lots: Decimal) -> bool:
    """Broker lots = entry lots - closed lots, per the broker's own deals."""
    pid = t.position_id
    lots_in = sum((d.lots for d in deals if d.position_id == pid and d.entry == "in"), ZERO)
    lots_out = sum((d.lots for d in deals if d.position_id == pid and d.entry == "out"), ZERO)
    return lots_in > 0 and lots_in - lots_out == broker_lots
