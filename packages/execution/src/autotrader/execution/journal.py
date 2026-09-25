"""Execution's own record of what it sent (write-ahead) and what the broker confirmed.

Persisted atomically with a content hash after every change, so a restart knows which orders may be
in flight. A corrupt journal raises JournalCorruptError: the caller must fail closed (RECON_HALT and
an owner resync), never start from an empty journal as if nothing were open.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from autotrader.core.broker import OrderType
from autotrader.core.fileio import atomic_write_text
from autotrader.core.hashing import canonical_json, sha256_hex
from autotrader.core.models import DEFAULT_TENANT, Side, UtcDatetime

OrderState = Literal["sending", "pending", "open", "closed", "cancelled", "expired", "rejected", "failed"]
ReconEpisode = Literal["clean", "unreachable", "resyncing", "stuck"]
ACTIVE: frozenset[OrderState] = frozenset({"sending", "pending", "open"})
SEEN_DEAL_RETENTION = timedelta(days=7)


class JournalCorruptError(Exception):
    pass


class TrackedOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    client_order_id: str
    intent_id: UUID | None  # None only for positions adopted by an owner resync
    account_id: str
    tenant_id: str = DEFAULT_TENANT
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Side
    order_type: OrderType
    lots: Decimal  # open lots once filled (reduced by partial closes)
    requested_price: Decimal | None
    sl: Decimal
    tp: Decimal | None
    magic: int
    state: OrderState
    broker_order_id: str | None = None
    position_id: str | None = None
    created_at: UtcDatetime
    expires_at: UtcDatetime | None = None
    stop_confirmed: bool = False
    note: str = ""
    # filled position, for the closed-trade record (R needs the entry, initial stop and opened lots)
    entry_price: Decimal | None = None
    initial_sl: Decimal | None = None
    opened_lots: Decimal | None = None
    filled_at: UtcDatetime | None = None
    realized: Decimal = Decimal(0)  # profit of closing deals, account currency
    costs: Decimal = Decimal(0)  # commissions paid minus swaps received
    out_value: Decimal = Decimal(0)  # sum of exit price x lots
    out_lots: Decimal = Decimal(0)


class ExecState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orders: dict[str, TrackedOrder] = Field(default_factory=dict)
    seen_deals: dict[str, UtcDatetime] = Field(default_factory=dict)
    deals_cursor: UtcDatetime | None = None
    cash_since: UtcDatetime | None = None  # deals at or after this change expected_balance
    expected_balance: Decimal | None = None
    last_decision_sequence: int = -1  # survives restarts so an old decision cannot be replayed
    alerted_external: list[str] = Field(default_factory=list)
    # why reconciliation last halted; persisted so a restart neither forgets to clear nor clears wrongly
    recon_episode: ReconEpisode = "clean"

    def active(self) -> list[TrackedOrder]:
        return [o for o in self.orders.values() if o.state in ACTIVE]

    def by_position(self, position_id: str) -> TrackedOrder | None:
        return next((o for o in self.orders.values() if o.position_id == position_id), None)

    def by_broker_order(self, broker_order_id: str) -> TrackedOrder | None:
        return next((o for o in self.orders.values() if o.broker_order_id == broker_order_id), None)


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> ExecState:
        if not self.path.exists():
            return ExecState()
        try:
            raw = json.loads(self.path.read_text())
            body, digest = raw["state"], raw["sha256"]
            if sha256_hex(canonical_json(body)) != digest:
                raise ValueError("journal hash mismatch")
            return ExecState.model_validate(body)
        except (ValueError, KeyError, TypeError) as e:
            raise JournalCorruptError(str(e)) from e

    def save(self, state: ExecState, now: datetime | None = None) -> None:
        if now is not None:
            cutoff = now - SEEN_DEAL_RETENTION
            state.seen_deals = {k: t for k, t in state.seen_deals.items() if t >= cutoff}
        body = state.model_dump(mode="json")
        atomic_write_text(self.path, json.dumps({"state": body, "sha256": sha256_hex(canonical_json(body))}))
