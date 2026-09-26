"""Message bus between services (spec sections 3 and 8): Redis Streams in production, in memory in tests.

Every service reads each stream through its own consumer group, so every service sees every message,
and a message is acknowledged only after its handler finished. A service that crashes mid-message
gets it again after restart (handlers are idempotent: the order manager dedupes by client order id,
the risk gate by intent). A handler that raises is logged, alerted and acknowledged, so one poison
message cannot block a stream forever.

`pump` delivers everything available to a set of services until nothing moves: deterministic, used by
integration tests and by the in-process pipeline. `run` is the production loop with blocking reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import TypeAdapter

from autotrader.core.events import BusMessage

log = logging.getLogger(__name__)
_ADAPTER: TypeAdapter[Any] = TypeAdapter(BusMessage)

# stream names
QUOTES = "quotes"
SIGNALS = "signals"
INTENTS = "intents"
DECISIONS = "decisions"
HALTS = "halts"
ACCOUNT = "account"
TRADES = "trades"
ORDERS = "orders"  # every order placed, modified, filled, cancelled (audit)
CONFIG = "config"  # ConfigChanged: config files hashed at load (spec section 17)
STAGES = "stages"
CONTROL = "control"
REQUESTS = "strategy_requests"
HEARTBEATS = "heartbeats"
ALERTS = "alerts"
ALL_STREAMS = (
    QUOTES,
    SIGNALS,
    INTENTS,
    DECISIONS,
    HALTS,
    ACCOUNT,
    TRADES,
    ORDERS,
    CONFIG,
    STAGES,
    CONTROL,
    REQUESTS,
    HEARTBEATS,
    ALERTS,
)

Handler = Callable[[Any], Awaitable[None]]


def encode(msg: Any) -> str:
    return str(_ADAPTER.dump_json(msg).decode())


def decode(raw: str | bytes) -> Any:
    return _ADAPTER.validate_json(raw)


class Bus(Protocol):
    async def publish(self, stream: str, msg: Any) -> str: ...

    async def read(
        self, stream: str, group: str, consumer: str, *, count: int = 100, block_ms: int | None = None
    ) -> list[tuple[str, Any]]: ...

    async def ack(self, stream: str, group: str, ids: Sequence[str]) -> None: ...


@dataclass
class InMemoryBus:
    """Same semantics as Redis Streams consumer groups (one cursor per group, pending until acked).

    Messages are stored serialized, like Redis, and decoded once: every group reading a message gets the same
    frozen object (one decode per message instead of one per reader). Handlers never mutate a message."""

    streams: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    cursors: dict[tuple[str, str], int] = field(default_factory=dict)
    pending: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
    _seq: int = 0
    _decoded: dict[str, Any] = field(default_factory=dict)

    async def publish(self, stream: str, msg: Any) -> str:
        self._seq += 1
        mid = f"{self._seq}-0"
        self.streams.setdefault(stream, []).append((mid, encode(msg)))  # serialized like Redis
        return mid

    async def read(
        self, stream: str, group: str, consumer: str, *, count: int = 100, block_ms: int | None = None
    ) -> list[tuple[str, Any]]:
        k = (stream, group)
        rows = self.streams.get(stream)
        start = self.cursors.get(k, 0)
        if rows is None or start >= len(rows):  # fast path: nothing new
            if block_ms:
                await asyncio.sleep(block_ms / 1000)
            return []
        pend = self.pending.setdefault(k, {})
        batch = rows[start : start + count]
        self.cursors[k] = start + len(batch)
        for mid, raw in batch:
            pend[mid] = raw
        out = []
        for mid, raw in batch:
            msg = self._decoded.get(mid)
            if msg is None:
                msg = self._decoded[mid] = decode(raw)
            out.append((mid, msg))
        return out

    async def ack(self, stream: str, group: str, ids: Sequence[str]) -> None:
        pend = self.pending.setdefault((stream, group), {})
        for i in ids:
            pend.pop(i, None)


class RedisStreamsBus:
    """Redis Streams with consumer groups (`redis.asyncio` client, or fakeredis in tests)."""

    def __init__(self, client: Any, maxlen: int = 100_000) -> None:
        self.r = client
        self.maxlen = maxlen
        self._ready: set[tuple[str, str]] = set()
        self._recovered: set[tuple[str, str, str]] = set()

    async def publish(self, stream: str, msg: Any) -> str:
        mid = await self.r.xadd(stream, {"m": encode(msg)}, maxlen=self.maxlen, approximate=True)
        return mid.decode() if isinstance(mid, bytes) else str(mid)

    async def _group(self, stream: str, group: str) -> None:
        if (stream, group) in self._ready:
            return
        try:
            await self.r.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as e:  # BUSYGROUP: exists already
            if "BUSYGROUP" not in str(e):
                raise
        self._ready.add((stream, group))

    async def read(
        self, stream: str, group: str, consumer: str, *, count: int = 100, block_ms: int | None = None
    ) -> list[tuple[str, Any]]:
        await self._group(stream, group)
        k = (stream, group, consumer)
        start = ">" if k in self._recovered else "0"  # after a restart: our unacked messages first
        res = await self.r.xreadgroup(group, consumer, {stream: start}, count=count, block=block_ms)
        if start == "0":
            self._recovered.add(k)
        out = []
        for _name, entries in res or []:
            for mid, fields in entries:
                raw = fields.get(b"m") or fields.get("m")
                out.append((mid.decode() if isinstance(mid, bytes) else str(mid), decode(raw)))
        if start == "0" and not out:
            return await self.read(stream, group, consumer, count=count, block_ms=block_ms)
        return out

    async def ack(self, stream: str, group: str, ids: Sequence[str]) -> None:
        if ids:
            await self.r.xack(stream, group, *ids)


class Service(Protocol):
    name: str

    def handlers(self) -> Mapping[str, Handler]: ...


async def _deliver(bus: Bus, svc: Service, stream: str, handler: Handler, block_ms: int | None) -> int:
    msgs = await bus.read(stream, svc.name, f"{svc.name}-1", block_ms=block_ms)
    for mid, msg in msgs:
        try:
            await handler(msg)
        except Exception:  # logged and acked: one bad message must not block the stream
            log.exception("%s failed on %s %s", svc.name, stream, mid)
            on_error = getattr(svc, "on_handler_error", None)
            if on_error is not None:
                on_error(stream, mid)
        await bus.ack(stream, svc.name, [mid])
    return len(msgs)


async def pump(bus: Bus, services: Sequence[Service], max_rounds: int = 10_000) -> int:
    """Deliver until no service has anything left to read. Returns the number of deliveries."""
    total = 0
    for _ in range(max_rounds):
        moved = 0
        for svc in services:
            for stream, handler in svc.handlers().items():
                moved += await _deliver(bus, svc, stream, handler, None)
        total += moved
        if moved == 0:
            return total
    raise RuntimeError("bus did not settle (a message loop between services?)")


async def run(bus: Bus, svc: Service, stop: asyncio.Event, block_ms: int = 1000) -> None:
    """Production loop: one reader task per subscribed stream until `stop` is set."""

    async def reader(stream: str, handler: Handler) -> None:
        while not stop.is_set():
            await _deliver(bus, svc, stream, handler, block_ms)

    tasks = [asyncio.create_task(reader(s, h)) for s, h in svc.handlers().items()]
    try:
        await stop.wait()
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
