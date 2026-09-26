"""API (spec section 16): read-only status endpoints, owner-only control endpoints, and a live dashboard.

Control endpoints need `Authorization: Bearer <AT_API_OWNER_TOKEN>` (constant-time compare) and every
call is written to the audit log. There is no endpoint that changes risk limits; a FULL_HALT resume is
only forwarded, and the risk gate accepts it only with an owner-signed token.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import Awaitable, Callable
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from autotrader.core.bus import CONTROL, Bus
from autotrader.core.clock import Clock
from autotrader.core.events import ResumeRequested
from autotrader.core.indicators.candlestick import BEARISH, BULLISH, candlestick_patterns
from autotrader.core.models import Stage, Timeframe
from autotrader.lifecycle.registry import IllegalTransitionError, Registry
from autotrader.monitor.audit import AuditLog
from autotrader.monitor.candles import candle_json
from autotrader.monitor.feed import STEPS
from autotrader.monitor.journey import journey_json
from autotrader.monitor.state import MonitorService

MIN_TOKEN = 32
# reversal and continuation patterns worth marking on a chart (dojis and spinning tops are everywhere)
SHOWN_PATTERNS = (
    "bullish_engulfing",
    "bearish_engulfing",
    "hammer",
    "inverted_hammer",
    "hanging_man",
    "shooting_star",
    "bullish_harami",
    "bearish_harami",
    "piercing_line",
    "dark_cloud_cover",
    "tweezers_top",
    "tweezers_bottom",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "bullish_kicker",
    "bearish_kicker",
)


class RetireBody(BaseModel):
    strategy_id: str
    version: str
    reason: str


class PaperTradeBody(BaseModel):
    strategy_id: str
    version: str
    on: bool


class AssistantMessage(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=20_000)
    base: str | None = None  # a strategy id to modify (a new session only)


class AssistantSave(BaseModel):
    session_id: str
    responsible: bool  # the owner confirms they are responsible for what they save


class PauseBody(BaseModel):
    paused: bool


class ResumeBody(BaseModel):
    token: dict[str, str]
    signature: str


def _f(x: Decimal | None) -> float | None:
    return None if x is None else float(x)


def create_app(
    monitor: MonitorService,
    *,
    audit: AuditLog,
    bus: Bus,
    clock: Clock,
    registry: Registry | None = None,
    owner_token: str | None = None,
    learning_freeze_path: Path = Path("var/learning.freeze"),
    contract_size: dict[str, Decimal] | None = None,
    quote_ccy: dict[str, str] | None = None,
    info: dict[str, str] | None = None,
    allocation: Callable[[], dict[str, Any]] | None = None,
    risk_view: Callable[[Decimal], dict[str, Any]] | None = None,
    calendar: Callable[[], dict[str, Any]] | None = None,
    research_dir: Path | None = None,
    catalog: Callable[[], list[dict[str, Any]]] | None = None,
    paper_trade: Callable[[str, str, bool], Awaitable[str]] | None = None,
    ai: Callable[[], dict[str, Any]] | None = None,
    learning: Callable[[], dict[str, Any]] | None = None,
    assistant: Any = None,  # autotrader.ai.builder.Assistant (api may not import ai)
    known_versions: Callable[[], dict[str, list[str]]] | None = None,
    restart: Callable[[], None] | None = None,
    protect_reads: bool = False,
) -> FastAPI:
    """`protect_reads`: every /api and /research request needs the owner token (public hosting); the page
    itself stays public, it holds no data."""
    if owner_token is not None and len(owner_token) < MIN_TOKEN:
        raise ValueError(f"owner token must be at least {MIN_TOKEN} characters")
    expected = f"Bearer {owner_token}".encode() if owner_token else None
    app = FastAPI(title="autotrader", docs_url=None, redoc_url=None)
    if protect_reads:
        if expected is None:
            raise ValueError("a public hub needs an owner token")

        @app.middleware("http")
        async def require_token(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            path = request.url.path
            if path.startswith(("/api/", "/research/")):
                got = request.headers.get("authorization", "")
                if not hmac.compare_digest(got.encode(), expected):
                    return JSONResponse({"detail": "access token required"}, status_code=401)
            return await call_next(request)

    st = monitor.state
    router = monitor.router

    def owner(authorization: Annotated[str | None, Header()] = None) -> None:
        if expected is None:
            raise HTTPException(403, "control endpoints are disabled (no owner token configured)")
        if authorization is None or not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(401, "owner token required")

    owner_only = [Depends(owner)]
    _background: set[asyncio.Task[Any]] = set()  # assistant replies in flight

    # ------------------------------------------------------------ dashboard

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return files("autotrader.api").joinpath("dashboard.html").read_text()

    # ------------------------------------------------------------ read only

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        now = clock.now()
        a = st.account
        peak = st.peak_equity
        return {
            "now": now.isoformat(),
            "info": info or {},
            "halt": st.halt.value,
            "halt_reason": st.halt_reason,
            "account": None
            if a is None
            else {
                "id": a.account_id,
                "currency": a.currency,
                "balance": _f(a.balance),
                "equity": _f(a.equity),
                "free_margin": _f(a.free_margin),
                "drawdown": float((peak - a.equity) / peak) if peak else 0.0,
                "day_pnl": _f(a.equity - st.day_start[1]) if st.day_start else 0.0,
                "updated": a.at.isoformat(),
            },
            "open_risk": _f(st.open_risk(contract_size or {}, quote_ccy or {})),
            "heartbeats": {k: (now - v).total_seconds() for k, v in sorted(st.heartbeats.items())},
            "signals": dict(st.signals),
            "verdicts": dict(st.verdicts),
            "critical_open": len(router.active),
            "learning_paused": learning_freeze_path.exists(),
            "pipeline": {k: st.feed.counts.get(k, 0) for k in STEPS},
            "controls": expected is not None,
        }

    @app.get("/api/feed")
    def feed(
        after: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=500)] = 200
    ) -> dict[str, Any]:
        """Events newer than `after` (the last seq the page has), oldest first."""
        items = st.feed.after(after, limit)
        return {
            "seq": st.feed.seq,
            "items": [
                {
                    "seq": i.seq,
                    "at": i.at.isoformat(),
                    "step": i.step,
                    "tone": i.tone,
                    "text": i.text,
                    "strategy": i.strategy,
                    "symbol": i.symbol,
                    "ref": i.ref,
                }
                for i in items
            ],
        }

    @app.get("/api/candles")
    def candles(
        symbol: str,
        tf: Timeframe = Timeframe.M15,
        limit: Annotated[int, Query(ge=10, le=5000)] = 1000,
    ) -> list[dict[str, Any]]:
        """Bid candles built from the quotes the system receives, aligned like the strategies' bars."""
        return [candle_json(c) for c in st.candles.series(symbol, tf, limit)]

    @app.get("/api/patterns")
    def candle_patterns(
        symbol: str,
        tf: Timeframe = Timeframe.M15,
        limit: Annotated[int, Query(ge=10, le=5000)] = 1000,
    ) -> list[dict[str, Any]]:
        """Classic candlestick patterns on the chart's candles, each at the bar that completes it."""
        cs = st.candles.series(symbol, tf, limit)
        if not cs:
            return []
        pats = candlestick_patterns(
            [c.o for c in cs], [c.h for c in cs], [c.l for c in cs], [c.c for c in cs]
        )
        out = []
        for i, c in enumerate(cs):
            names = [n for n in SHOWN_PATTERNS if bool(pats[n][i])]
            if names:
                out.append(
                    {
                        "t": c.open_ns // 1_000_000_000,
                        "names": names,
                        "bias": "bull"
                        if names[0] in BULLISH
                        else "bear"
                        if names[0] in BEARISH
                        else "neutral",
                    }
                )
        return out

    @app.get("/api/journeys")
    def journeys(limit: Annotated[int, Query(ge=1, le=300)] = 50) -> list[dict[str, Any]]:
        return [journey_json(j, steps=False) for j in st.journeys.summaries(limit)]

    @app.get("/api/journey/{ref}")
    def journey(ref: str) -> dict[str, Any]:
        j = st.journeys.by_ref.get(ref)
        if j is None:
            raise HTTPException(404, "no such trade journey (older ones are dropped after 300)")
        return journey_json(j, steps=True)

    @app.get("/api/execution")
    def execution() -> dict[str, Any]:
        """Every fill with its slippage next to what the backtest cost model assumed, and per symbol."""
        rows = list(st.journeys.fills)
        per: dict[str, list[Any]] = {}
        for r in rows:
            per.setdefault(r.symbol, []).append(r)

        def mean(xs: list[float]) -> float | None:
            return sum(xs) / len(xs) if xs else None

        return {
            "fills": [
                {
                    "at": r.at.isoformat(),
                    "ref": r.ref,
                    "symbol": r.symbol,
                    "side": r.side,
                    "requested": r.requested,
                    "filled": r.filled,
                    "slippage": r.slippage,
                    "modelled": r.modelled,
                    "spread": r.spread,
                    "latency_ms": r.latency_ms,
                }
                for r in reversed(rows)
            ],
            "symbols": [
                {
                    "symbol": sym,
                    "fills": len(rs),
                    "avg_slippage": mean([r.slippage for r in rs if r.slippage is not None]),
                    "avg_modelled": mean([r.modelled for r in rs]),
                    "worse_than_model": sum(
                        1 for r in rs if r.slippage is not None and r.slippage > r.modelled
                    ),
                    "avg_latency_ms": mean([r.latency_ms for r in rs]),
                    "max_latency_ms": max(r.latency_ms for r in rs),
                }
                for sym, rs in sorted(per.items())
            ],
        }

    @app.get("/api/limits")
    def limits() -> dict[str, Any]:
        """The signed risk limits and how close the account is to each loss halt (the gate's formula)."""
        if risk_view is None or st.account is None:
            return {}
        return risk_view(st.account.equity)

    def _reports() -> dict[str, Path]:
        if research_dir is None or not research_dir.is_dir():
            return {}
        return {p.stem: p for p in sorted(research_dir.glob("*.json"))}

    @app.get("/api/research")
    def research() -> list[dict[str, Any]]:
        """Validation reports: what each strategy learned (walk-forward) and which gates it passed."""
        out = []
        for name, path in _reports().items():
            try:
                r = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            out.append(
                {
                    "name": name,
                    "strategy_id": r.get("strategy_id"),
                    "version": r.get("version"),
                    "family": r.get("family"),
                    "passed": r.get("passed"),
                    "synthetic": r.get("synthetic"),
                    "eligible_for_promotion": r.get("eligible_for_promotion"),
                    "research_window": r.get("research_window"),
                    "holdout_window": r.get("holdout_window"),
                    "data_versions": r.get("data_versions"),
                    "checks": r.get("checks", []),
                    "warnings": r.get("warnings", []),
                    "windows": r.get("windows", []),
                    "final_params": r.get("final_params", {}),
                    "oos_trades": len(r.get("oos_trades", [])),
                    "oos_r": [t.get("r_multiple") for t in r.get("oos_trades", [])],
                    "cross_market": r.get("cross_market", []),
                    "stability": r.get("stability", []),
                    "holdout": r.get("holdout"),
                    "html": (research_dir / f"{name}.html").exists() if research_dir else False,
                    "modified": path.stat().st_mtime,
                }
            )
        return sorted(out, key=lambda x: -float(x["modified"]))

    @app.get("/research/{name}.html")
    def research_html(name: str) -> FileResponse:
        path = _reports().get(name)
        html = path.with_suffix(".html") if path else None
        if html is None or not html.exists():
            raise HTTPException(404, "no such report")
        return FileResponse(html, media_type="text/html")

    @app.get("/api/calendar")
    def get_calendar() -> dict[str, Any]:
        """The economic calendar (ForexFactory's weekly export) and whether it feeds the news blackout."""
        return calendar() if calendar is not None else {}

    @app.get("/api/market")
    def market(points: Annotated[int, Query(ge=10, le=7500)] = 600) -> list[dict[str, Any]]:
        """Per symbol: the latest quote, mid prices (the stored week, thinned to `points`, last one kept),
        open positions and the trades inside that window, to mark on the chart."""
        a = st.account
        quotes = {q.symbol: q for q in a.quotes} if a is not None else {}
        trades = list(st.trades) + list(st.shadow_trades)
        out = []
        for sym in sorted(set(st.prices) | set(quotes)):
            full = list(st.prices.get(sym, ()))
            step = max(1, -(-len(full) // points))
            hist = full[::step]
            if full and hist[-1] is not full[-1]:
                hist.append(full[-1])
            since = hist[0][0] if hist else None
            q = quotes.get(sym)
            out.append(
                {
                    "symbol": sym,
                    "bid": _f(q.bid) if q else None,
                    "ask": _f(q.ask) if q else None,
                    "time": q.time.isoformat() if q else None,
                    "mids": [(t.isoformat(), round((b + k) / 2, 6)) for t, b, k in hist],
                    "positions": [
                        {"side": e.side, "entry": float(e.entry), "stop": _f(e.stop), "pending": e.pending}
                        for e in (a.exposures if a is not None else ())
                        if e.symbol == sym
                    ],
                    "trades": [
                        {
                            "entry_time": t.entry_time.isoformat(),
                            "exit_time": t.exit_time.isoformat(),
                            "entry": float(t.entry_price),
                            "exit": float(t.exit_price),
                            "side": t.side,
                            "r": round(t.r_multiple, 2),
                            "shadow": t.account_id == "shadow",
                        }
                        for t in trades
                        if t.symbol == sym and (since is None or t.exit_time >= since)
                    ],
                }
            )
        return out

    @app.get("/api/positions")
    def positions() -> list[dict[str, Any]]:
        a = st.account
        if a is None:
            return []
        quotes = {q.symbol: q for q in a.quotes}
        out = []
        for e in a.exposures:
            q = quotes.get(e.symbol)
            mark = None if q is None else float(q.bid if e.side == "buy" else q.ask)
            out.append(
                {
                    "symbol": e.symbol,
                    "side": e.side,
                    "lots": float(e.lots),
                    "entry": float(e.entry),
                    "stop": _f(e.stop),
                    "mark": mark,
                    "strategy": f"{e.strategy_id} {e.strategy_version}" if e.strategy_id else "external",
                    "pending": e.pending,
                }
            )
        return out

    @app.get("/api/versions")
    def versions() -> list[dict[str, Any]]:
        stages = registry.stages() if registry is not None else st.stages
        trades = list(st.trades) + list(st.shadow_trades)
        out = []
        for (sid, v), stage in sorted(stages.items()):
            mine = [t for t in trades if (t.strategy_id, t.strategy_version) == (sid, v)]
            row: dict[str, Any] = {
                "strategy_id": sid,
                "version": v,
                "stage": stage.value,
                "trades": len(mine),
                "total_r": round(sum(t.r_multiple for t in mine), 2),
            }
            if registry is not None:
                vs = registry.get(sid, v)
                row["since"] = vs.stage_since.isoformat()
                row["demo_only"] = vs.info.demo_only
            out.append(row)
        return out

    @app.get("/api/trades")
    def trades(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[dict[str, Any]]:
        rows = sorted(list(st.trades) + list(st.shadow_trades), key=lambda t: t.exit_time)[-limit:]
        return [
            {
                "id": t.trade_id,
                "strategy": f"{t.strategy_id} {t.strategy_version}",
                "shadow": t.account_id == "shadow",
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry": float(t.entry_price),
                "exit": float(t.exit_price),
                "pnl": float(t.pnl_net),
                "r": round(t.r_multiple, 2),
            }
            for t in reversed(rows)
        ]

    @app.get("/api/equity")
    def equity() -> list[tuple[str, float]]:
        return [(t.isoformat(), e) for t, e in st.equity_curve]

    @app.get("/api/decisions")
    def decisions(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[dict[str, Any]]:
        return [
            {
                "at": d.decided_at.isoformat(),
                "verdict": d.verdict,
                "lots": float(d.approved_lots),
                "reasons": list(d.reasons),
                "sequence": d.sequence,
            }
            for d in list(st.decisions)[-limit:][::-1]
        ]

    @app.get("/api/alerts")
    def alerts(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[dict[str, Any]]:
        return [
            {
                "id": aid,
                "severity": a.severity.value,
                "kind": a.kind,
                "message": a.message,
                "at": a.at.isoformat(),
                "open": aid in router.active,
            }
            for aid, a in list(router.history)[-limit:][::-1]
        ]

    @app.get("/api/audit")
    def audit_tail(
        limit: Annotated[int, Query(ge=1, le=1000)] = 100, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        return audit.tail(limit, event_type)[::-1]

    @app.get("/api/summary")
    def summary() -> dict[str, str]:
        return {"text": monitor.summary_text(clock.now())}

    @app.get("/api/catalog")
    def get_catalog() -> list[dict[str, Any]]:
        """Every strategy this process can run and whether it is trading now."""
        return catalog() if catalog is not None else []

    @app.get("/api/ai")
    def get_ai() -> dict[str, Any]:
        """The owner's Claude tracks: switched on or not, the call budget, and every recent decision."""
        return (
            ai() if ai is not None else {"enabled": False, "disabled_reason": "not running", "decisions": []}
        )

    # ------------------------------------------------------------ the strategy assistant

    def _assistant() -> Any:
        if assistant is None:
            raise HTTPException(503, "the strategy assistant is not running here")
        return assistant

    @app.get("/api/assistant")
    def assistant_status() -> dict[str, Any]:
        a = _assistant()
        return {**a.status(), "sessions": a.sessions()}

    @app.get("/api/assistant/{session_id}")
    def assistant_session(session_id: str) -> dict[str, Any]:
        a = _assistant()
        try:
            return dict(a.view(a.session(session_id)))
        except KeyError as e:
            raise HTTPException(404, "no such session") from e

    @app.post("/api/assistant/message", dependencies=owner_only)
    async def assistant_message(body: AssistantMessage) -> dict[str, Any]:
        """Owner: talk to the assistant. It answers in the background; poll the session for the reply."""
        a = _assistant()
        _audit(
            "assistant_message",
            {"session_id": body.session_id, "base": body.base, "chars": len(body.message)},
        )
        try:
            sid = a.post(body.session_id, body.message, body.base)
        except KeyError as e:
            raise HTTPException(404, f"unknown session or strategy: {e}") from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        task = asyncio.create_task(a.reply(sid))
        _background.add(task)
        task.add_done_callback(_background.discard)
        return {"session_id": sid, "thinking": True}

    @app.post("/api/assistant/save", dependencies=owner_only)
    def assistant_save(body: AssistantSave) -> dict[str, Any]:
        """Owner: save the session's checked draft as a new strategy version (loads on restart)."""
        a = _assistant()
        try:
            saved = a.save(body.session_id, body.responsible, known_versions() if known_versions else {})
        except PermissionError as e:
            raise HTTPException(400, str(e)) from e
        except (KeyError, ValueError) as e:
            raise HTTPException(409, str(e)) from e
        _audit("assistant_save", saved)
        return dict(saved)

    @app.get("/api/strategy/{strategy_id}/code")
    def strategy_code(strategy_id: str) -> dict[str, Any]:
        """A strategy's manifest and code, as the demo runs it."""
        try:
            return dict(_assistant().code(strategy_id))
        except KeyError as e:
            raise HTTPException(404, "unknown strategy") from e

    @app.post("/api/control/restart", dependencies=owner_only)
    def control_restart() -> dict[str, bool]:
        """Owner: restart the demo so saved strategies load (the simulated market starts again)."""
        if restart is None:
            raise HTTPException(503, "restart is not available here")
        _audit("restart", {})
        restart()
        return {"restarting": True}

    @app.get("/api/learning")
    def get_learning() -> dict[str, Any]:
        """The trade journal (each signal, its market snapshot and outcome) and the learning loops."""
        return (
            learning()
            if learning is not None
            else {"signals": 0, "by_strategy": [], "recent": [], "loops": []}
        )

    @app.get("/api/allocation")
    def get_allocation() -> dict[str, Any]:
        return allocation() if allocation is not None else {}

    # ------------------------------------------------------------ owner control (audited)

    def _audit(action: str, payload: dict[str, Any]) -> None:
        audit.record(
            "OwnerCommand", "owner:api", {"action": action, **payload, "at": clock.now().isoformat()}
        )

    @app.post("/api/control/retire", dependencies=owner_only)
    def retire(body: RetireBody) -> dict[str, str]:
        if registry is None:
            raise HTTPException(503, "registry not available here")
        _audit("retire", body.model_dump())
        try:
            registry.retire(body.strategy_id, body.version, f"owner: {body.reason}")
        except (IllegalTransitionError, KeyError) as e:
            raise HTTPException(409, str(e)) from e
        return {"stage": Stage.RETIRED.value}

    @app.post("/api/control/paper-trade", dependencies=owner_only)
    async def control_paper_trade(body: PaperTradeBody) -> dict[str, str]:
        """Owner: switch a strategy's paper trading on or off (demo only; the registry decides if allowed)."""
        if paper_trade is None:
            raise HTTPException(503, "paper trading control is not available here")
        _audit("paper_trade", body.model_dump())
        try:
            stage = await paper_trade(body.strategy_id, body.version, body.on)
        except (IllegalTransitionError, KeyError) as e:
            raise HTTPException(409, str(e)) from e
        return {"stage": stage}

    @app.post("/api/control/pause-learning", dependencies=owner_only)
    def pause_learning(body: PauseBody) -> dict[str, bool]:
        _audit("pause_learning", body.model_dump())
        if body.paused:
            learning_freeze_path.parent.mkdir(parents=True, exist_ok=True)
            learning_freeze_path.write_text(clock.now().isoformat())
        else:
            learning_freeze_path.unlink(missing_ok=True)
        return {"paused": learning_freeze_path.exists()}

    @app.post("/api/control/resume", dependencies=owner_only)
    async def resume(body: ResumeBody) -> dict[str, str]:
        """Forwarded to the risk gate, which verifies the owner signature, expiry and nonce itself."""
        _audit("resume_request", {"nonce": body.token.get("nonce", "")})
        await bus.publish(
            CONTROL, ResumeRequested(at=clock.now(), token=body.token, signature=body.signature)
        )
        return {"status": "forwarded to the risk gate"}

    @app.post("/api/alerts/{alert_id}/ack", dependencies=owner_only)
    def ack(alert_id: str) -> dict[str, int]:
        _audit("ack_alert", {"id": alert_id})
        return {"acknowledged": router.ack(alert_id, actor="owner:api")}

    return app
