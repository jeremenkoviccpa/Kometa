"""`at demo run`: the whole system in one process, on one bus, with the dashboard.

Every service is the real one: engine-live, allocator, risk gate (signed config, signed decisions),
execution (order manager, reconciliation, watchdog), lifecycle, monitor, audit and API. Only the
market differs:

- `--broker sim` (default): a synthetic market for `--symbol` (XAUUSD by default, clearly labelled as
  simulated prices) played on a simulated clock, first a catch-up period as fast as possible, then at
  `--speed` simulated seconds per real second. The instrument is the real spec from
  config/instruments.yaml (contract size, lot steps, commission) and the simulated broker fills with
  seeded slippage and latency, so sizes, costs, P&L and R behave like the real thing. Only the prices
  are made up. A throwaway owner key signs a copy of config/risk.paper.yaml; nothing leaves the process.
- `--broker mt5`: a broker DEMO account through the MT5 bridge, real time. Needs AT_ENV=paper, the
  owner-signed config/risk.paper.yaml(.sig) and the bridge settings; execution refuses real accounts.

The only strategy is demo_ma_cross (plumbing test, NOT a trading method). It runs in the owner-granted
demo_only stage: paper trading at the instrument's minimum size. Its manifest lists SYNTH (the golden
backtest pins it); the demo runs the same code and version on `--symbol` instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from autotrader.ai.claude import ClaudeDecider
from autotrader.ai.trader import AiConfig, AiTraderService
from autotrader.allocator.allocator import Allocator
from autotrader.allocator.config import load_allocator_config
from autotrader.allocator.service import AllocatorService
from autotrader.api.app import create_app
from autotrader.core.alerts import Alert, Severity
from autotrader.core.broker import Quote, SymbolInfo
from autotrader.core.bus import CONFIG, CONTROL, HEARTBEATS, InMemoryBus, Service, pump
from autotrader.core.clock import Clock, LiveClock, SimClock
from autotrader.core.configs import config_events
from autotrader.core.events import DemotionOrder, Heartbeat
from autotrader.core.indicators.sessions import EventIndex
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import Instrument, Stage, Timeframe
from autotrader.core.series import BarsArray, from_ns, to_ns
from autotrader.core.settings import Settings
from autotrader.core.signing import DecisionVerifier, generate_keypair, load_private, load_public, sign_bytes
from autotrader.core.timeframes import bucket_open_ns
from autotrader.data.calendar import ForexFactoryCalendar
from autotrader.data.instruments import load_instruments
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.engine.costs import DEFAULT_SLIPPAGE_MULT
from autotrader.engine.market import news_from_index
from autotrader.engine.service import EngineLiveService
from autotrader.execution.adapter import BrokerAdapter
from autotrader.execution.bus_service import ExecutionBusService
from autotrader.execution.config import load_execution_config
from autotrader.execution.ctrader import CTraderAdapter
from autotrader.execution.fake import FakeBroker
from autotrader.execution.journal import Journal
from autotrader.execution.mt5 import MT5Adapter
from autotrader.execution.oanda import OandaAdapter
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.quality import JsonlQualityLog
from autotrader.execution.quotes import QuoteBook
from autotrader.execution.reconcile import Reconciler
from autotrader.execution.service import ExecutionService, startup_checks
from autotrader.execution.watchdog import Watchdog
from autotrader.learning.journal import JournalService
from autotrader.learning.lessons import LessonBook, LessonService, diagnose
from autotrader.lifecycle.config import load_promotion_config
from autotrader.lifecycle.evaluator import Evaluator, MemoryStageData
from autotrader.lifecycle.registry import Registry, VersionInfo
from autotrader.lifecycle.service import LifecycleService
from autotrader.monitor.alerts import AlertRouter, TelegramNotifier
from autotrader.monitor.audit import AuditLog, AuditService, check_chain
from autotrader.monitor.state import MonitorService, SlippageWatch
from autotrader.risk.gate import RiskGate
from autotrader.risk.service import RiskGateService, load_limits_or_alert
from autotrader.risk.state import StateStore
from autotrader.strategies_api.loader import LoadedStrategy, load_strategy
from autotrader.validation.inputs import prepare

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
SERVICES_WITH_HEARTBEAT = ("engine", "risk-gate", "allocator", "lifecycle", "execution")


@dataclass
class DemoConfig:
    root: Path
    var: Path
    broker: str = "sim"
    catchup_days: int = 15
    run_days: float = 60.0
    speed: float = 3600.0  # simulated seconds per real second (sim only): one market hour per second
    seed: int = 7
    equity: Decimal = Decimal(50_000)  # 0.1% (the paper stage) must cover a minimum-lot gold stop
    port: int = 8000
    serve: bool = True
    symbol: str = "XAUUSD"
    start_price: float | None = None  # sim: first price; default from MARKETS
    leverage: int = 100  # sim broker margin: notional / leverage per lot
    calendar: bool = True  # fetch ForexFactory's weekly calendar (run_demo only; build never fetches)
    host: str = "127.0.0.1"  # anything else is public hosting: the whole API then needs the owner token
    history_days: int = 260  # sim: bars played before the start only to warm strategies up (D1 needs ~160)
    trade: tuple[str, ...] | None = None  # strategies paper trading at start; None = the whole library


# spec section 14: what exists of each learning loop in this build (shown in the hub, updated per slice)
LOOPS = [
    {
        "loop": "Journal + features",
        "status": "running",
        "what": "every signal, its market snapshot and outcome",
    },
    {"loop": "L1 Discovery", "status": "not built", "what": "Claude research agents propose new strategies"},
    {
        "loop": "L2 Re-optimization",
        "status": "running",
        "what": "at learn reopt makes challengers; the lifecycle swaps them",
    },
    {
        "loop": "L3 Meta-labeling",
        "status": "not built",
        "what": "learn which signals to skip (needs 300 signals)",
    },
    {"loop": "L4 Regime", "status": "not built", "what": "trending/ranging/volatile days per strategy"},
    {"loop": "L5 Allocation", "status": "running", "what": "risk budget by live results (live stages only)"},
    {
        "loop": "L6 Failure lessons",
        "status": "running",
        "what": "a lesson on every demotion or failed validation",
    },
    {"loop": "L7 Meta-learning", "status": "not built", "what": "which research sources produce survivors"},
    {"loop": "L8 Cost calibration", "status": "not built", "what": "real spreads and slippage from fills"},
]

STYLE = {
    "swing_trend_pullback": "Swing",
    "candle_price_action": "Price action",
    "session_breakout": "Scalping",
    "smc_top_down": "SMC / ICT sniper",
    "demo_ma_cross": "Plumbing test",
    "claude_smc": "Claude AI",
}


@dataclass(frozen=True)
class SimMarket:
    """How a simulated market moves: yearly volatility, typical spread in pips, a plausible start."""

    annual_vol: float
    spread_pips: float
    start_price: float


MARKETS = {
    "XAUUSD": SimMarket(annual_vol=0.16, spread_pips=25.0, start_price=4300.0),  # ~$0.25 spread
    "EURUSD": SimMarket(annual_vol=0.07, spread_pips=0.9, start_price=1.10),
    "SYNTH": SimMarket(annual_vol=0.10, spread_pips=1.0, start_price=100.0),
}


@dataclass
class DemoStack:
    cfg: DemoConfig
    clock: Clock
    bus: InMemoryBus
    router: AlertRouter
    audit: AuditLog
    monitor: MonitorService
    registry: Registry
    execution: ExecutionBusService
    exec_service: ExecutionService
    engine: EngineLiveService
    risk: RiskGateService
    lifecycle: LifecycleService
    allocator: AllocatorService
    adapter: BrokerAdapter
    services: list[Service] = field(default_factory=list)
    frame: pl.DataFrame | None = None  # the synthetic market (sim only)
    info: dict[str, str] = field(default_factory=dict)
    contract: dict[str, Decimal] = field(default_factory=dict)
    quote_ccy: dict[str, str] = field(default_factory=dict)
    calendar: ForexFactoryCalendar | None = None
    news_applied: bool = False  # the calendar feeds the risk gate's blackout (paper/live only)
    strategies: list[LoadedStrategy] = field(default_factory=list)
    ai: AiTraderService | None = None  # the owner's Claude tracks
    journal: JournalService | None = None  # every signal, its market snapshot and its outcome
    lessons: LessonBook | None = None  # L6: a lesson on every demotion, retirement, failed validation

    def catalog(self) -> list[dict[str, Any]]:
        """Every strategy the demo can run, with its stage in the demo's registry."""
        stages = self.registry.stages()
        out = []
        for ls in self.strategies:
            m = ls.manifest
            stage = stages.get((m.id, m.version))
            out.append(
                {
                    "strategy_id": m.id,
                    "version": m.version,
                    "family": m.family,
                    "style": STYLE.get(m.family, m.family),
                    "description": m.description,
                    "symbols": list(m.symbols),
                    "timeframes": [t.value for t in m.timeframes],
                    "params": {k: v.value for k, v in m.params.items()},
                    "expected": m.expected.model_dump(),
                    "stage": stage.value if stage else None,
                    "trading": stage == Stage.DEMO_ONLY,
                    "plumbing": m.demo_only,
                    "activity": self._activity(m.id, m.version),
                }
            )
        return out

    def _activity(self, strategy_id: str, version: str) -> dict[str, Any]:
        """What this version did in the demo, paper and shadow alike, from the journal."""
        es = sorted(
            (
                e
                for e in (self.journal.entries.values() if self.journal is not None else [])
                if e.strategy_id == strategy_id and e.strategy_version == version
            ),
            key=lambda e: e.created_at,
        )
        done = [e.outcome for e in es if e.outcome is not None]
        last = es[-1] if es else None
        return {
            "signals": len(es),
            "paper": sum(not e.shadow for e in es),
            "shadow": sum(e.shadow for e in es),
            "resolved": len(done),
            "target_first": sum(o.label == 1 for o in done),
            "stop_first": sum(o.label == -1 for o in done),
            "avg_r": sum(o.r for o in done) / len(done) if done else None,
            "last": None
            if last is None
            else {
                "at": last.created_at.isoformat(),
                "side": last.side,
                "shadow": last.shadow,
                "status": last.status,
                "outcome": None
                if last.outcome is None
                else {"label": last.outcome.label, "r": last.outcome.r},
            },
        }

    async def paper_trade(self, strategy_id: str, version: str, on: bool) -> str:
        """Owner: start or stop paper trading a strategy. Stopping cancels its entries and closes its
        positions (like leaving a money stage). The stage change reaches every service first."""
        self.registry.paper_trade(strategy_id, version, on)
        save_choice(choices_path(self.cfg), strategy_id, on)
        await self.lifecycle.flush()
        if not on:
            order = DemotionOrder(
                at=self.clock.now(),
                strategy_id=strategy_id,
                strategy_version=version,
                close_positions=True,
                reason="owner: paper trading stopped",
            )
            await self.bus.publish(CONTROL, order)
        await pump(self.bus, self.services)
        return self.registry.get(strategy_id, version).stage.value

    def calendar_view(self) -> dict[str, Any]:
        cal, now = self.calendar, datetime.now(UTC)
        if cal is None:
            return {}
        ccys = {c for s in self.contract for c in (s[:3], s[3:])}
        return {
            "source": "ForexFactory weekly export",
            "fetched_at": cal.fetched_at.isoformat() if cal.fetched_at else None,
            "fresh": cal.fresh(now),
            "applied_to_trading": self.news_applied,
            "relevant_currencies": sorted(ccys),
            "events": cal.rows,
        }

    def app(self, owner_token: str | None, *, protect_reads: bool = False) -> Any:
        return create_app(
            self.monitor,
            audit=self.audit,
            bus=self.bus,
            clock=self.clock,
            registry=self.registry,
            owner_token=owner_token,
            learning_freeze_path=self.cfg.var / "learning.freeze",
            contract_size=self.contract,
            quote_ccy=self.quote_ccy,
            info=self.info,
            allocation=self._allocation,
            risk_view=self.risk.gate.usage,
            calendar=self.calendar_view,
            research_dir=Settings().reports_dir,
            catalog=self.catalog,
            paper_trade=self.paper_trade,
            ai=self.ai.view if self.ai is not None else None,
            learning=self.learning_view,
            protect_reads=protect_reads,
        )

    def learning_view(self) -> dict[str, Any]:
        """The journal plus where each learning loop of spec section 14 stands in this build."""
        j = (
            self.journal.view()
            if self.journal is not None
            else {"signals": 0, "by_strategy": [], "recent": []}
        )
        entries = list(self.journal.entries.values()) if self.journal is not None else []
        by: dict[str, list[Any]] = {}
        for e in entries:
            by.setdefault(f"{e.strategy_id} {e.strategy_version}", []).append(e)
        lessons = self.lessons.top(k=20) if self.lessons is not None else []
        return {
            **j,
            "diagnosis": {k: diagnose(v).model_dump(mode="json") for k, v in sorted(by.items())},
            "lessons": [x.model_dump(mode="json") for x in lessons],
            "challengers": [
                {
                    "strategy_id": v.info.strategy_id,
                    "version": v.info.version,
                    "champion": v.info.parent_version,
                    "stage": v.stage.value,
                    "since": v.stage_since.isoformat(),
                    "params": v.info.params,
                }
                for v in self.registry.versions()
                if v.info.origin == "learning_reopt"
            ],
            "swap_tests": list(reversed(self.registry.swap_tests[-20:])),
            "loops": LOOPS,
        }

    def _allocation(self) -> dict[str, Any]:
        cur = self.allocator.allocator.current
        return {} if cur is None else cur.model_dump(mode="json")


async def broker_history(
    adapter: BrokerAdapter,
    clock: Clock,
    strategies: list[LoadedStrategy],
    instruments: dict[str, Instrument],
    cfg: DemoConfig,
) -> dict[tuple[str, Timeframe], BarsArray]:
    """Warm-up from the broker's own M1 history, up to the start of the current trading day (17:00 New
    York): later bars come live, so no timeframe gets a partial bar where history meets live data."""
    now = clock.now()
    day_start = from_ns(bucket_open_ns(to_ns(now) - to_ns(now) % 60_000_000_000, Timeframe.D1))
    syms = sorted({s for ls in strategies for s in ls.manifest.symbols})
    frames = []
    for sym in syms:
        say(f"warming up {sym}: {cfg.history_days} days of M1 history from the broker...")
        bars = await adapter.history_bars(
            sym, Timeframe.M1, day_start - timedelta(days=cfg.history_days), day_start
        )
        if bars:
            frames.append(
                pl.DataFrame([{"open_time": b.open_time, **b.model_dump(include=BAR_COLUMNS)} for b in bars])
            )
    if not frames:
        return {}
    return warmup_history(frames[0] if len(frames) == 1 else pl.concat(frames), strategies, instruments)


BAR_COLUMNS = {"bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "volume"}


def trading_day_start(frame: pl.DataFrame, after: datetime) -> datetime:
    """The first bar at or after `after` that opens a trading day (17:00 New York): every bar of every
    timeframe that ends before it is complete, so history and live bars meet without a partial bar."""
    for t in frame.filter(pl.col("open_time") >= after)["open_time"]:
        if bucket_open_ns(to_ns(t), Timeframe.D1) == to_ns(t):
            return t  # type: ignore[no-any-return]
    raise SystemExit("not enough synthetic data after the warm-up period")


def choices_path(cfg: DemoConfig) -> Path:
    """The owner's on/off switches live next to the state directory, which a fresh start wipes."""
    return cfg.var.parent / "owner_choices.json"


def load_choices(path: Path) -> dict[str, bool]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): bool(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def save_choice(path: Path, strategy_id: str, on: bool) -> None:
    choices = {**load_choices(path), strategy_id: on}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(choices, sort_keys=True, indent=1))
    tmp.replace(path)


def warmup_history(
    frame: pl.DataFrame, strategies: list[LoadedStrategy], instruments: dict[str, Instrument]
) -> dict[tuple[str, Timeframe], BarsArray]:
    """Closed bars of every (symbol, timeframe) any strategy subscribes to, from the warm-up period."""
    out: dict[tuple[str, Timeframe], BarsArray] = {}
    if frame.is_empty():
        return out
    for ls in strategies:
        syms = ls.manifest.symbols
        inp = prepare({s: frame for s in syms}, ls.manifest, instruments, synthetic=True)
        out.update(inp.series)
    return out


def on_market(demo: LoadedStrategy, symbol: str) -> LoadedStrategy:
    """The same strategy code and version, run on `symbol` instead of its manifest's SYNTH."""
    if demo.manifest.symbols == (symbol,):
        return demo
    manifest = demo.manifest.model_copy(update={"symbols": (symbol,)})
    cls = type(demo.cls.__name__, (demo.cls,), {"manifest": manifest})
    return LoadedStrategy(cls, manifest, demo.code_hash, demo.path)


def realistic_slippage(rng: random.Random, tick: Decimal) -> Callable[[str, str, Decimal], Decimal]:
    """Adverse slippage of market executions: usually none or a tick or two, sometimes a fraction of the
    spread, rarely more (fast markets). Mean about 0.15 spreads, a little under the backtest's 0.2."""

    def slip(_symbol: str, _side: str, spread: Decimal) -> Decimal:
        u = rng.random()
        if u < 0.55:
            return Decimal(0)
        size = (
            Decimal(str(rng.expovariate(1 / 0.3))) * spread
            if u < 0.97
            else spread * Decimal(str(rng.uniform(1, 3)))
        )
        return (size / tick).quantize(Decimal(1)) * tick

    return slip


def symbol_info(inst: Instrument, margin_per_lot: Decimal) -> SymbolInfo:
    exponent = inst.tick_size.normalize().as_tuple().exponent
    digits = max(0, -exponent) if isinstance(exponent, int) else 5
    return SymbolInfo(
        symbol=inst.symbol,
        digits=digits,
        point=inst.tick_size,
        contract_size=inst.contract_size,
        min_lot=inst.min_lot,
        lot_step=inst.lot_step,
        max_lot=inst.max_lot,
        margin_per_lot=margin_per_lot,
    )


def _signed_paper_limits(
    cfg: DemoConfig, settings: Settings, router: AlertRouter, clock: Clock
) -> tuple[Any, str, Any]:
    """sim: a throwaway owner key signs a private copy; mt5 paper: the owner's own signature is required."""
    src = cfg.root / "config" / "risk.paper.yaml"
    if cfg.broker == "sim":
        priv, pub = generate_keypair()
        copy = cfg.var / "risk.paper.yaml"
        copy.write_text(src.read_text())
        (cfg.var / "risk.paper.yaml.sig").write_text(sign_bytes(load_private(priv), copy.read_bytes()))
        owner = load_public(pub)
        limits, h = load_limits_or_alert(copy, cfg.var / "risk.paper.yaml.sig", owner, router, clock.now())
        return limits, h, owner
    owner = load_public(settings.owner_public_key_path)
    limits, h = load_limits_or_alert(src, src.with_name("risk.paper.yaml.sig"), owner, router, clock.now())
    return limits, h, owner


async def build(cfg: DemoConfig, settings: Settings | None = None) -> DemoStack:
    settings = settings or Settings()
    cfg.var.mkdir(parents=True, exist_ok=True)
    root = cfg.root
    demo = on_market(load_strategy(root / "strategies" / "examples" / "demo_ma_cross"), cfg.symbol)
    library = [
        on_market(load_strategy(d), cfg.symbol)
        for d in sorted((root / "strategies" / "library").iterdir())
        if (d / "strategy.yaml").exists()
    ]
    # the owner's Claude tracks: registered like strategies (a stage and a switch), off until switched on
    ai_tracks = [
        on_market(load_strategy(d), cfg.symbol)
        for d in sorted((root / "strategies" / "ai").iterdir())
        if (d / "strategy.yaml").exists()
    ]
    strategies = [demo, *library, *ai_tracks]
    instruments, _ = load_instruments(root / "config" / "instruments.yaml")
    instruments = {**instruments, "SYNTH": synthetic_instrument()}
    if cfg.symbol not in instruments:
        raise SystemExit(f"unknown symbol {cfg.symbol!r}: add it to config/instruments.yaml")
    symbols = list(demo.manifest.symbols)
    inst = instruments[cfg.symbol]

    clock: Clock
    adapter: BrokerAdapter
    if cfg.broker == "sim":
        mk = MARKETS.get(cfg.symbol, MARKETS["SYNTH"])
        start = cfg.start_price or mk.start_price
        frame = generate(
            SyntheticSpec(
                symbol=cfg.symbol,
                days=cfg.history_days + cfg.catchup_days + int(cfg.run_days) + 3,
                seed=cfg.seed,
                start_price=start,
                pip_size=float(inst.pip_size),
                annual_vol=mk.annual_vol,
                base_spread_pips=mk.spread_pips,
            )
        )
        play_start = trading_day_start(frame, frame["open_time"][0] + timedelta(days=cfg.history_days))
        history = warmup_history(frame.filter(pl.col("open_time") < play_start), strategies, {**instruments})
        clock = SimClock(play_start)
        rng = random.Random(cfg.seed)  # noqa: S311 - simulation noise, not security
        margin = inst.contract_size * Decimal(str(start)) / cfg.leverage
        adapter = FakeBroker(
            symbols=[symbol_info(inst, margin)],
            clock=clock,
            balance=cfg.equity,
            account_id="sim-demo",
            commission_per_lot_side=inst.commission_per_lot / 2,
            slippage=realistic_slippage(rng, inst.tick_size),
            latency_ms=lambda: rng.uniform(35.0, 180.0),
        )
    elif cfg.broker == "mt5":
        if settings.env != "paper":
            raise SystemExit("the MT5 demo needs AT_ENV=paper (and a broker DEMO account)")
        if not settings.bridge_url or settings.bridge_token is None:
            raise SystemExit("AT_BRIDGE_URL and AT_BRIDGE_TOKEN must be set")
        clock = LiveClock()
        adapter = MT5Adapter(settings.bridge_url, settings.bridge_token.get_secret_value())
        frame = None
        history = await broker_history(adapter, clock, strategies, instruments, cfg)
    elif cfg.broker == "ctrader":
        if settings.env != "paper" or settings.ctrader_environment != "demo":
            raise SystemExit("the cTrader demo needs AT_ENV=paper and AT_CTRADER_ENVIRONMENT=demo")
        c = settings
        if not (
            c.ctrader_client_id
            and c.ctrader_client_secret
            and c.ctrader_access_token
            and c.ctrader_account_id
        ):
            raise SystemExit("put the AT_CTRADER_* values in .env (see config/settings.example.env)")
        clock = LiveClock()
        adapter = CTraderAdapter(
            c.ctrader_client_id,
            c.ctrader_client_secret.get_secret_value(),
            c.ctrader_access_token.get_secret_value(),
            c.ctrader_account_id,
            symbols,
            environment="demo",
        )
        await adapter.connect()
        frame = None
        history = await broker_history(adapter, clock, strategies, instruments, cfg)
    elif cfg.broker == "oanda":
        if settings.env != "paper" or settings.oanda_environment != "practice":
            raise SystemExit(
                "the OANDA demo needs AT_ENV=paper and a PRACTICE account (AT_OANDA_ENVIRONMENT)"
            )
        if not settings.oanda_account or settings.oanda_token is None:
            raise SystemExit(
                "put AT_OANDA_TOKEN and AT_OANDA_ACCOUNT in .env (see config/settings.example.env)"
            )
        clock = LiveClock()
        adapter = OandaAdapter(
            settings.oanda_account,
            settings.oanda_token.get_secret_value(),
            {s: instruments[s].contract_size for s in symbols},
            environment="practice",
        )
        frame = None
        history = await broker_history(adapter, clock, strategies, instruments, cfg)
    else:
        raise SystemExit(f"unknown broker {cfg.broker!r}")

    bus = InMemoryBus()
    audit = AuditLog(JsonlLedger(cfg.var / "audit.jsonl"))
    primary = None
    if settings.telegram_bot_token is not None and settings.telegram_chat_id:
        primary = TelegramNotifier(settings.telegram_bot_token.get_secret_value(), settings.telegram_chat_id)
    router = AlertRouter(clock, primary=primary, audit=audit)
    # the backtest's slippage assumption, so the warning compares like with like
    monitor = MonitorService(router, clock, slippage=SlippageWatch(mult=DEFAULT_SLIPPAGE_MULT))

    # risk gate: signed limits; the decision key lives only in this process
    limits, limits_hash, owner = _signed_paper_limits(cfg, settings, router, clock)
    gate_priv, gate_pub = generate_keypair()
    gate = RiskGate(
        limits, limits_hash, load_private(gate_priv), owner, StateStore(cfg.var / "risk_state.json")
    )
    # economic calendar: shown in the hub always; feeds the news blackout only on a real clock (paper/live)
    calendar = ForexFactoryCalendar(cfg.var / "ff_calendar.json") if cfg.calendar else None
    news = None
    if calendar is not None and cfg.broker != "sim":
        cal = calendar

        def news_index() -> EventIndex | None:
            now = clock.now()
            fresh = cal.fresh(now)
            return EventIndex(cal.events(now - timedelta(days=1), now + timedelta(days=8))) if fresh else None

        news = news_index
    risk = RiskGateService(gate, bus, clock, instruments, news=news)

    # execution
    ecfg = load_execution_config(root / "config" / "execution.yaml")
    if cfg.broker != "sim":
        acct = await startup_checks(
            adapter,
            clock,
            env=settings.env,
            expected_account_id=None,
            max_skew_seconds=settings.max_clock_skew_seconds,
        )
        infos = {s.symbol: s for s in await adapter.symbols()}
    else:
        acct = await adapter.account()
        infos = {s.symbol: s for s in await adapter.symbols()}
    om = OrderManager(
        adapter=adapter,
        verifier=DecisionVerifier(load_public(gate_pub)),
        journal=Journal(cfg.var / "execution_journal.json"),
        alerts=router,
        clock=clock,
        config=ecfg,
        quotes=QuoteBook(),
        quality=JsonlQualityLog(cfg.var / "execution_quality.jsonl"),
        account_id=acct.account_id,
        symbols={s: infos[s] for s in symbols},
    )
    watchdog = Watchdog(om, clock.now())
    exec_service = ExecutionService(om, Reconciler(om, gate), watchdog)
    execution = ExecutionBusService(om, watchdog, bus)

    # lifecycle: every strategy is registered in the DEMO's own registry as a paper trial (demo_only: it can
    # never be promoted from here); the owner switches each between shadow (off) and paper trading (on)
    registry = Registry(JsonlLedger(cfg.var / "registry.jsonl"), clock, router)
    known = {v.key for v in registry.versions()}
    trade = set(cfg.trade) if cfg.trade is not None else {ls.manifest.id for ls in library}
    # the owner's own switches win over the start-up default, and survive restarts and deploys
    for sid, on in load_choices(choices_path(cfg)).items():
        (trade.add if on else trade.discard)(sid)
    for ls in strategies:
        m = ls.manifest
        if (m.id, m.version) in known:
            continue  # a restart keeps the owner's choices
        registry.submit_candidate(
            VersionInfo(
                strategy_id=m.id,
                version=m.version,
                family=m.family,
                origin=m.origin,
                demo_only=True,  # paper trial in the demo, whatever the strategy is elsewhere
                code_hash=ls.code_hash,
                params=dict(m.param_values({})),
                created_by="owner",
            )
        )
        registry.promote_candidate(m.id, m.version, validation_passed=False, synthetic=True)
        if m.id in trade:
            registry.paper_trade(m.id, m.version)
    data = MemoryStageData()
    ev = Evaluator(registry, load_promotion_config(root / "config" / "promotion.yaml"), data, clock, router)
    lifecycle = LifecycleService(registry, ev, data, bus, clock)

    acfg = load_allocator_config(root / "config" / "allocator.yaml", root / "config" / "promotion.yaml")
    allocator = AllocatorService(Allocator(acfg, cfg.var / "allocation.json"), bus, clock, instruments)
    # strategies see scheduled high-impact news only where the calendar applies (paper/live, a real clock)
    news_fn = (
        news_from_index(news, {s: (instruments[s].base, instruments[s].quote) for s in symbols})
        if news is not None
        else None
    )
    engine = EngineLiveService(
        bus,
        clock,
        money=[(ls.cls, {}) for ls in strategies],
        history=history,
        news_fn=news_fn,
        shadow_signals=True,  # strategies switched off still show what they would trade
    )
    rules = next((ls for ls in library if ls.manifest.id == "smc_sniper"), None)
    ai = None
    if rules is not None and ai_tracks:
        key = settings.anthropic_api_key
        ai = AiTraderService(
            bus,
            clock,
            decider=ClaudeDecider(key.get_secret_value(), settings.ai_model) if key is not None else None,
            versions={ls.manifest.id: ls.manifest.version for ls in ai_tracks},
            rules=rules.cls,
            config=AiConfig(
                model=settings.ai_model,
                max_calls_per_day=settings.ai_max_calls_per_day,
                free_every_s=settings.ai_free_every_s,
            ),
            history=history,
            log_path=cfg.var / "ai_decisions.jsonl",
            env=settings.env,
            news_fn=news_fn,
        )

    stack = DemoStack(
        cfg=cfg,
        clock=clock,
        bus=bus,
        router=router,
        audit=audit,
        monitor=monitor,
        registry=registry,
        execution=execution,
        exec_service=exec_service,
        engine=engine,
        risk=risk,
        lifecycle=lifecycle,
        allocator=allocator,
        adapter=adapter,
        info={
            "mode": f"SIMULATED {cfg.symbol} · synthetic prices"
            if cfg.broker == "sim"
            else f"PAPER · {acct.trade_mode.upper()} ACCOUNT",
            "symbol": cfg.symbol,
            "strategy": ", ".join(ls.manifest.id for ls in strategies),
        },
        contract={s: instruments[s].contract_size for s in symbols},
        quote_ccy={s: instruments[s].quote for s in symbols},
    )
    stack.services = [lifecycle, risk, allocator, execution, engine, monitor, AuditService(audit)]
    stack.journal = JournalService(
        clock, symbols, path=cfg.var / "journal.jsonl", history=history, events=news, bus=bus
    )
    stack.services.append(stack.journal)
    journal = stack.journal
    stack.lessons = LessonBook(cfg.var / "lessons.jsonl")
    stack.services.append(
        LessonService(
            stack.lessons,
            lambda: list(journal.entries.values()),
            {ls.manifest.id: ls.manifest.family for ls in strategies},
            {ls.manifest.id: dict(ls.manifest.param_values({})) for ls in strategies},
        )
    )
    if ai is not None:
        stack.ai = ai
        stack.services.append(ai)
    stack.frame = frame
    stack.strategies = strategies
    stack.calendar, stack.news_applied = calendar, news is not None
    await om.initialize()
    for e in config_events("demo", clock.now()):  # every config file this process read, by hash
        await bus.publish(CONFIG, e)
    await lifecycle.publish_snapshot()
    await risk.start()
    await pump(bus, stack.services)
    return stack


def _quotes_of(row: dict[str, Any], symbol: str, digits: int) -> Iterator[Quote]:
    """Four quotes per M1 bar along the usual path: open, low, high, close on an up bar and open,
    high, low, close on a down bar (visiting the high first on every bar would favour one side's stops)."""
    t0 = row["open_time"]
    up = row["bid_c"] >= row["bid_o"]
    path = ("o", "l", "h", "c") if up else ("o", "h", "l", "c")
    for k, p in enumerate(path):
        yield Quote(
            symbol=symbol,
            bid=Decimal(str(round(row[f"bid_{p}"], digits))),
            ask=Decimal(str(round(row[f"ask_{p}"], digits))),
            time=t0 + timedelta(seconds=5 + 15 * k),
        )


class Scheduler:
    """Time-driven jobs, identical for the simulated and the real clock."""

    def __init__(self, s: DemoStack) -> None:
        self.s = s
        self.last: dict[str, datetime] = {}
        self.rollover_day: str | None = None

    def due(self, job: str, every: timedelta, now: datetime) -> bool:
        last = self.last.get(job)
        if last is None or now - last >= every:
            self.last[job] = now
            return True
        return False

    async def minute(self) -> None:
        s = self.s
        now = s.clock.now()
        if self.due("heartbeat", timedelta(seconds=30), now):
            # every in-process service is alive while this loop runs; the watchdog must see that before
            # it checks (only execution is pumped here, so buffered quotes and signals wait)
            for name in SERVICES_WITH_HEARTBEAT:
                await s.bus.publish(HEARTBEATS, Heartbeat(at=now, service=name))
            await pump(s.bus, [s.execution])
        await s.exec_service.cycle()  # deals, expiry, stops, watchdog, reconciliation (every 60 s)
        # the account goes out BEFORE this minute's signals are delivered: the risk gate refuses to
        # decide on account data older than 10 s (fail closed)
        await s.execution.cycle()
        await s.engine.on_time(now)
        if s.ai is not None:
            await s.ai.on_time(now)
        if s.journal is not None:
            await s.journal.on_time(now)
        local = now.astimezone(NY)
        day_key = local.date().isoformat()
        if local.hour >= 17 and self.rollover_day != day_key and local.weekday() < 5:
            self.rollover_day = day_key
            await s.risk.roll_day(new_week=local.weekday() == 4)
            s.monitor.daily_summary(now)
        if self.due("hourly", timedelta(hours=1), now):
            await s.lifecycle.hourly()
            check_chain(s.audit, s.router, now)
        await pump(s.bus, s.services)
        await s.router.tick()


async def play_sim(s: DemoStack, *, until: datetime | None = None, realtime: bool = False) -> int:
    """Play the synthetic market minute by minute. Weekends (no bars) are skipped. Returns minutes played."""
    frame, clock, broker = s.frame, s.clock, s.adapter
    if frame is None or not isinstance(clock, SimClock) or not isinstance(broker, FakeBroker):
        raise RuntimeError("play_sim needs the simulated market")
    sched = Scheduler(s)
    n = 0
    start = clock.now()
    digits = broker.symbol_info[s.cfg.symbol].digits
    for row in frame.iter_rows(named=True):
        t_end = row["open_time"] + timedelta(minutes=1)
        if row["open_time"] < start:
            continue
        if until is not None and t_end > until:
            break
        for q in _quotes_of(row, s.cfg.symbol, digits):
            clock.advance_to(q.time)
            broker.set_quote(q.symbol, q.bid, q.ask)
            await s.execution.on_quote(q)
        clock.advance_to(t_end)
        await sched.minute()
        if realtime and s.ai is not None:
            # the simulated market waits while Claude thinks: on a real clock only seconds would pass, but at
            # 300x a 10 s answer would be 50 market minutes late and refused as stale
            await s.ai.settle()
        n += 1
        # always yield, so the dashboard stays responsive while history plays at full speed
        await asyncio.sleep(60.0 / s.cfg.speed if realtime else 0)
    return n


async def play_live(s: DemoStack, stop: asyncio.Event) -> None:
    """Paper on a broker demo account: real quotes, real clock; jobs run every few seconds."""
    sched = Scheduler(s)

    async def quotes() -> None:
        async for q in s.adapter.stream_quotes(list(s.contract)):
            await s.execution.on_quote(q)
            if stop.is_set():
                return

    task = asyncio.create_task(quotes())
    try:
        while not stop.is_set():
            await sched.minute()
            await asyncio.sleep(5)
    finally:
        task.cancel()


async def refresh_calendar(cal: ForexFactoryCalendar, every: timedelta = timedelta(hours=1)) -> None:
    """Wall-clock refresh of the ForexFactory export (the feed asks not to be polled often)."""
    while True:
        ok = await asyncio.to_thread(cal.refresh, datetime.now(UTC))
        log.info("forexfactory calendar %s (%d events)", "updated" if ok else "not updated", len(cal.rows))
        await asyncio.sleep(every.total_seconds() if ok or cal.fresh(datetime.now(UTC)) else 300)


def say(text: str) -> None:
    print(text, flush=True)


async def run_demo(cfg: DemoConfig, owner_token: str | None = None) -> DemoStack:
    stack = await build(cfg)
    server = None
    if cfg.serve:
        import uvicorn  # noqa: PLC0415

        public = cfg.host not in ("127.0.0.1", "localhost", "::1")
        configured = Settings().api_owner_token
        token = owner_token or (configured.get_secret_value() if configured else None)
        if public and (token is None or len(token) < 32):
            raise SystemExit("a public hub needs AT_API_OWNER_TOKEN (32+ characters); it is never generated")
        shown = token is None  # a generated token is printed for the local console only
        token = token or secrets.token_urlsafe(32)
        server = uvicorn.Server(
            uvicorn.Config(
                stack.app(token, protect_reads=public), host=cfg.host, port=cfg.port, log_level="warning"
            )
        )
        asyncio.create_task(server.serve())  # noqa: RUF006
        if shown:
            say(f"dashboard: http://{cfg.host}:{cfg.port}/   owner token for control endpoints: {token}")
        else:
            say(f"dashboard on {cfg.host}:{cfg.port} (access token from AT_API_OWNER_TOKEN)")
    stack.router.send(
        Alert(severity=Severity.INFO, kind="demo_started", message=stack.info["mode"], at=stack.clock.now())
    )
    if stack.calendar is not None:
        asyncio.create_task(refresh_calendar(stack.calendar))  # noqa: RUF006
    try:
        if cfg.broker == "sim":
            catchup_end = stack.clock.now() + timedelta(days=cfg.catchup_days)
            say(f"playing {cfg.catchup_days} market days of history at full speed (watch the dashboard)...")
            n = await play_sim(stack, until=catchup_end)
            say(f"caught up {n} market minutes; now playing at {cfg.speed:g}x (Ctrl-C to stop)")
            await play_sim(stack, realtime=True)
        else:
            await play_live(stack, asyncio.Event())
    finally:
        if server is not None:
            server.should_exit = True
    return stack


def stage_of(stack: DemoStack, strategy_id: str = "demo_ma_cross") -> Stage:
    [v] = [v for v in stack.registry.versions() if v.info.strategy_id == strategy_id]
    return v.stage
