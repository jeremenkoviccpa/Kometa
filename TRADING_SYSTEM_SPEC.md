# Autonomous Trading System: Build Specification for Claude Code

Version 1.1, 2026-09-24. Owner: Nemanja.

### Changes in 1.1 (review pass)

- Section 0: the build process itself is self-improving (phase retros, lessons fed back into `CLAUDE.md`).
- Section 5: `account_id` and `tenant_id` on every money-bearing model (was only in section 21).
- Section 7: data versioning and a deterministic synthetic data generator so Phases 1 to 3 are not blocked on the data source decision.
- Section 9: Monte Carlo defaults to a block bootstrap (trade R is serially correlated); holdout epochs and a per-family holdout budget, so one holdout cannot be overfit by many versions.
- Section 11: risk decisions are signed with Ed25519 instead of a shared HMAC key, so a compromised execution service cannot forge approvals. Defined the equity reference for loss halts.
- Section 14: two new loops. L7 meta-learning tunes the research process itself (which agents, prompts and families earn research budget). L8 cost-model calibration learns spreads and slippage from real fills, asymmetrically (pessimistic changes apply automatically, optimistic ones need evidence). Every loop is scored on the value it adds, and loops that add none lose budget.
- Section 14.8: champion vs challenger tests are corrected for multiple concurrent challengers.
- Section 20: acceptance criteria updated for the above.

This document is the single source of truth for building the full infrastructure of an autonomous, self-learning, multi-strategy trading system for FX and metals (for example EURUSD, XAUUSD). Trading methods are NOT part of this build. They are added later as strategy plugins through the interface in section 6, or discovered by the self-learning layer in section 14. Your job is to build everything around them so that any strategy can be tested, promoted, sized, executed, monitored, improved and killed without human review.

---

## 0. Instructions for Claude Code

1. Build in the phase order of section 20. Do not start a phase until the previous phase's acceptance criteria pass.
2. Every module ships with tests. A phase is done only when `make check` (lint, type check, tests) passes.
3. Do not invent trading strategies. The only strategy you write is `examples/demo_ma_cross`, which exists purely to exercise the plumbing. Mark it clearly as not for live trading.
4. Never hardcode secrets, account numbers, API keys or broker credentials. Everything comes from environment variables or the secrets file described in section 17.
5. The risk gate (section 11) is the most important component. Treat any code path that could bypass it as a critical bug.
6. When this spec is ambiguous, choose the safer option (smaller size, fewer trades, halt instead of continue) and leave a `# SPEC-QUESTION:` comment plus an entry in `docs/open_questions.md`.
7. Keep a running `docs/decisions.md` log of design decisions you make that this spec does not cover.
8. Copy section 22 into the repo's `CLAUDE.md` at the start of Phase 0.
9. **Self-improving build.** At the end of every phase, write a short retro in `docs/retros/phase_<n>.md`: what broke, what took longest, which spec assumptions were wrong, which tests caught real bugs. Turn every recurring mistake into a rule in the "Lessons" section of `CLAUDE.md` or into an automated check (a test, lint rule or `make check` step). Prefer the automated check; a rule a machine enforces cannot be forgotten.

---

## 1. Principles

- **Strategy agnostic.** The platform knows nothing about any specific trading method. Strategies are plugins that emit signals.
- **One engine for everything.** The same event loop and the same strategy code run in backtest, shadow, paper and live. Only the clock, data feed and broker adapter change.
- **No lookahead, ever.** Strategies only see closed bars. The engine enforces this structurally, not by convention.
- **Deterministic.** The same data, code version and parameters must always produce the same signals and the same backtest result.
- **Risk is outside the strategy.** Strategies propose. The allocator sizes. The risk gate can reduce or reject, never increase. Execution only executes what the risk gate approved.
- **Automated judgement replaces human review.** Promotion and demotion are decided by fixed statistical gates in config, not by people or LLMs.
- **Self-learning inside fences.** The system continuously learns: it discovers strategies, re-tunes parameters, learns which signals to skip, adapts allocation and learns from its failures (section 14). Every learned change is a new version that must beat the current one on unseen data before it touches money. Learning can never modify risk limits, validation thresholds or promotion thresholds.
- **LLMs research, they do not trade.** Claude is used to generate and critique strategy code and features offline. No LLM call ever sits in the live order path.
- **Everything is audited.** Every signal, risk decision, order, fill, config change and stage change is written to an append-only, hash-chained log.
- **Fail closed.** On any unexpected state (data gap, reconciliation mismatch, lost heartbeat, bad config signature) the system stops opening positions and alerts.

---

## 2. Scope

### In scope

- Historical and live market data ingestion, storage and quality checks
- Event-driven engine with simulated, shadow, paper and live modes
- Realistic cost model: spread, commission, slippage, swap
- Validation suite: walk-forward, locked holdout, Monte Carlo, parameter stability, cross-market, deflated Sharpe
- Strategy lifecycle: registry, versioning, promotion ladder, automatic demotion and retirement
- Portfolio allocator across many strategy bots on one account
- Independent risk gate with signed config and kill switches
- Broker adapter layer with a simulated broker and a MetaTrader 5 adapter first
- Self-learning layer (section 14): strategy discovery agents using the Claude API, scheduled re-optimization with champion and challenger, meta-labeling models that learn which signals to take, regime detection, adaptive allocation, and a failure knowledge base. Enabled by default; everything it produces enters at Candidate and goes through the same gates
- Strategy intake tooling so a trader's method can be turned into a plugin
- Monitoring, alerts, dashboards, audit log
- Deployment with Docker Compose

### Out of scope for this build

- Any real trading strategy (added later)
- Multi-tenant white-label features (design for it, do not build it; see section 21)
- Mobile apps
- Any LLM decision in the live path

---

## 3. Tech stack

| Area | Choice | Notes |
| --- | --- | --- |
| Language | Python 3.12 | Type hints everywhere, `mypy --strict` |
| Packaging | `uv` workspace | One repo, several packages |
| Models and config | Pydantic v2, pydantic-settings | All domain objects are Pydantic models |
| Numerics | numpy, polars | Polars for bulk data, numpy in hot loops |
| Database | PostgreSQL 16 with TimescaleDB | Bars, ticks, trades, audit log, registry |
| DB access | SQLAlchemy 2 (async) with asyncpg, Alembic migrations | |
| Message bus | Redis Streams | Between engine, risk gate, execution, monitor |
| Scheduling | APScheduler inside a `scheduler` service | Nightly data, weekly research, evaluator jobs |
| API | FastAPI | Read-only status API plus owner-only control endpoints |
| Dashboards | Grafana on TimescaleDB | Provisioned dashboards as code |
| Alerts | Telegram bot (primary), email fallback | |
| LLM | Official `anthropic` Python SDK | Model name from config, never hardcoded |
| Machine learning | LightGBM, scikit-learn | Meta-labeling and regime models; fixed seeds |
| Vector store | pgvector extension in the same Postgres | Failure and lesson knowledge base for research agents |
| Sandbox | Docker containers with no network for generated strategy code | |
| Tests | pytest, hypothesis, pytest-asyncio | |
| Lint and format | ruff | |
| Deployment | Docker Compose; Linux VPS for core, Windows VPS for the MT5 bridge | |

---

## 4. Repository layout

```
autotrader/
  CLAUDE.md
  Makefile
  pyproject.toml                 # uv workspace root
  docker-compose.yml
  config/
    settings.example.env
    instruments.yaml             # contract sizes, pip sizes, sessions, swap rules
    validation.yaml              # pass thresholds (section 9)
    promotion.yaml               # ladder thresholds (section 10)
    allocator.yaml
    risk.yaml                    # signed, see section 11
    risk.yaml.sig
  packages/
    core/                        # domain models, clock, events, indicators, utils
    data/                        # ingestion, storage, quality checks, calendar, swaps
    engine/                      # event loop, MarketView, strategy runner, fill simulator
    strategies_api/              # Strategy base class, manifest schema, sandbox loader
    validation/                  # walk-forward, holdout, Monte Carlo, DSR, reports
    lifecycle/                   # registry, promotion ladder, evaluator
    allocator/
    risk/                        # risk gate service; separate package, separate owner
    execution/                   # broker adapters, order manager, reconciliation, watchdog
    mt5_bridge/                  # runs on Windows next to the MT5 terminal
    research/                    # Claude agents, prompts, trial registry, knowledge base
    learning/                    # re-optimization, meta-labeling, regime models, model registry
    monitor/                     # alerts, daily summaries, audit writer
    api/                         # FastAPI service
    cli/                         # `at` command line tool
  strategies/
    examples/demo_ma_cross/      # plumbing test only, never promoted past Shadow
    library/                     # human or trader supplied strategies (empty for now)
    generated/                   # research agent output, loaded only via sandbox
  grafana/
  migrations/
  tests/
    unit/  integration/  e2e/  golden/
  docs/
    decisions.md
    open_questions.md
    runbook.md
```

Package dependency rules (enforce with an import linter test):

- `core` depends on nothing internal.
- `strategies_api` depends only on `core`.
- Strategy code may import only `strategies_api`, `core.indicators`, `numpy`, `math`. Nothing else.
- `risk` depends only on `core`. Nothing depends on `risk` except `execution` through the bus message schema.
- `research` and `learning` must never import `risk`, `execution` or `allocator`. They may read `lifecycle` only through its public read API, and may submit new versions only through `lifecycle.submit_candidate()`.

---

## 5. Domain model

All models live in `packages/core/models.py` as frozen Pydantic models. Times are UTC `datetime` with timezone. Prices are `Decimal` at the boundary (broker, DB) and `float64` inside numeric hot loops.

```python
class Instrument(BaseModel):
    symbol: str                  # broker symbol, e.g. "XAUUSD"
    asset_class: Literal["fx", "metal", "index", "crypto"]
    base: str                    # "XAU"
    quote: str                   # "USD"
    contract_size: Decimal       # units per 1.0 lot, e.g. 100000 for EURUSD, 100 for XAUUSD
    pip_size: Decimal            # 0.0001 EURUSD, 0.01 USDJPY, 0.01 or 0.1 XAUUSD per broker
    tick_size: Decimal
    min_lot: Decimal
    lot_step: Decimal
    max_lot: Decimal
    commission_per_lot: Decimal  # round turn, account currency
    swap_long: Decimal           # per lot per night, account currency or points (see swap_mode)
    swap_short: Decimal
    swap_mode: Literal["points", "money", "percent"]
    triple_swap_weekday: int     # 2 = Wednesday for FX; broker specific for metals
    trading_sessions: list[SessionWindow]

class Bar(BaseModel):
    symbol: str
    timeframe: Timeframe         # M1, M5, M15, H1, H4, D1
    open_time: datetime
    close_time: datetime
    bid_o: float; bid_h: float; bid_l: float; bid_c: float
    ask_o: float; ask_h: float; ask_l: float; ask_c: float
    volume: float                # tick volume

class Signal(BaseModel):
    signal_id: UUID
    strategy_id: str
    strategy_version: str
    symbol: str
    side: Literal["buy", "sell"]
    entry_type: Literal["market", "limit", "stop"]
    entry_price: float | None    # required for limit and stop
    stop_price: float            # REQUIRED, no stop means rejected
    target_price: float | None
    expiry_bars: int | None      # pending order cancel after N bars of the signal timeframe
    created_at: datetime         # close_time of the bar that produced it
    reason: str                  # short human readable reason, stored in audit
    tags: dict[str, str] = {}

class OrderIntent(BaseModel):    # allocator output, risk gate input
    intent_id: UUID
    signal: Signal
    proposed_lots: Decimal
    risk_fraction: float         # fraction of equity at risk if stopped out

class RiskDecision(BaseModel):
    intent_id: UUID
    verdict: Literal["approve", "resize", "reject"]
    approved_lots: Decimal       # always <= proposed_lots
    reasons: list[str]
    limits_snapshot_hash: str    # hash of risk.yaml used
    decided_at: datetime

class Order(BaseModel): ...      # client_order_id, broker_order_id, status, lots, prices, sl, tp, timestamps
class Fill(BaseModel): ...       # order ref, price, lots, commission, spread_at_fill, latency_ms
class Position(BaseModel): ...   # symbol, side, lots, avg_price, sl, tp, opened_at, strategy_id, swap_accrued
class Trade(BaseModel): ...      # closed round trip: entry, exit, pnl gross, costs, net, R multiple, MAE, MFE

class Stage(StrEnum):
    CANDIDATE = "candidate"; SHADOW = "shadow"; MICRO = "micro"
    LIVE = "live"; SCALED = "scaled"; RETIRED = "retired"; DEMO_ONLY = "demo_only"
```

`Order`, `Fill`, `Position` and `Trade` also carry `account_id` and `tenant_id` (default `"default"`) from day one (section 21); retrofitting them later touches every table.

Every `Trade` stores its R multiple: net P and L divided by the money at risk at entry (entry to stop distance times lots times value per point). R is the main unit used across validation and promotion.

---

## 6. Strategy plugin interface

A strategy is a Python class plus a manifest. It receives closed bars and returns signals. It has no access to money, the broker, the database, the network, the file system or other strategies.

```python
class MarketView(Protocol):
    now: datetime                                     # close_time of the bar being processed
    def bars(self, symbol: str, tf: Timeframe, n: int) -> BarsArray: ...  # closed bars only, oldest first
    def spread(self, symbol: str) -> float: ...       # current or modelled spread

class StrategyContext(Protocol):
    market: MarketView
    params: Mapping[str, float | int | str | bool]    # frozen for this run
    state: MutableMapping[str, Any]                   # persisted per strategy version, JSON serializable
    def my_positions(self, symbol: str | None = None) -> list[PositionView]: ...
    def my_pending(self, symbol: str | None = None) -> list[PendingView]: ...

class Strategy(ABC):
    manifest: StrategyManifest

    def warmup(self) -> dict[tuple[str, Timeframe], int]:
        """Bars needed per series before on_bar is called."""

    @abstractmethod
    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Signal | CancelRequest]:
        """Called once per closed bar of every subscribed series."""

    def on_fill(self, ctx: StrategyContext, fill: FillView) -> list[Signal | CancelRequest]:
        return []
```

Manifest (`strategy.yaml` next to the code):

```yaml
id: s1_pinbar_sr_flip          # stable across versions
version: 1.0.0
origin: trader | research_agent | owner | learning_reopt
family: pinbar_sr              # groups variants for the trial registry (section 9)
symbols: [EURUSD, GBPUSD, XAUUSD]
timeframes: [H1, M5]
params:                        # max 6 tunable params; each needs bounds
  ema_len: {value: 21, min: 10, max: 50, tunable: true}
  wick_ratio: {value: 0.66, min: 0.5, max: 0.8, tunable: true}
expected:                      # hypothesis, used to detect live drift
  trades_per_month: 12
  win_rate: 0.45
  avg_r: 0.35
intake_ref: intake/2026-10-trader-x.yaml   # optional
description: short plain text
```

### Indicator and pattern library (`core.indicators`)

Build these now so traders' methods can be expressed later without new infrastructure. Each function exists in a vectorized form (for research) and an incremental form (for live), and a test proves both give identical output.

- Moving averages: SMA, EMA, WMA; slope over N bars
- Volatility: ATR, true range, rolling standard deviation, ATR percentile
- Oscillators: RSI, MACD, stochastic
- Bands: Bollinger, Keltner, Donchian
- Structure: swing highs and lows (fractal with left and right N), higher highs and lower lows sequence, trend state from swings
- Levels: support and resistance finder that clusters swing points within k times ATR and reports touch count, last touch, and whether the level has flipped (broken by a close and retested from the other side)
- Candle anatomy: body, upper wick, lower wick, range, and their ratios; inside bar, outside bar, engulfing
- Geometry helpers for chart patterns: fit a line through swing points, measure convergence of two lines, breakout of a line by a close. These are building blocks for flags, triangles and double tops; the patterns themselves are strategies and are NOT built now
- Session and calendar helpers: session of a timestamp, minutes to next high-impact event for a currency

### Lookahead protection

- `MarketView` is backed by a buffer that only ever contains bars with `close_time <= ctx.market.now`. There is no API to reach later data.
- Multi-timeframe: an H1 bar is visible to an M5 strategy only after the H1 bar has closed.
- Test (mandatory, runs on every strategy in CI and in the sandbox): the future poisoning test. Run the strategy on data up to time T, then run it again on data where every bar after T is replaced with random garbage. Signals up to T must be byte for byte identical.

### Static rules for strategy code (checked by AST before loading)

- Allowed imports: `strategies_api`, `core.indicators`, `core.models`, `numpy`, `math`, `dataclasses`, `typing`, `enum`
- Forbidden: `open`, `eval`, `exec`, `__import__`, `os`, `sys`, `subprocess`, `socket`, `time`, `datetime.now`, `random` without the seeded RNG from context, threads, async
- Max 6 tunable params. Max file size 2,000 lines.

`strategies/examples/demo_ma_cross` implements a trivial EMA crossover to exercise the whole pipeline. Its manifest sets `origin: owner` and a `demo_only: true` flag that caps it at the Shadow stage forever.

---

## 7. Data layer

### Sources (behind a `DataSource` interface)

| Source | Use | Status |
| --- | --- | --- |
| MT5 history via the bridge | Broker's own bars and spreads for the traded symbols | Build first |
| Dukascopy historical tick archive | Long bid and ask history for backtests | Candidate; confirm terms of use and coverage in Phase 0 |
| CSV and Parquet import | Anything else | Build |
| Live quotes from the broker adapter | Live bar building | Build |

### Storage (TimescaleDB)

- `instruments` from `config/instruments.yaml`, versioned
- `ticks` hypertable (optional, compressed after 7 days)
- `bars_m1` hypertable with bid and ask OHLC and tick volume; the canonical store
- Higher timeframes built from M1 as continuous aggregates. D1 bars close at 17:00 New York time (FX convention); make the daily boundary configurable per instrument
- `spread_stats` per symbol and hour of week: median and 90th percentile spread
- `swap_history` per symbol and day
- `calendar_events`: time, currency, impact (low, medium, high), name
- `data_quality_issues`: symbol, time range, issue type, severity

### Quality checks (nightly job and on import)

Flag and store, never silently fix: gaps inside trading sessions, ask below bid, zero or negative spread, spikes larger than 8 times ATR that revert within 3 bars, duplicate timestamps, stale quotes (no change for 10 minutes in session). Validation windows that contain high severity issues are marked and excluded from pass thresholds.

### Data versioning

Every dataset used by a backtest is identified by a `data_version`: a hash over (symbol, timeframe, first and last timestamp, row count, content hash per month partition). The data version is stored with every trial, validation report and model. Determinism (section 1) is defined as: same code hash, same `data_version`, same params, same config hashes give the same result.

### Synthetic data generator

`data.synthetic` generates seeded M1 bid and ask bars (regime switching random walk with sessions, spread by hour of week, weekend gaps, occasional spikes and gaps for the quality checks). It is used by CI, golden tests and Phases 1 to 3, so they do not depend on the historical data source decision. Synthetic data is never used for promotion decisions; the registry rejects validation results whose `data_version` is synthetic.

### Economic calendar

`CalendarSource` interface with one implementation chosen in Phase 0 (open question). Used by the risk gate news blackout and by strategy helpers.

### Live bar builder

Consumes quotes from the broker adapter, builds M1 bid and ask bars, emits `BarClosed` events for every subscribed timeframe. Nightly job compares live built bars to the broker's own bars and logs any differences above 1 tick.

---

## 8. Engine

### Events

`BarClosed`, `QuoteUpdate` (live modes only), `SignalEmitted`, `OrderIntentCreated`, `RiskDecided`, `OrderPlaced`, `OrderModified`, `OrderFilled`, `OrderCancelled`, `PositionClosed`, `StageChanged`, `HaltEntered`, `HaltCleared`, `ConfigChanged`, `ModelVersionActivated`.

### Modes

| Mode | Clock | Data | Broker | Purpose |
| --- | --- | --- | --- | --- |
| backtest | SimClock | Historical | SimBroker | Validation and research |
| shadow | LiveClock | Live feed | ShadowBroker (records would-be fills from live quotes) | Stage 1 of the ladder, champion vs challenger comparisons |
| paper | LiveClock | Live feed | Broker demo account | Execution testing |
| live | LiveClock | Live feed | Real broker account | Trading |

The strategy runner, allocator and risk gate code are identical in all modes. In backtest they run in process for speed; in live they run as separate services over Redis Streams. Both paths use the same classes; only the transport differs.

### Fill simulator (SimBroker and ShadowBroker)

- Market order: fills at the next bar open, ask for buys and bid for sells, plus a slippage sample from the model (default: 0.2 times median spread, configurable per symbol)
- Limit order: fills only when price trades through the limit by at least 1 tick, at the limit price
- Stop entry: fills at the stop price plus slippage; if the bar gaps through, fills at the open
- Stop loss and take profit on the same bar: resolve with M1 data; if M1 cannot resolve it, assume the stop was hit first
- Swap charged at the instrument's rollover time, triple on the configured weekday
- Commission per lot per side from `instruments.yaml`
- Spread at fill taken from `spread_stats` for that hour of week, widened by 3x within 2 minutes of high impact news and at rollover
- Partial fills are not simulated in v1 (record this in `docs/decisions.md`)

### Performance target

Backtest of one strategy on one symbol, 10 years, H1 signals with M1 fill resolution: under 2 minutes on one core. Parallelize across strategies, symbols and walk-forward windows with a process pool.

---

## 9. Validation suite

Everything configurable lives in `config/validation.yaml`. Defaults:

```yaml
walk_forward:
  train_years: 3
  test_months: 6
  step_months: 6
holdout:
  months: 12                  # locked window, see "Locked holdout"
  max_attempts_per_version: 1
  max_attempts_per_family_per_epoch: 5
monte_carlo:
  runs: 5000
  method: block_bootstrap_trades  # resample blocks of consecutive trade R, preserve count
  block_length: auto              # default: round(n_trades ** (1/3))
stability:
  param_shift: 0.20           # every tunable param moved +20% and -20%
cross_market:
  min_pairs_passing: 3
  of_pairs: 5
thresholds:
  min_oos_trades: 200
  min_profit_factor: 1.3
  min_deflated_sharpe_prob: 0.95
  max_mc_dd_p95_at_half_percent_risk: 0.12
  min_stability_profit_factor: 1.1
  min_holdout_profit_factor: 1.1
  suspicious_monthly_return: 0.10   # above this in backtest -> flag as probable bug, block promotion
```

### Walk-forward

Tune tunable params on the train window (grid or Bayesian search, bounded budget), evaluate on the next test window, roll. Only concatenated out-of-sample segments count toward thresholds. Store the chosen params per window; large jumps between windows are a stability warning in the report.

### Locked holdout

The holdout window is stored as a separate data permission. Research and learning processes have no read access to it. The validation service opens it once per strategy version, records the attempt in `holdout_attempts`, and refuses a second attempt for the same version.

One attempt per version is not enough on its own: a family that submits 100 versions still overfits the holdout by selection. Therefore:

- **Holdout epochs.** The holdout is a fixed window, not "the latest 12 months". It is re-cut once per epoch (default yearly, owner command). When a new epoch starts, the old holdout becomes ordinary research data and the new one is locked. The epoch id is stored with every attempt.
- **Family budget.** At most `max_attempts_per_family_per_epoch` holdout attempts per family per epoch. Further versions wait for the next epoch.
- **Counted.** Holdout attempts are also recorded as trials, so they raise N in the deflated Sharpe of the family.

Monte Carlo uses a block bootstrap because trade outcomes cluster (volatility regimes, streaks). A plain i.i.d. bootstrap understates drawdown tails.

### Deflated Sharpe ratio

Computed on daily returns of the strategy equity curve. Implement in `validation/dsr.py` with unit tests against hand computed examples.

```
Probabilistic Sharpe Ratio:
PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T - 1) / sqrt(1 - skew * SR_hat + ((kurt - 1) / 4) * SR_hat^2) )

Deflated benchmark for N trials:
SR* = sqrt(Var(SR_trials)) * ( (1 - g) * Phi_inv(1 - 1/N) + g * Phi_inv(1 - 1/(N * e)) )

Phi = standard normal CDF, Phi_inv its inverse, g = 0.5772 (Euler-Mascheroni),
kurt = non-excess kurtosis, T = number of daily returns,
N = trials in the same family from the trial registry, Var(SR_trials) = variance of their Sharpe ratios.
```

### Trial registry

Every backtest run of every variant is recorded in `trials` (family, version, params hash, data window, Sharpe, trade count, passed or failed). N for the deflated Sharpe is the count of trials in the strategy's family. Deleting trials is impossible (append-only table). This is what stops the self-learning layer from fooling itself by trying thousands of variants.

### Reports

Each validation run writes a JSON result and an HTML report: equity and drawdown curves, monthly returns table, R distribution, results per symbol, per session and per weekday, walk-forward parameter drift, Monte Carlo drawdown distribution, cost breakdown, and a header with code hash, data version, config hash and pass or fail per threshold.

---

## 10. Lifecycle and promotion ladder

### Registry tables

`strategies` (id, family, origin), `strategy_versions` (id, version, code hash, manifest, params, parent version, created by), `stage_history` (version, from, to, reason, metrics snapshot, timestamp), `holdout_attempts`, `trials`.

### State machine

```
candidate -> shadow -> micro -> live -> scaled
scaled -> live            (rolling stats slip)
live -> micro             (any demotion rule)
micro -> retired          (second demotion within 6 months)
shadow -> retired         (signals diverge from backtest)
any -> retired            (owner command)
```

Illegal transitions raise and alert. `demo_only` versions can never leave shadow.

### Thresholds (`config/promotion.yaml`)

```yaml
shadow:
  risk_per_trade: 0.0
  exit: {min_weeks: 4, min_signals: 30, band: 0.90}   # signal rate and simulated R inside backtest 90% band
micro:
  risk_per_trade: 0.001
  exit: {min_trades: 60, max_slippage_vs_model: 1.5, band: 0.90}
live:
  risk_per_trade: 0.005
  exit: {min_trades: 150, min_rolling_sharpe: 1.0}
scaled:
  risk_per_trade_max: 0.01    # actual value set by allocator
demotion:
  dd_vs_mc_p95: 1.5
  rolling_pf_window: 50
  rolling_pf_min: 1.0
  slippage_vs_model_max: 2.0
  slippage_window: 20
  drift_window: 50
  drift_interval: 0.95        # win rate or avg R outside backtest 95% interval
  retire_after_demotions: 2
  retire_window_months: 6
global:
  max_promotions_to_live_per_week: 2
```

### Evaluator

Runs every hour and after every closed trade. Computes stage metrics per version from the trade table, applies the thresholds, writes `stage_history`, emits `StageChanged`, sends alerts. Demotions take effect immediately: pending orders of a demoted version are cancelled and its open positions are closed at market if it drops to shadow or retired, or kept with their stops if it drops one live stage.

---

## 11. Risk gate

The risk gate is a separate service (`packages/risk`, container `risk-gate`) and the only path from an order intent to the broker. Execution refuses any order that does not carry a matching `RiskDecision` with a valid Ed25519 signature from the risk gate.

Why a signature and not an HMAC: with a shared HMAC key, execution holds the same secret as the risk gate and a compromised execution service could mint its own approvals. With Ed25519, only the risk gate holds the decision signing key; execution holds only the public key. Each decision also carries a monotonic sequence number and an expiry (default 5 seconds); execution rejects replays and stale decisions.

### Signed config

- Limits live in `config/risk.yaml`. At startup the gate verifies `risk.yaml.sig` (Ed25519) against the owner public key baked into the image. Bad or missing signature: refuse to start, alert.
- The private key never lives on any server. The owner signs with `at risk sign config/risk.yaml` on their own machine.
- No other process, including research, learning and the API, has write access to the risk config or the risk package code at runtime.

### Default limits (`config/risk.yaml`)

```yaml
account_currency: EUR
risk_per_trade_default: 0.005
risk_per_trade_max: 0.01
open_risk_per_strategy_max: 0.015
open_risk_total_max: 0.03
same_currency_same_direction_max_positions: 2
leverage_notional_max: 10.0          # total notional / equity
per_symbol_max_lots: {XAUUSD: 2.0, default: 5.0}
daily_loss_halt: 0.02                # close all, halt until next trading day
weekly_loss_halt: 0.05               # close all, halt until next week
peak_drawdown_full_halt: 0.15        # close all, full halt, owner resume only
news_blackout_minutes: 15            # before and after high impact events on either currency
no_new_entries_after: "FRI 18:00 Europe/Belgrade"
min_free_margin_ratio: 3.0           # free margin must stay above 3x required margin after the trade
require_stop: true
```

### Checks, in order

1. Halt state allows entries
2. Signal has a stop, stop is on the correct side, stop distance at least 1.5 times current spread
3. Independent position size calculation (never trusts the allocator number)
4. Per trade, per strategy and total open risk limits
5. Same currency exposure
6. Leverage and per symbol lot caps
7. Margin check using live account margin data
8. News blackout and weekend rule
9. Strategy version stage allows trading and risk fraction does not exceed the stage limit

Result: approve, resize (smaller only) or reject, with reasons. Every decision goes to the audit log.

### Position sizing

```
money_at_risk = equity * risk_fraction
lots = money_at_risk / (abs(entry - stop) * contract_size * fx_to_account_ccy)
lots = floor_to_step(lots, lot_step)
if lots < min_lot: reject (never round up)
```

Worked example to include as a unit test: XAUUSD, contract size 100 oz, entry 4342.00, stop 4332.00 (10 dollars), account 20,000 USD equivalent, risk 0.5 percent = 100 USD. Lots = 100 / (10 x 100) = 0.10 lot. The same setup with 9 lots would risk 9,000 USD, 45 percent of the account, and must be rejected.

### Halt states

Loss references: daily loss is measured against equity (including floating P and L) at the last 17:00 New York rollover; weekly loss against equity at the Sunday open; peak drawdown against the highest end-of-day equity ever recorded. All three are recomputed on every account update, not only on closed trades.

`NORMAL`, `DAILY_HALT`, `WEEKLY_HALT`, `RECON_HALT` (reconciliation mismatch), `FULL_HALT`. Entering any halt cancels pending entries; daily, weekly and full halts also close open positions at market. `FULL_HALT` clears only with an owner signed resume token (`at risk resume`). Halts are persisted so a restart never clears them.

### Mandatory property tests (hypothesis)

- `approved_lots <= proposed_lots` for all inputs
- No approval without a valid stop
- No approval in any halt state
- Total open risk after approval never exceeds `open_risk_total_max`
- Tampered `risk.yaml` prevents startup

---

## 12. Execution and brokers

### Broker adapter interface

```python
class BrokerAdapter(Protocol):
    async def connect(self) -> None: ...
    async def account(self) -> AccountInfo: ...            # balance, equity, margin, free margin, currency
    async def symbols(self) -> list[SymbolInfo]: ...
    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]: ...
    async def history_bars(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> list[Bar]: ...
    async def place(self, req: PlaceRequest) -> BrokerAck: ...     # always with SL, TP optional
    async def modify(self, req: ModifyRequest) -> BrokerAck: ...
    async def cancel(self, broker_order_id: str) -> BrokerAck: ...
    async def close_position(self, position_id: str, lots: Decimal | None = None) -> BrokerAck: ...
    async def open_positions(self) -> list[BrokerPosition]: ...
    async def pending_orders(self) -> list[BrokerOrder]: ...
    async def deals(self, since: datetime) -> list[BrokerDeal]: ...
```

### Implementations

| Adapter | Phase | Notes |
| --- | --- | --- |
| SimBroker | 2 | Backtests |
| ShadowBroker | 6 | Would-be fills from live quotes |
| MT5Adapter | 5 | Talks to `mt5_bridge` over an authenticated private tunnel |
| cTrader or OANDA adapter | later | Only if the chosen broker needs it |

### MT5 bridge (`packages/mt5_bridge`)

- The official `MetaTrader5` Python package works together with a running MT5 terminal on Windows. Confirm this in Phase 0 and record the result in `docs/decisions.md`. Plan for a small Windows VPS running the terminal plus the bridge.
- The bridge exposes the adapter methods over HTTPS with mutual TLS or over a WireGuard or Tailscale tunnel. It holds no strategy logic and no risk logic.
- Each strategy version gets its own MT5 magic number; the order comment carries the `client_order_id`. This makes every broker position traceable to a strategy version.

### Order manager

- `client_order_id` is derived from the risk decision id, so retries are idempotent: before placing, check open orders and positions for that id
- Stop loss is attached at placement. After every fill, confirm the broker position has a stop. If not, set it; if that fails twice, close the position and alert
- Pending orders expire per the signal's `expiry_bars`
- Log fill price, spread at fill, requested vs filled price and latency to `execution_quality`

### Reconciliation and watchdog

- Every 60 seconds compare broker positions, orders and equity with internal state. Any mismatch: enter `RECON_HALT`, alert, attempt automatic resync once; if still mismatched, stay halted.
- Positions opened outside the system (for example a manual trade from the phone app) are flagged as `external`, excluded from strategy stats, counted in exposure limits, and alerted.
- Watchdog: every service sends a heartbeat every 30 seconds. If the engine or risk gate is silent for 2 minutes, the watchdog cancels all pending entry orders directly through the adapter and alerts. Open positions stay protected by their broker side stops.

---

## 13. Allocator

Turns approved signals into proposed lot sizes by splitting the total risk budget across live strategy versions. It is also self-learning loop L5 (section 14).

1. Each version's weight is proportional to its live Sharpe ratio shrunk toward its backtest Sharpe: `w_raw = (n / (n + k)) * sharpe_live + (k / (n + k)) * sharpe_backtest`, with n = live trades and k = 150.
2. Negative or zero weights get zero allocation.
3. Versions whose daily returns correlate above 0.6 are clustered and share one budget slot.
4. No version gets more than 30 percent of the total risk budget.
5. The resulting per trade risk fraction is capped by the version's stage limit from `promotion.yaml`.
6. Rebalance weekly; never intraday. New versions in micro get the micro stage fraction regardless of weight.

Config in `config/allocator.yaml`. The allocator proposes; the risk gate decides.

---

## 14. Self-learning system

The system improves itself through eight learning loops. All of them follow the same four rules:

1. **Nothing learned goes live directly.** Every learned change produces a new strategy version or model version that enters as a challenger.
2. **Champion vs challenger.** The challenger runs in shadow next to the current champion on the same live data. It replaces the champion only if it wins on unseen data by the rule in 14.8.
3. **Every attempt is counted.** All variants tried are written to the trial registry, so the deflated Sharpe gets harder to pass the more the system tries.
4. **Learning cannot touch the fences.** No loop can modify `risk.yaml`, `validation.yaml`, `promotion.yaml`, the risk package, the holdout data or the audit log.

### 14.1 Overview

| Loop | What it learns | From | Cadence | Output |
| --- | --- | --- | --- | --- |
| L1 Discovery | New strategies and variants | Strategy library, trade journal, knowledge base | Weekly | New candidate versions |
| L2 Re-optimization | Better parameters for existing strategies | Rolling recent data | Monthly per strategy | Challenger versions |
| L3 Meta-labeling | Which signals of a strategy to take, and a size reduction factor | The strategy's own signal history and outcomes | Monthly retrain | Meta model versions |
| L4 Regime | Market regime labels and which strategies work in which regime | Market features | Weekly retrain | Regime model versions and enable or disable filters |
| L5 Allocation | How much risk each strategy deserves | Live trade results | Weekly | Allocator weights (section 13) |
| L6 Failure learning | Why strategies failed | Demotions, retirements, rejected candidates | On every event | Lessons in the knowledge base, fed to L1 |
| L7 Meta-learning | Which research agents, prompt versions and families produce survivors | Trial registry, loop scorecard | Weekly | Research budget split, prompt challengers |
| L8 Cost calibration | Real spreads, slippage and swaps | `execution_quality`, broker swap history | Weekly | New cost model version (asymmetric, see 14.10) |

```
live trades + market data
        |
        v
  trade journal + feature store  ---> L3 meta models, L4 regime models
        |                                     |
        v                                     v
  L6 failure lessons ---> L1 discovery ---> candidates ---> validation ---> ladder
                         L2 re-opt   ---> challengers ---> shadow A/B ---> swap champion
        ^                                                                     |
        |______________________ results feed back ____________________________|
```

### 14.2 Trade journal and feature store

Every signal (taken or not) and every trade is stored with a feature snapshot at signal time: ATR and ATR percentile, trend slope on each timeframe, distance to nearest level in ATR, session, hour, weekday, spread vs median, minutes to next high impact event, recent realized volatility, the strategy's rolling win rate, and any strategy provided tags. Outcomes are attached when known: R multiple, MAE, MFE, bars held, and the triple barrier label (target hit first, stop hit first, or time out). Features are computed only from data available at signal time; the future poisoning test from section 6 also runs on the feature pipeline.

### 14.3 L1 Discovery agents (Claude API)

Agents live in `packages/research`. Each is a prompt template in `research/prompts/` plus a Python orchestrator. Model names come from config.

| Agent | Input | Output |
| --- | --- | --- |
| Hypothesis | Strategy library, trade journal summaries, knowledge base lessons (retrieved via pgvector), live performance | A written hypothesis with expected trades per month, win rate and average R, and the family it belongs to |
| Coder | Hypothesis, `strategies_api` docs, indicator library docs | Strategy code, manifest and unit tests |
| Critic | Code, manifest, validation report | Checks for lookahead, parameter count, cost blind spots, suspiciously good results. Can veto with reasons |
| Feature scout | Journal and meta model importances | Proposes new features for L3 and L4 as code in the feature store |

Rules:

- Generated code goes to `strategies/generated/<family>/<id>/<version>/` and is only ever loaded through the sandbox: a Docker container with no network, read-only data mount without the holdout, CPU and memory limits, 10 minute timeout.
- AST static checks from section 6 run before anything is executed.
- Budgets in `config/research.yaml`: max 200 variants per week, max API spend per week, max 3 new families per week.
- Any price data shown to an LLM is anonymized: symbol removed, prices rescaled to start at 100, timestamps shifted. This stops the model from recalling what really happened next.
- Every prompt, response, token count and cost is stored with the version it produced.

### 14.4 L2 Re-optimization

For each live strategy, monthly: re-run the walk-forward optimizer on the most recent window (default 3 years), produce a challenger with new params only if they differ from the champion by more than the stability band, run full validation, then shadow A/B for at least 4 weeks. Parameter changes are capped at 25 percent per step to prevent lurching.

### 14.5 L3 Meta-labeling

For each strategy with at least 300 historical signals with outcomes:

- Train a LightGBM classifier to predict the probability that a signal hits target before stop, using features from 14.2.
- Cross-validation must be purged and embargoed (drop training samples whose outcome window overlaps the test window, plus an embargo of 1 percent of samples after each test fold) to prevent leakage.
- Output is a take or skip decision plus a size factor in [0, 1]. The meta model can only reduce or skip; it can never increase size above what the allocator proposes.
- A meta model version activates only if, out of sample, the filtered strategy has higher expectancy in R per signal and a higher deflated Sharpe than the unfiltered strategy, and it then wins the shadow A/B.
- Models are stored in `model_registry` with training data window, feature list, hyperparameters, seed, metrics and file hash. Inference is deterministic.

### 14.6 L4 Regime detection

A regime model classifies each day per symbol into a small number of regimes (for example trending, ranging, high volatility, low volatility) from returns, ATR percentile, trend strength and cross-pair correlation. Start with a simple, explainable method (rule based thresholds, then a Gaussian mixture or hidden Markov model as a challenger). Per strategy, the system learns which regimes it performs badly in from the journal and proposes a regime filter as a new strategy version, which must pass validation like any other version.

### 14.7 L6 Failure knowledge base

On every demotion, retirement, Critic veto and failed validation, a Diagnostician agent writes a structured lesson: what was tried, what failed, which metric broke, suspected cause, and evidence (links to reports and trades). Lessons are stored in `lessons` with pgvector embeddings. The Hypothesis agent must retrieve the top lessons for its idea's family before proposing, and must explain why the new idea avoids them.

### 14.8 Champion vs challenger rule

A challenger replaces the champion only if all of these hold:

- It passed full validation (section 9) as its own version
- In shadow, over at least 4 weeks and at least 40 signals for both, its average R per signal is higher, and a one sided bootstrap test on the difference gives p below 0.10
- Its drawdown in shadow is not worse than 1.2 times the champion's
- At most one champion swap per strategy per month
- If k challengers of the same strategy are in shadow at the same time, the p threshold is 0.10 / k (Bonferroni), so running more challengers does not buy more lucky swaps
- Every swap test, passed or failed, is recorded as a trial

On swap, the old champion is kept in shadow for 4 more weeks. If the new champion triggers any demotion rule in that period, the system rolls back automatically.

### 14.9 Learning safety switches

- `learning.freeze: true` in config stops all loops without affecting live trading
- If total equity drawdown exceeds 8 percent, loops L1 to L4 pause automatically until equity recovers to within 4 percent of peak (learning during a drawdown tends to chase noise)
- A weekly learning report is sent: variants tried, passed, promoted, swapped, rolled back, retired, lessons added, API cost

### 14.10 L7 Meta-learning: improving the improver

The research process is itself a system with parameters: which agent prompts, which families, which hypothesis sources (journal, lessons, library variants) and how much budget each gets. L7 tunes these, never the fences.

- **Prompt versions.** Every prompt in `research/prompts/` is versioned like a strategy. A prompt change is a challenger: it runs on a share of the weekly research budget and is scored by the survival rate of what it produces.
- **Budget as a bandit.** The weekly variant and API budget (fixed totals in `config/research.yaml`) is split across (agent prompt version, family) arms with Thompson sampling on the survival rate. Survival is measured downstream, in order of weight: passed validation, reached micro, reached live, live R after 100 trades. A 20 percent exploration floor keeps new arms alive.
- **Family saturation.** A family whose trial count makes its deflated Sharpe benchmark unreachable at the current best Sharpe is marked saturated and gets no new budget until a lesson or regime change reopens it.
- **Loop scorecard.** Every loop (L1 to L8) reports the value it added each month in R: live R of versions it produced minus the R of the champions they replaced, meta model R saved by skipped losers minus R lost on skipped winners, and so on. A loop whose scorecard is negative for 3 consecutive months has its compute and API budget halved and an alert is sent. The scorecard is part of the weekly learning report.
- **Fences.** L7 only moves budget and prompt versions. It cannot change totals, thresholds, risk, the holdout or the ladder.

### 14.11 L8 Cost-model calibration

Backtests are only as honest as their cost model. L8 re-estimates the model weekly from real data: spread by hour of week from live quotes, slippage from `execution_quality` (requested vs filled, by symbol, session and order type), swaps from broker history.

- **Asymmetric updates.** A new cost model version that is more pessimistic (higher costs) activates automatically. One that is more optimistic needs at least 200 fills per affected symbol and an owner-visible alert, and activates only after 4 weeks of consistent evidence.
- **Revalidation.** When a new cost model version activates, all versions at micro and above are re-run under it. Any version that no longer meets the validation thresholds is demoted one stage by the normal evaluator.
- Cost model versions live in `model_registry` and their hash is part of every trial.

### 14.12 Not in v1

Reinforcement learning agents that trade directly are not built. They need far more data than one account produces and overfit easily. They can be added later as ordinary challengers that go through the same gates.

---

## 15. Strategy intake (for traders' methods, added later)

Trading methods arrive later from traders or the owner. Build the tooling now so adding one is fast.

- `docs/intake_template.yaml` with these required fields: market and timeframes, trend filter, setup definition in numbers, level definition, entry (order type and price), stop placement, target rule, invalidation, no-trade times, expected trades per month, expected win rate and average R. Every field must be measurable; free text like "when it feels strong" is rejected by the validator.
- `docs/trader_interview.md`: a question list that walks a trader through every field, including "show me 10 past examples with dates and prices".
- `at strategy new --from intake/<file>.yaml` scaffolds `strategies/library/<id>/` with manifest, a code skeleton that has the rules as comments, and a test file that checks the trader's example trades are reproduced by the code.
- `at strategy check <id>` runs static checks, the future poisoning test and the example trade test.
- `at strategy submit <id>` registers the version as a candidate and queues validation.

---

## 16. Monitoring, alerts and audit

### Audit log

- Table `audit_log`: id, timestamp, event type, actor (service or owner), payload (canonical JSON), `prev_hash`, `hash = sha256(prev_hash + payload)`.
- A database trigger rejects UPDATE and DELETE on this table. The application database role has INSERT and SELECT only.
- `at audit verify` walks the chain and reports any break.
- Logged: every signal, allocator proposal, risk decision, order, fill, cancel, stage change, halt, config change, model activation, champion swap, research agent run.

### Alerts (Telegram)

| Severity | Events |
| --- | --- |
| Critical | Any halt, reconciliation mismatch, missing stop after fill, heartbeat lost, bad config signature, external position detected, audit chain break |
| Warning | Demotion, rollback, data quality issue on a traded symbol, slippage above model, API budget 80 percent used |
| Info | Promotion, champion swap, daily summary, weekly learning report |

Critical alerts repeat every 10 minutes until acknowledged with `/ack` in the Telegram chat.

### Daily summary (after the 17:00 New York close)

Equity, day and month P and L, drawdown from peak, open risk, per version P and L and R, trades taken and skipped (with meta model skips), halts, execution quality.

### Grafana dashboards (provisioned from `grafana/`)

Account overview, strategy versions by stage, live vs backtest drift, execution quality, risk utilization, learning activity, data quality.

### API (`packages/api`)

Read-only endpoints for status, positions, versions, stages, reports and audit queries. Control endpoints (pause learning, retire a version, request resume) require an owner token and write to the audit log. There is no API endpoint that changes risk limits.

---

## 17. Config and secrets

- `pydantic-settings` loads from environment. `config/settings.example.env` lists every variable with a comment; the real `.env` is gitignored.
- Secrets: broker login, bridge credentials, Anthropic API key, Telegram token, database passwords, the risk gate's Ed25519 decision signing key (only the risk-gate container mounts it; execution gets the public key). Mounted as Docker secrets in production.
- A logging filter redacts anything that looks like a secret. A test feeds known secret values through every logger and asserts they never appear.
- Every config file is hashed at load time and the hash is written to the audit log with the `ConfigChanged` event.

---

## 18. Deployment

### Services (docker-compose)

| Service | Host | Role |
| --- | --- | --- |
| postgres (TimescaleDB + pgvector) | Linux | All storage |
| redis | Linux | Event bus |
| engine-live | Linux | Strategy runner for shadow, micro, live, scaled versions |
| allocator | Linux | Sizing proposals |
| risk-gate | Linux | Risk decisions |
| execution | Linux | Order manager, reconciliation, watchdog |
| mt5-bridge | Windows VPS | MT5 terminal plus bridge |
| backtest-workers | Linux (can be a separate box) | Validation jobs |
| research | Linux | L1 and L6 agents |
| learning | Linux | L2 to L4 training jobs |
| scheduler | Linux | Cron style jobs |
| monitor | Linux | Alerts, summaries, audit writer |
| api | Linux | FastAPI |
| grafana | Linux | Dashboards |

- Live hosts in the same region as the broker's trade servers (confirm with broker; typically London or New York).
- Private network between the Linux host and the Windows VPS via WireGuard or Tailscale.
- Nightly `pg_dump` plus continuous WAL archiving to object storage. A monthly job restores the latest backup into a scratch database and runs `at audit verify` on it.
- NTP time sync on all hosts; the engine refuses to run live if clock skew exceeds 2 seconds against the broker server time.
- `docs/runbook.md` covers: start and stop, restart after full halt, broker outage, rotating keys, restoring from backup.

---

## 19. Testing

| Type | What it proves |
| --- | --- |
| Unit | Indicators vs reference values, sizing math, DSR math, state machine transitions |
| Property (hypothesis) | Risk gate invariants from section 11; allocator never exceeds caps |
| Future poisoning | No lookahead in strategies and feature pipeline |
| Golden backtest | `demo_ma_cross` on a fixed dataset produces an exact stored result; any change to engine math shows up as a diff |
| Parity | Backtest and shadow runs on the same recorded quote stream produce identical signals |
| Integration | Engine, allocator, risk gate, execution over Redis with SimBroker |
| Chaos | Kill the bridge mid order, drop quotes, duplicate fills, restart services during a halt; system must end in a safe, consistent state |
| E2E paper | Full stack against a broker demo account for 48 hours with `demo_ma_cross` at minimum size |
| Import linter | Package dependency rules from section 4 |
| Secret leak | Section 17 |

CI runs everything except E2E paper on every commit. `make check` = ruff + mypy strict + pytest (unit, property, golden, parity, integration, poisoning, import linter).

---

## 20. Build phases and acceptance criteria

| Phase | Build | Done when |
| --- | --- | --- |
| 0 Foundations | Repo, uv workspace, Makefile, CI, docker-compose with Postgres, Redis, Grafana; `CLAUDE.md`; settings; instruments config; Alembic baseline; confirm MT5 Python on Windows and the data source choice | `make check` green on an empty skeleton; decisions logged |
| 1 Domain and data | Models, indicators with dual implementations, data sources (CSV, MT5 history, historical archive), M1 storage, aggregates, quality checks, spread stats, calendar interface | Synthetic generator and data versioning in place; quality report generated on synthetic data with injected faults (every fault type detected); indicator parity tests pass. Loading 10 years of real M1 for 5 symbols is a separate gate that closes once the data source decision (section 23) is made, and must pass before Phase 3 reports are used for anything real |
| 2 Engine and backtest | Event loop, SimClock, MarketView with lookahead protection, strategy API, sandbox loader, AST checks, SimBroker with full cost model, `demo_ma_cross` | Golden backtest stored; poisoning test passes; performance target met |
| 3 Validation | Walk-forward, holdout lock, Monte Carlo, stability, cross-market, DSR, trial registry, HTML and JSON reports | Validation report for `demo_ma_cross` generated with all sections; DSR tests pass |
| 4 Risk gate | Signed config, all checks, sizing, halt states, Ed25519-signed decisions with sequence and expiry, `at risk sign` and `at risk resume` | All property tests pass; tampered config test passes; forged, replayed and expired decision tests pass; XAUUSD sizing example test passes |
| 5 Execution | Broker adapter interface, MT5 bridge and adapter, order manager, stop confirmation, reconciliation, watchdog, execution quality log | 48 hour paper run on a demo account with zero unreconciled states; chaos tests pass |
| 6 Lifecycle | Registry, state machine, evaluator, ShadowBroker, shadow mode, promotion and demotion with alerts | `demo_ma_cross` reaches shadow and stays capped; simulated demotion scenarios behave as specified |
| 7 Allocator | Shrunk Sharpe weights, correlation clusters, caps, weekly rebalance | Allocator tests pass; integrates with risk gate end to end |
| 8 Monitoring and audit | Hash chained audit log, alerts, daily summary, dashboards, API | `at audit verify` passes; every critical alert fires in tests |
| 9 Self-learning | Trade journal, feature store, L1 agents with sandbox and budgets, knowledge base, L2 re-optimization, L3 meta-labeling with purged CV, L4 regime, champion vs challenger, rollback, learning switches, L7 meta-learning with loop scorecard, L8 cost calibration, weekly report | A full learning cycle runs on historical data: agents produce candidates, validation rejects or passes them, a challenger runs in simulated shadow, swap and rollback are exercised; L7 shifts budget away from an arm that is seeded to always fail; L8 activates a pessimistic cost model automatically, refuses an optimistic one without evidence, and triggers revalidation |
| 10 Intake | Intake template, interview doc, `at strategy new/check/submit` | A sample intake file becomes a candidate version end to end |
| 11 Go live | Deploy to VPS hosts, backups, runbook, restore test | Owner signs `risk.yaml`; system runs in shadow with zero critical alerts for 2 weeks |

Real strategies are added after Phase 11 through Phase 10 tooling or discovered by Phase 9 loops.

---

## 21. Designed for later, not built now

The owner plans to sell this system to trading firms. Keep these doors open without building the features:

- Every table has a `tenant_id` column, default `default`. All queries go through a repository layer that filters by tenant.
- Broker adapters, data sources and calendar sources are plugins registered by name in config.
- Branding, reports and alerts take their names and logos from config.
- No code assumes a single account; the account id is part of every order, position and trade.

---

## 22. Content for CLAUDE.md

```
# Project rules
- Source of truth: TRADING_SYSTEM_SPEC.md. Follow phase order in section 20.
- Never write a real trading strategy. Only strategies/examples/demo_ma_cross exists.
- The risk gate is sacred: nothing reaches the broker without an approved RiskDecision. It can only reduce size.
- No lookahead: strategies see closed bars only. Every strategy and feature passes the future poisoning test.
- Research and learning code must never import risk, execution or allocator, and never edit config/risk.yaml,
  config/validation.yaml or config/promotion.yaml.
- No LLM calls in the live order path.
- No secrets in code or logs. Use settings.
- Every change ships with tests. Run `make check` before calling anything done.
- If the spec is unclear, pick the safer option, add a SPEC-QUESTION comment and an entry in docs/open_questions.md.
- Log design decisions in docs/decisions.md.
- Plain, typed Python 3.12. Pydantic models for all data crossing a boundary.
- End every phase with a retro in docs/retros/. Turn recurring mistakes into automated checks first, rules below second.

# Lessons
(Appended by retros. Each lesson: the rule, why, and the phase it came from.)
```

---

## 23. Open decisions for the owner

- [ ] Broker: which MT5 broker accepts the owner's account from Serbia, with what spreads and commission on the chosen symbols
- [ ] Historical data source and its terms of use
- [ ] Economic calendar source
- [ ] Account currency and starting capital for the micro stage
- [ ] Symbols for the first universe (suggested: EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD)
- [ ] Monthly budget for Claude API usage in research
- [ ] Hosting providers for the Linux host and the Windows VPS
- [ ] Holdout epoch start date and cadence (default: yearly)
- [ ] Weekly API spend cap per research agent and the L7 exploration floor (default 20 percent)
