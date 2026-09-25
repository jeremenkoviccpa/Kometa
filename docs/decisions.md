# Design decisions not covered by the spec

Newest last. Format: date, decision, why, alternatives.

## 2026-09-24 Namespace packages under `autotrader.*`
Each workspace member installs `autotrader.<name>` (e.g. `autotrader.core.indicators`) instead of a top-level
`core`. Why: top-level names like `core` and `data` collide with other distributions. The spec's `core.indicators`
means `autotrader.core.indicators`.

## 2026-09-24 Import rules enforced by a custom AST test
`tests/unit/test_import_rules.py` instead of the `import-linter` package: no extra dependency, and it also checks that
pyproject internal dependencies never widen the allowed graph. The CLI is the only non-risk package allowed to
import `risk` (owner-side `at risk sign`).

## 2026-09-24 Indicator parity tolerance 1e-9
Vectorized and incremental forms agree within 1e-9 (relative and absolute), not bitwise: numpy's pairwise summation
differs from sequential sums in the last bits. Signal determinism does not depend on this because backtest and live
both run strategies on the same MarketView windows with the same functions.

## 2026-09-24 Indicator conventions
EMA seeded with the SMA of the first n values; ATR and RSI use Wilder smoothing seeded the same way; rolling std is
population (ddof=0); stochastic %K is 50 on a zero range bar; RSI is 50 when there were no moves at all.
Swings: strictly above the left bars and >= the right bars (one swing per flat top), reported at the confirmation bar.

## 2026-09-24 Higher timeframe alignment
Buckets align to a trading day closing at 17:00 New York (configurable), computed by shifting local time so the
close is midnight. Incomplete trailing buckets are dropped. On DST switch days an intraday bucket spanning 02:00 NY
can be one hour long or short; accepted.

## 2026-09-24 Data quality thresholds
Gaps: more than 3 open-market minutes missing is flagged; severity low under 15, medium under 60, high otherwise.
Spikes: excursion from previous close above 8 x ATR(14) that returns within 2 x ATR in 3 bars. Stale: 10 or more
flat bars equal to the previous close. Duplicates and ask-below-bid are high; zero spread medium.

## 2026-09-24 Risk decisions signed with Ed25519, not HMAC
See spec 1.1 changelog. Adds `sequence`, `expires_at`, `signature` to RiskDecision.

## 2026-09-24 Synthetic data generator
Regime-switching random walk with Student-t(4) shocks, session volatility, hour-of-week spreads, weekend closure.
Fault injection returns ground truth so quality checks are tested for recall, not just "runs".

## 2026-09-24 Golden update: data_version hashing scheme
data_version now hashes raw little-endian column bytes per month instead of CSV text (about 10x faster).
Golden `demo_ma_cross.json` changed ONLY in `data_version`; `result_hash`, trade count and total R were identical,
which also confirmed that moving EMA/Wilder/RSI loops to plain Python floats did not change any engine math.

## 2026-09-24 Engine design (Phase 2)
- Event-jump simulation: per symbol, the broker finds the first M1 bar that triggers any open order or position
  with numpy and jumps there. 10 years of M1 with H1 signals runs in about 35 s including data prep.
- Fill prices are built from BID bars plus a modelled spread (hour-of-week medians, 3x within 2 minutes of
  high-impact news and rollover) so the same price history can be re-costed with broker or L8-calibrated spreads.
- Slippage is deterministic: 0.2 x median spread of that hour, always adverse (no random draw) to keep runs
  reproducible.
- Signal ids are uuid5(strategy, version, bar time, call order): identical across runs and modes.
- Strategies get copies of bars only; the AST checker forbids `.base`, dunder access, wall clock, `np.random`.
- `BacktestGuard` does sizing and basic stop checks for backtests only and refuses any other mode; the real risk
  gate plugs in behind the same OrderGate protocol in Phase 4.
- Hedging account model: every signal opens its own position; commission is round-turn/2 per side.
- Rollovers (swaps and daily equity marks) are Mon-Fri 17:00 New York; the Wednesday rollover is triple.

## 2026-09-24 Validation design (Phase 3)
- Sharpe and DSR use daily R returns: risk_fraction x R of each trade on the day it closes, zero days included.
  Comparable across windows and independent of sizing noise.
- Walk-forward search: seeded random search within param bounds (budget in validation.yaml), objective is the
  t-statistic of mean R with a minimum trade count. Every evaluation is a trial.
- DSR is computed after all pre-holdout trials of the run are recorded (a test caught it being computed too early).
- Stability runs on the full research range; cross-market passing = PF >= min_stability_profit_factor.
- The holdout is opened only if every other check passed, and the attempt is recorded before evaluation.
- The file-backed JsonlLedger (hash-chained) stands in for Postgres until the DB is verified.

## 2026-09-25 Execution design (Phase 5)
- `client_order_id` = "at" + first 29 hex chars of the intent id (31 chars, the MT5 comment limit). One intent can
  never become two orders, even if the gate decides it twice.
- Write-ahead journal (`var/execution_journal.json`, hash-checked, atomic replace): the order is recorded as
  `sending` before it is sent. Before every send the broker is searched for the id; one retry at most. If the
  broker never answers, the order stays `sending` and reconciliation settles it (adopt if found, `failed` if the
  broker has neither the order nor a deal for it). A corrupt journal refuses startup (fail closed).
- The last verified decision sequence is persisted in the journal, so a restart cannot reopen a replay window.
- Execution refuses a decision whose approved lots exceed the intent's proposed lots or are off the lot step,
  even with a valid signature (defense in depth behind the gate's own invariant).
- Stops are rounded toward the entry (buy: ceiling, sell: floor), so rounding only shrinks risk.
- Closing positions and tightening stops need no risk decision: they only reduce risk. Loosening is refused.
- Market entries need a quote younger than `max_quote_age_seconds` (10 s); a dropped feed blocks entries.
- Reconciliation: a mismatch enters RECON_HALT, then one resync that only accepts explanations from the broker's
  own records (orders we sent, closing deals, entry deals, expiry). If anything is left, the halt stays until an
  owner resync, which adopts the broker's state. An unreachable broker also halts but clears by itself on the
  next clean check, because a failed read is not evidence of a wrong state. Loss halts are never cleared here.
- Cash reconciliation: every new deal (trades, swaps, commissions, balance operations) moves an expected balance;
  the broker balance must match it within 0.01. Deposits show up as balance deals (warning alert), so only
  unexplained balance changes halt.
- `HaltCommand` (core.models) is the risk gate -> execution message; `Alert`/`AlertSink` live in core so any
  service can raise alerts before the monitor service exists (Phase 8).
- Startup: live needs a real account, every other env a demo account; hedging only; account id must match;
  clock skew against the newest broker tick at most 2 s.
- Bridge: FastAPI, bearer token (>= 32 chars, constant-time compare), bound to the tunnel address; terminal
  failures are 503 so the adapter treats them as "may or may not have happened", never as a rejection.
- In backtests the pieces still run in process; the Redis Streams transport between engine, allocator, risk gate
  and execution is built with the allocator integration (Phase 7), where spec section 19 puts that test.

## 2026-09-25 Lifecycle and shadow mode (Phase 6)
- Live bars: `core.timeframes.bucket_open_ns` is the one bucket alignment for historical aggregation and the live
  bar builder; a test compares both around the 2020 US DST switches. Bars are emitted 1 s after they close
  (grace) and in backtest order, so live and backtest runs see identical sequences.
- The request checks (ids, created_at, symbols, duplicates) moved to `engine.requests` and are shared by the
  backtest loop and the live runner. The golden backtest hash did not change.
- ShadowBroker fills from quotes with the SimBroker's rules; nominal size only scales costs, R is what the
  evaluator reads. Shadow trades carry account_id "shadow" and are never counted as money-stage trades.
- The registry is an append-only hash-chained ledger (`core.ledger`, moved from validation) replayed on load.
  Illegal transitions raise and alert critical. Non-finite metrics (e.g. profit factor with no losses) are stored
  as null.
- Candidates without a BacktestProfile cannot enter shadow (nothing to compare against). Validation failures and
  synthetic-only passes retire the candidate.
- Bands are bootstrap intervals of the backtest sample at the live sample size (seeded, 4000 draws). The rolling
  Sharpe is annualized daily R over the last `live.exit.min_trades` money trades.
- Demotions: stage change first (it stands even if execution fails, with a critical alert), then cancel the
  version's pending entries; close its positions only when it leaves the money stages.

## 2026-09-25 Allocator and service bus (Phase 7)
- The 30% cap applies to a correlation cluster's slot, not to each member: capping members and redistributing let
  two near-identical versions take 60% (a property test found it). Slots are water-filled under the cap and the
  remainder stays unallocated.
- Allocations change only at the weekly rebalance; a stage cap applies immediately and can only lower a fraction.
  Micro versions always get the micro fraction. One intent per signal: intent id = uuid5(signal id).
- Bus: one consumer group per service per stream, ack after the handler; a failing handler is logged, reported and
  acked (no poison-message stalls); a crash before ack means redelivery, which every handler tolerates (order
  manager dedupes by client order id; decisions carry sequence numbers). `pump` gives deterministic delivery for
  tests and the in-process pipeline; `run` is the production loop.
- Stages reach the risk gate, allocator and engine as a StageSnapshot (start, hourly) plus StageChanged events.
  An unknown version does not trade and does not publish.
- Execution publishes the account (with each position mapped to its strategy version) every cycle; the risk gate
  answers intents without fresh account data (10 s) with a signed rejection. On start the risk gate re-publishes a
  persisted halt so execution finishes acting on it.
- Closed live trades are built in execution from broker deals (R on the initial stop and the opened lots,
  commissions and swaps included) and go to the lifecycle; MAE/MFE are not tracked on the live path yet.

## 2026-09-25 Monitoring and audit (Phase 8)
- Audit coverage: execution publishes every order change on a new `orders` stream. The order manager diffs
  tracked orders (state, stop, lots, broker id) each time it writes the journal and reports OrderPlaced,
  OrderModified or OrderCancelled; fills come from the existing fill callback as OrderFilled. Reporting happens
  after the journal write, so an event never describes state that could be lost, and a restart does not report
  old changes again.
- Config hashes (spec 17): every loader of a `config/` file reads it through `core.configs.read_config`, which
  records the sha256; the composition root publishes them as ConfigChanged on a `config` stream and the audit
  records them with the reading service as actor. A process-rule test fails on any YAML loader that skips it.
- Postgres audit log: `monitor.pg_ledger.PgLedger`, same chain as the JSON Lines ledger. Rows keep the exact
  canonical text that was hashed (migration 0005), because jsonb normalizes numbers and key order; `verify`
  checks the chain over that text and that event_type/actor/at/payload still agree with it. `payload` holds the
  event data (the actor has its own column). Appends take a transaction-scoped advisory lock so several writers
  keep one chain. Synchronous psycopg 3: the Ledger interface is synchronous and this never runs in the order
  path. `at audit verify --db` walks it.
- Dashboards as code: `scripts/dashboards.py` generates the seven spec dashboards; a unit test fails on drift
  between it and the committed JSON, and a Postgres test runs every panel as Grafana's role on data written by
  the real audit service and monitor and requires a fully non-null row (a wrong JSON path reads null). Equity is
  not audited (too frequent), so the monitor writes one `account_snapshots` row per account per minute (0006).
- Grafana logs in as `autotrader_readonly` (SELECT only, including future tables) with its own secret; the role
  is created without a login and `make db-roles` gives it one. It previously pointed at a role that did not exist
  and at the owner's password.
- Warnings: slippage above the model compares the mean adverse slippage of a symbol's last 20 fills with the
  cost model's `0.2 x spread` (spread at fill standing in for the median), one warning per breach; data quality
  warns per traded symbol for medium/high issues only. Rollback and API budget warnings arrive with Phase 9.
- `at demo run` composes every service in one process, so the CLI may import allocator and api (it already
  imported risk and execution for owner-side commands). Nothing else changes in the import graph.
- The `demo_only` stage: owner-only, only for demo_only versions, trades only if the owner-signed risk config
  grants it a limit, and then at the instrument's minimum size. config/risk.paper.yaml grants 0.1%;
  config/risk.yaml grants nothing, so a production config can never trade it.
- Compose host ports are variables (AT_PG_PORT, AT_REDIS_PORT, AT_GRAFANA_PORT) so the stack runs next to other
  projects on one machine.
- `make mutate` edits source in place, so it now holds an exclusive lock and writes a restore file before each
  edit; the next run restores an interrupted edit and `make check` fails while the file exists. An interrupted
  run had left the live-bar grace rule mutated in the source.
- Kometa Trading Hub (the API's `/` page): live view built only on the monitor's state. The monitor keeps a
  feed of every decision-relevant bus message (`monitor.feed`, 600 items, one readable line each) and a week of
  minute prices per symbol from the account updates; `/api/feed?after=` and `/api/market` serve them. Owner
  controls on the page (ack, pause learning, retire) use the existing token-checked, audited endpoints; the token
  stays in the browser tab's session storage. The page cannot change risk limits because no endpoint can.
- Economic calendar (open question 3): ForexFactory's weekly export (nfs.faireconomy.media), the JSON feed
  ForexFactory publishes for tools; the website is never scraped. Fetched at most every 30 minutes, hourly by the
  demo, cached raw on disk, stale after 12 h. It feeds check 8 (news blackout) through the risk service on a real
  clock only; a stale configured calendar blocks new entries (open question 31). Until now the service never
  passed events to the gate, so the blackout was off on the bus path.
- The loss formulas moved into `RiskGate.losses()`, used by both the halt checks and the read-only `usage()`
  view the hub shows, so the display cannot disagree with the gate. Weekly halt got its first test; the three
  loss halts and the calendar rule got mutations.
- Demo realism: `at demo run` trades XAUUSD by default with the real contract spec (100 oz, 0.01 lot steps,
  commission from config), gold-like synthetic prices (16% yearly volatility, ~$0.25 spread, start 4300), seeded
  adverse slippage and 35-180 ms latency in the simulated broker (both off by default for tests), and quotes that
  walk each bar low-then-high or high-then-low by its direction. Equity defaults to 50,000: at the paper stage's
  0.1%, 10,000 cannot cover a minimum-lot gold stop and the gate (correctly) rejects every trade. The strategy's
  manifest still lists SYNTH; the demo runs the same code and version on the chosen symbol.
- The hub's chart is TradingView Lightweight Charts (Apache 2.0, from jsDelivr, SVG fallback offline) over M1
  candles the monitor builds from bus quotes and aggregates with `core.timeframes` (the strategies' alignment).
  The real market is TradingView's embedded widget (OANDA:XAUUSD), display only: its data cannot be read by the
  system and never reaches trading.

## 2026-09-25 Strategies and real data (owner decisions)
- The owner lifted spec section 0 rule 3 ("do not invent trading strategies") for the strategy library: methods
  may be written into strategies/library/, entering only as candidates through the normal ladder. Written from
  general price-action knowledge (trend pullbacks, pin bars and engulfing bars at levels, session breakouts),
  not copied from any book; the owner's "Candlestick Trading Bible" link was an unauthorised copy and was not
  downloaded.
- Historical data: Dukascopy's public datafeed (open question 2), daily M1 bid/ask candle files, cached raw and
  resumable, built into data/dukascopy/<SYMBOL>_M1.parquet. Closed-market minutes (flat, zero volume) dropped.

- Candlestick patterns: the classic set (29 patterns) is a port of github.com/cm45t3r/candlestick v3.0.0 (MIT,
  notice in THIRD_PARTY_NOTICES.md), vectorized plus incremental like every indicator. The original reports a
  multi-candle pattern at its first candle; the port reports it at the completing bar (no lookahead). Proven
  faithful against the original library's own output on 4,000 candles (tests/golden/candlestick_oracle.json,
  regenerated with scripts/candlestick_oracle.py and Node). The hub marks them on the chart (toggle).
- Choosing strategies in the demo: `at demo run` loads demo_ma_cross and the whole strategies/library into the
  engine and registers each in the DEMO's own registry as a paper trial (demo_only, so none can be promoted from
  there). The owner switches each between shadow (off) and the demo_only paper stage (on) from the hub
  (`/api/control/paper-trade`, owner token, audited); switching off cancels its entries and closes its
  positions. Default: the library trades, the plumbing test is parked (`--trade` overrides). The simulated
  market is played 260 days before the visible start, cut at a 17:00 New York day boundary, only to warm the
  strategies' charts (no partial bar where history meets live bars).
- Demo broker (owner choice 2026-09-25): OANDA practice account through the v20 REST API (`--broker oanda`),
  because it needs neither MT5 nor Windows. The adapter maps OANDA onto Kometa's MT5-shaped broker model: units =
  lots x contract size, our client order id in clientExtensions of both order and trade, one OANDA trade per
  position (hedging must be on, else startup refuses the "netting" account), fills/closes/financing/transfers as
  deals. The practice host reports a demo account, so startup refuses real money in paper. History for warm-up and
  tuning comes from the same API (`at data fetch --source oanda`, cached per month).
- Hosting: Vercel is unsuitable (short-lived functions; Kometa is a set of long-running services). Paper trading
  runs on the owner's Mac or Windows machine (WSL2); live money moves to a small Linux VPS (Phase 11).

