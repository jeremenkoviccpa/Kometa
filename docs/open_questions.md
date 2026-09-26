# Open questions

Items the spec leaves open. The safer option was chosen meanwhile; each has a SPEC-QUESTION in code where relevant.

| # | Question | Interim choice | Where |
|---|---|---|---|
| 1 | Broker (MT5, accepts accounts from Serbia), real spreads, commissions, swaps, metal pip size | Placeholder values in config/instruments.yaml | config/instruments.yaml |
| 2 | Historical data source and terms | Dukascopy public datafeed (owner choice 2026-09-25): M1 bid/ask day files, `at data fetch`; owner to confirm the terms allow commercial use before selling the system | data/dukascopy.py |
| 3 | Economic calendar source | ForexFactory weekly export (their published feed, cached, 12 h freshness); owner to confirm the terms suit commercial use before selling the system | data/calendar.py |
| 4 | Account currency and micro-stage capital | EUR in risk.yaml | config/risk.yaml |
| 5 | Monthly Claude API budget | 0, research disabled | config/research.yaml |
| 6 | Hosting providers | - | - |
| 7 | Gap tolerance for broker micro-gaps | 3 minutes | data/quality.py |
| 8 | Grafana read-only DB role password provisioning | Uses db_password secret in dev | grafana/provisioning |
| 9 | Confirm MT5 Python package with a terminal on a Windows VPS | Not yet verified (needs Windows) | spec section 12 |
| 10 | Holdout epoch start date and cadence | Yearly | spec section 9 |
| 11 | Currency conversion in backtests uses static rates | StaticRates raises on a missing pair (never guesses); move to historical rates from conversion pairs' bars | engine/costs.py |
| 12 | Strategies need exits other than SL/TP (time exits, trailing stops) | Added CloseRequest and ModifyStopRequest (tighten only); both can only reduce risk | core/models.py |
| 13 | Slippage "sample" | Deterministic 0.2 x median spread per hour of week (no random draw) so backtests stay deterministic | engine/costs.py |
| 14 | DSR counts every optimizer evaluation as an independent trial; short windows make trial Sharpes noisy, so SR* is very high | Kept as spec says (strictest). Consider effective N via clustering of correlated trials | validation/dsr.py |
| 15 | Cross-market "passing" is not defined in the spec | PF >= min_stability_profit_factor (1.1) on that symbol alone | validation/runner.py |
| 16 | Money at risk of a position without a stop (e.g. a manual/external trade) | Larger of a 2% adverse move on notional and risk_per_trade_max x equity; never zero | risk/gate.py |
| 17 | Weekend cutoff reopening | No new entries from FRI 18:00 Belgrade until Monday 00:00 Belgrade (Sunday evening open is skipped: safer) | risk/gate.py |
| 18 | Should a daily/weekly/full halt also close external (manual) positions? | No: only system positions are closed; each external position left open raises a critical alert. They still count in the gate's exposure | execution/order_manager.py |
| 19 | MT5 timestamps are broker server wall-clock; which zone does the chosen broker use? | Bridge refuses to start without `AT_BRIDGE_SERVER_TZ` (IANA or "NY+h", e.g. "NY+7"); verify on the VPS with a known tick | mt5_bridge/servertime.py |
| 20 | Broker server time for the 2 s skew check: MT5 has no clock call | Newest tick time over the bridge's symbols; so execution must be started while the market is open (a stale tick refuses startup) | execution/service.py |
| 21 | Some brokers rewrite the order comment (e.g. on SL/TP close), which carries `client_order_id` | Positions are matched by ticket once known; the comment is only used before the first ack. Verify the broker keeps the comment on open positions during the paper run | execution/order_manager.py |
| 22 | Netting accounts | Not supported: startup refuses anything but a hedging account (one position per order) | execution/service.py |
| 23 | Phase 5 acceptance: 48 hour paper run on a broker demo account | Blocked on Q1 (broker) and Q9 (Windows VPS). Stand-in: a 48 hour simulated soak on FakeBroker with injected faults | tests/integration/test_paper_soak.py |
| 24 | The ladder has no transition for a first demotion while in micro (micro -> retired is only for the second) | micro -> shadow: out of money, keeps being observed, positions closed. A second demotion within the retire window retires | lifecycle/registry.py, lifecycle/evaluator.py |
| 25 | Shadow "signal rate" vs the backtest: validation reports keep trades, not raw signals | Compare entries (opened trades) per week in both, so rejected signals on either side do not skew the rate | lifecycle/evaluator.py |
| 26 | Drift interval is two-sided: a version doing much better than its backtest is demoted too | Kept as written (safer: it is not the strategy that was validated) | lifecycle/evaluator.py |
| 27 | How demo_only versions reach shadow without passing validation, and how the 48 h paper run trades them | `promote_candidate` sends demo_only straight to shadow (risks nothing). The only other exits: retire, or the owner-only `demo_only` stage (paper trading at minimum size), which the risk gate trades only when the owner-signed config grants it a limit (config/risk.paper.yaml; config/risk.yaml has none) | lifecycle/registry.py, risk/gate.py |
| 28 | Size of the allocator's "total risk budget" | 3% of equity per trade before stage caps (= open_risk_total_max); with the 30% cap one version gets at most 0.9%, then its stage cap (live 0.5%) | config/allocator.yaml |
| 29 | Pending-order views for money-stage strategies (they need the signal id of each broker order) | Empty for now: live strategies see their positions, not their pending orders; cancel requests still work by signal id | engine/service.py |
| 30 | "Live vs backtest drift" dashboard: the backtest band lives in the lifecycle registry (a ledger), not in Postgres | Until the registry moves to Postgres, the dashboard shows money-stage R next to shadow R per version and the evaluator's demotion reasons; the evaluator still enforces the band itself | scripts/dashboards.py |
| 31 | What the risk gate does when the economic calendar is configured but stale or unreachable | Refuses every new entry with a signed rejection ("economic calendar unavailable or stale") until the calendar is fresh again (12 h); a simulated market runs without a calendar, since its clock is not the real one | risk/service.py |
| 32 | OANDA daily financing and cash reconciliation: financing is booked daily (DAILY_FINANCING) and a closed trade also reports `financing` | Daily financing becomes zero-lot deals per trade and the close's financing is counted too; if the first practice runs show double counting, the cash check halts (fail closed) and the mapping is corrected then | execution/oanda.py |
| 33 | cTrader access tokens last about 30 days; refreshing them (ProtoOARefreshTokenReq) is not automated yet | The owner renews the token in the Open API Playground and updates AT_CTRADER_ACCESS_TOKEN; an expired token stops the connector (PermissionError), never a silent retry | execution/ctrader.py |

| 34 | Where the trade journal and feature store live: the spec names Postgres tables, the demo has no database | JSONL in the state directory (newest line per signal wins); the Postgres tables and migration come with the first loop that queries them (slice 9.3), so the schema is shaped by a real reader. |
| 35 | Lesson retrieval (learning/lessons.py): the spec retrieves the top lessons by pgvector similarity | Until the Postgres store exists, the newest lessons of the same family are returned; similarity retrieval comes with the lessons table (with L1 discovery, which is its reader). |
| 36 | The macro engine of the owner sniper method (strategies/library/smc_sniper/strategy.py): DXY and US yields are not data Kometa has yet | Macro is the scheduled-news filter only (no entries within 30 minutes of a high-impact event). DXY and yields come when their series are fed in (Dukascopy lists a dollar index and US bond futures; to be checked) and strategies can subscribe to context symbols they do not trade. |
