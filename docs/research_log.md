# Research log

Every look at a strategy's results on real data, including the ones outside `at validate`, so none of them
is forgotten when judging how much searching has been done. Rules: no rule or parameter changes are made
from these looks (that would be tuning on untracked trials); changes go through a new version and
`at validate`, where every trial is counted and the deflated Sharpe gets harder to pass.

## 2026-09-25 First look at the library candidates, real gold

Data: Dukascopy XAUUSD M1 bid/ask, 2017-01-02 to 2020-05-31 (1,205,699 bars, data_version f8e12c2c…), median
spread $0.26. Default parameters, $50,000, 0.5% risk per trade, full cost model (spread, slippage, commission,
swaps). This window lies in the research period; the holdout year (the last 12 months) was not touched.

| Strategy | Trades | Win | Avg R | Total R | PF | Max DD | Net | Spread cost |
|---|---|---|---|---|---|---|---|---|
| swing_trend_pullback 1.0.0 | 135 | 28.1% | -0.157 | -21.2 | 0.76 | 12.3% | -5,056 | 1,115 |
| candle_sr_reversal 1.0.0 | 15 | 13.3% | -0.580 | -8.7 | 0.32 | 5.5% | -2,015 | 451 |
| scalp_session_breakout 1.0.0 | 549 | 41.9% | -0.115 | -63.4 | 0.79 | 29.3% | -13,985 | 8,441 |

Reading: none shows an edge with textbook settings. The scalper pays about 0.1R per trade in spread, slippage
and commission and is near zero before costs; the S/R reversal trades too rarely to judge (15 trades in 3.4
years); the swing strategy loses on both sides. Next: full `at validate` once the whole history is downloaded
(walk-forward search, deflated Sharpe, Monte Carlo, stability, cross-market, holdout).

## 2026-09-25 candle_sr_reversal 1.1.0: the full classic pattern set

Change (decided from the method, not from the data): the trigger uses the complete reversal set ported from
cm45t3r/candlestick (hammer shape, engulfing, harami, piercing line / dark cloud cover, tweezers, stars,
kickers, hanging man) instead of pin bar, engulfing and star only. Same data and settings as above.

| Strategy | Trades | Win | Avg R | Total R | PF | Max DD | Net | Spread cost |
|---|---|---|---|---|---|---|---|---|
| candle_sr_reversal 1.1.0 | 25 | 12.0% | -0.550 | -13.7 | 0.34 | 7.3% | -3,415 | 655 |

Reading: more setups (25 vs 15), no better. A 12% win rate with a 1.5R minimum target points at stops too close
to the level for gold's noise; that is a question for the walk-forward search (stop_buffer_atr, zone_atr), not
for hand-tuning here.

## 2026-09-26 smc_sniper 1.0.0: first look, real gold 2018

The owner's SMC/ICT top-down "1M sniper entry" method as fixed rules (strategy.py lists each rule). Same data
source as above, calendar year 2018, default parameters, full cost model. A single look at the rules, not a
test of the method.

| Step reached | Count |
|---|---|
| H4 dealing range + agreed D1/H4 bias + discount POI present (M5 bars) | 8,172 of 54,375 |
| M5 reaction at the POI (opens a 90 min M1 hunt window) | 185 |
| M1: sweep with rejection | 2,671 bar checks |
| M1: close through the lower high (CHOCH) | 1,812 |
| ... made by a displacement candle with a fair value gap | 232 |
| ... then BOS, pullback into the gap, pin bar or engulfing | 1 |
| Trades (stop within 1 H1 ATR, nearest liquidity >= 3R) | 1 (-1.07R) |

Reading: the method, read literally, trades about once a year on gold M1: the full seven-step M1 sequence
almost never completes within the window. Nothing was loosened by hand; `disp_atr`, `window_m`, `max_sl_atr`,
`stop_buffer_atr` and `liq_lookback` are the walk-forward search's to try. At this rate it cannot reach the
200 out-of-sample trades validation needs on gold alone.

## 2026-09-26 smc_sniper 1.0.2 vs smc_sniper_active 1.0.0, real gold 2017-2023

Data: Dukascopy XAUUSD M1 bid/ask 2017-01-02 to 2023-12-29 (2,457,010 bars, data_version 3ba9fc25…); 2024 is
kept out as the holdout year. `at backtest` defaults, full cost model. The active variant's four relaxations
were chosen from the method before this run, not from its results. One look each; nothing tuned.

| Strategy | Trades | Win | Avg R | Total R | PF | Max DD | Net | Spread cost |
|---|---|---|---|---|---|---|---|---|
| smc_sniper 1.0.2 (strict) | 5 | 40.0% | +0.747 | +3.7 | 2.10 | 1.7% | +1,968 | 543 |
| smc_sniper_active 1.0.0 | 38 | 21.1% | +0.112 | +4.3 | 1.13 | 8.2% | +1,699 | 4,018 |

Check first: smc_sniper 1.0.2 (switches at their defaults) reproduces 1.0.1's 2018 result exactly (1 trade,
-1.07R), so the switches did not change the strict rules.

Reading: both too rare to judge (validation needs 200 out-of-sample trades; the active variant makes about
5 a year on gold). The active variant is near break-even after costs, which take about 0.3R a trade. More
trades would need more markets (the method is not gold-specific) rather than looser rules.

## 2026-09-26 smc_sniper 2.0.0 (the owner's engine diagram and sniper spec), real gold 2017-2023

Same data as the entry above (2024 held out). Defaults: the owner's score threshold 80 (90 off-session).

| Run | Trades | Win | Avg R | Total R |
|---|---|---|---|---|
| defaults (score >= 80) | 1 | 0% | -1.075 | -1.1 |
| score threshold off (every setup that passed the hard rules) | 14 | 7% | about -0.8 | about -10.9 |

The 14 setups by score: one 85 (VALID, London, -1.08R), eight 70-75 (WATCH), five 60-65. Thirteen stopped
out at about -1.07R; one (70, Asia) made +3.15R. Missing most often: D1 or H1 bias agreement, displacement
(a strong candle of 1.5x the threshold), S/R confluence.

How it got here (funnel probes, one look each on 2018 first): M15 arming was being cancelled by the rally
after a real shift (fixed: the trigger keeps the context of the shift); the M5 retest aimed at a small M5
swing (fixed: the breakout area of the M15 BOS); stops beyond the M15 sweep were 2x the H1 ATR limit with
0.2-0.5R to the nearest liquidity, so the stop moved to the retest's pullback low (the owner's "relevant
swing low"); the spread limit is 30% of the risk (gold's $0.23 spread is 20-40% of those tight stops).

Reading: the method as specified is extremely selective on gold (about 2 setups a year pass the hard rules)
and the tight pullback-low stop is hit in almost every one: price revisits below the retest low before
reaching the target. Not an edge on this evidence, and far too few trades to validate. Nothing was tuned on
these results.

## 2026-09-26 smc_sniper 2.0.0: five variants declared before running (scripts/sniper_variants.py)

Owner's ask: more profitable, more frequent. Variants chosen from the diagnosis above (13 of 14 setups hit
the tight stop), declared before any was run, all recorded as trials (kind research). Real gold 2017-2023.

| Variant | Trades | Win | Avg R | Total R |
|---|---|---|---|---|
| A current | 1 | 0% | -1.075 | -1.1 |
| B stop at the sweep low, stop cap at its maximum (2x H1 ATR) | 0 | - | - | 0 |
| C buffer 1.0 M5 ATR under the pullback low | 0 | - | - | 0 |
| D 8 hours for the retest | 1 | 0% | -1.075 | -1.1 |
| E trade WATCH too (score 70+, off-session 80+) | 5 | 0% | -1.071 | -5.4 |

Reading: no variant helps. A wider stop pushes the nearest liquidity below 1:2, so the RR rule removes the
setup (B, C); letting WATCH setups trade adds four losers in seven years (E). The method as specified does
not produce an edge or a useful trade count on gold, and tuning its parameters cannot change that: the
limit is the number of setups the rules allow, not their settings. The owner's 1-2% a day target was
declined as a goal (1% a day compounds to about 1,100% a year; the signed limits halt at a 2% daily loss).

## 2026-09-26 smc_sniper 2.0.0 on more markets (scripts/sniper_markets.py)

Owner's choice: the same rules on more markets to get enough setups to judge. Dukascopy M1 bid/ask, research
windows only (XAUUSD 2017-2023; EURUSD and GBPUSD 2019-2023; 2024 on is held out). Every run a trial.

| Market | Years | As written: trades | R | Hard rules only: setups | Win | Avg R | Total R |
|---|---|---|---|---|---|---|---|
| XAUUSD | 7.0 | 1 | -1.1 | 14 (2.0/yr) | 7% | -0.77 | -10.8 |
| EURUSD | 5.0 | 1 (A+) | -1.1 | 13 (2.6/yr) | 15% | -0.29 | -3.8 |
| GBPUSD | 5.0 | 2 | -2.1 | 15 (3.0/yr) | 7% | -0.90 | -13.5 |

Reading: consistent on every market. As written, 4 trades in 17 market-years, all losers; of the 42 setups
that pass the hard rules, about 1 in 10 reaches its 1:2 target first. The tight stop under the retest's
pullback low is hit first almost every time. More markets raise the count but not the result: the method
as coded has no edge on this evidence. USDJPY and AUDUSD are still downloading.
