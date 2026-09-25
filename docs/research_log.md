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
