# Phase 6 retro (2026-09-25)

What broke
- A profit factor with no losing trades is infinite; writing it into the registry ledger crashed the evaluator
  (JSON has no inf). Non-finite metrics are now stored as null.
- The "rolling" Sharpe was computed over every money-stage trade, so a version that had been good for months
  could never be demoted for a recent slide. It is now the last `min_trades` trades.
- Refactoring the request checks out of the backtest could have changed engine math; the golden result hash
  proved it did not.

What took longest
- Making live and backtest runs see the same event sequence: batching bars that close together, ordering them
  like the backtest loop, setting `now` to the bar close (never the wall clock) so signal ids match.

Spec assumptions that were wrong or incomplete
- No transition for a first demotion in micro; "signal rate" is not recorded by validation; the drift interval is
  two-sided (demotes outperformers); how demo_only versions reach shadow. Open questions 24 to 27.

Tests that caught real bugs
- Lifecycle scenario tests: the inf metric and the non-rolling Sharpe.
- Two test-design mistakes caught by reading failures instead of loosening asserts: a random "backtest-like"
  sample that happened to fall outside its own 90% band (1 in 10 does), and an assertion that accepted any
  risk-gate rejection. Both replaced: representative quantile samples, and a live-stage control intent that must
  pass where the shadow one is refused.

Lessons -> automation
- 7 new mutations (demo cap, illegal transitions, demotion rules and actions, weekly promotion cap, bar grace,
  shadow stage in the risk gate); `make mutate` kills 21/21.
- Rule for CLAUDE.md: statistical tests use representative samples, not lucky seeds; and a "refused" assertion
  needs a control case that is accepted.
