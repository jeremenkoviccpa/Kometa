# Phase 3 retro (2026-09-24)

What broke
- The deflated Sharpe was computed before stability and cross-market trials were recorded, undercounting N
  (6 instead of 17). Caught by a test asserting DSR's N equals every pre-holdout trial.
- A hand-computed constant in a DSR test comment was slightly off; tightened the test to 1e-6.

What took longest
- Test runtime: each validation runs ~17 backtests. Profiling showed the demo strategy's EMA recomputation, not
  the engine. Recursive indicators now loop over plain floats (identical IEEE math, golden result_hash unchanged)
  and the suite runs on all cores with pytest-xdist.

Spec assumptions that were wrong or incomplete
- Cross-market "passing" and the walk-forward objective were undefined; chosen conservatively and logged.
- Counting every optimizer evaluation as an independent trial makes DSR very strict (open question 14).

Tests that caught real bugs
- DSR trial ordering (above).

Lessons -> automation
- When a statistic depends on "everything tried so far", test the count explicitly, not just the value range.
