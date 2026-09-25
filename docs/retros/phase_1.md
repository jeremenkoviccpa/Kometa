# Phase 1 retro (2026-09-24)

What broke
- A "vectorized" percentile rank, stochastic and swing detector were Python loops; on 10 years of M1 they would
  have been minutes. Benchmarking on 3.76M bars found it; now vectorized (chunked where memory matters).
- The first spike and stale scans were also per-row Python loops; replaced by candidate masks and run-length edges.

What took longest
- Designing S/R levels so batch and incremental forms share the rules (`_LevelBook`) while still being fed by
  independently computed swings/ATR, so the parity test means something.

Spec assumptions that were wrong or incomplete
- Real data and the Postgres layer were blocked (data source undecided, Docker not running); split into a synthetic
  gate (passed) and a carry-over list in docs/progress.md.

Tests that caught real bugs
- None failed unexpectedly in Phase 1; the fault-injection recall test is the one that would.

Lessons -> automation
- "Vectorized" claims are now backed by a benchmark script recorded in progress.md; perf regressions are caught by
  `make perf` (Phase 2).
