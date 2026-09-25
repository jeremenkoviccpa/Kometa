# Phase 7 retro (2026-09-25)

What broke
- Allocator: the 30% cap was applied per version and the excess redistributed, which pushed budget back into a
  correlated cluster; two near-identical versions could take 60% between them. The property test found it once
  its invariant was stated correctly (a cluster is one slot, so its sum must stay under one cap). The first
  version of that invariant (`cap x members`) was too weak and would have passed the bug.
- The pipeline test "failed" because the lifecycle retired the version, which was correct: halt closures
  produced trades that re-triggered a drawdown demotion (second within 6 months). The scenario order was wrong,
  not the system.

What took longest
- Making the five services testable together deterministically: `pump` delivers everything available until
  nothing moves, over both the in-memory bus and Redis Streams, so one test proves both transports.

Spec assumptions that were wrong or incomplete
- The allocator's "total risk budget" is not sized in the spec (open question 28); pending-order views for live
  strategies need signal ids from execution (29).
- Live sizing uses the price when the allocator and gate see the signal, not the bar close, so lots differ
  slightly from a naive calculation; the tests assert money at risk, which is what the rules are about.

Tests that caught real bugs
- Allocator property test: the cluster cap.
- Mutation run: "shadow signals get sized" survived because the test had no account and no live stage, so the
  signal was dropped for another reason. Fixed with a control case (the same signal, not shadow, must size).
  Second phase in a row where a missing control case made a check vacuous; the CLAUDE.md rule from Phase 6 was
  right, and `make mutate` is what enforces it.

Lessons -> automation
- 7 new mutations (slot cap, stage cap between rebalances, micro fraction, shadow sizing, stale account,
  silent demoted engine, foreign strategy requests); 28/28 killed.
- Rule: invariants in property tests are written from the spec sentence (e.g. "shares one slot"), not from the
  implementation.
