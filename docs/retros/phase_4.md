# Phase 4 retro (2026-09-24)

What broke
- Nothing in the gate logic failed, which was itself a warning sign: the first property test generated 90%
  rejects, so "total open risk after approval" was checked only a handful of times. Added a second,
  approval-biased generator (77% approve/resize) that checks per-trade, per-strategy, total risk, lot caps and
  lot-step alignment on every approval.

What took longest
- Deciding where signature verification lives: execution must verify decisions without importing risk, so
  `core.signing` holds the payload format and the verifier; risk only signs.

Spec assumptions that were wrong or incomplete
- HMAC -> Ed25519 (spec 1.1). `require_stop` is now a Literal[True]: even a validly signed config cannot turn it off.
- Stage risk limits live in risk.yaml (signed, risk-owned) rather than being read from promotion.yaml.
- Risk of stop-less external positions and the Sunday reopen were undefined; conservative choices logged.

Tests that caught real bugs
- None; but measuring the property-test verdict distribution exposed weak coverage.

Lessons -> automation
- Property tests must report their case distribution (hypothesis `event`) and cover the branch they protect.
