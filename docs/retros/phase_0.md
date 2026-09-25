# Phase 0 retro (2026-09-24)

What broke
- Nothing structural. mypy strict tripped on numpy integer indices into polars Series; fixed by converting to
  Python ints (`.tolist()`).
- Ruff import ordering flagged after a scripted edit; `make fmt` fixes, `make check` never auto-fixes.

What took longest
- Deciding the package namespace (`autotrader.<pkg>`) and making mypy, ruff and uv agree on it.

Spec assumptions that were wrong or incomplete
- Phase 1 acceptance depended on an undecided data source; split into a synthetic gate and a real-data gate (spec 1.1).
- HMAC between risk gate and execution lets execution forge approvals; switched to Ed25519 (spec 1.1).
- Docker was not running on the dev machine, so compose is validated (`docker compose config`) but not started.

Tests that caught real bugs
- (none yet in Phase 0)

Rules added
- Process rules are now tests (tests/unit/test_process_rules.py): SPEC-QUESTIONs must be logged, done phases must
  have a retro, research/learning code may not reference fenced config files.
