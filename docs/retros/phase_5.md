# Phase 5 retro (2026-09-25)

What broke
- The random chaos test found a real bug: bridge down -> reconciliation enters RECON_HALT -> restart. The risk gate
  persisted its halt, but the reconciler kept "why I halted" in memory; after the restart it saw a clean state,
  never cleared the halt, and raised no alert. Entries would have stayed blocked with no explanation. Fixed by
  persisting the reconciliation episode in the execution journal, written *before* the halt is entered.
- The MT5 adapter sent JSON bodies without a content type; the bridge answered 422. Caught only by the end-to-end
  test (order manager -> HTTP adapter -> bridge app -> fake terminal), not by either side's unit tests.
- Two chaos scenarios queued faults that were consumed by an earlier call than intended, so they tested something
  else than their name. Fault injection now takes `skip=n` to hit a specific call.

What took longest
- Deciding what an automatic resync may do. Final rule: it only accepts explanations found in the broker's own
  records (our client order id, closing or entry deals, expiry); anything else waits for an owner resync. An
  unreachable broker is different: a failed read is not evidence of a wrong state, so that halt clears itself.

Spec assumptions that were wrong or incomplete
- MT5 has no server-clock call; server time = newest tick time, so execution can only start with the market open.
- MT5 timestamps are broker wall-clock written as UTC; the zone must be configured (never guessed).
- The spec does not say whether halts close external positions, how to treat netting accounts, or what to do when
  a broker rewrites order comments. Safer choices logged (open questions 18 to 22).
- "48 hour paper run" needs a broker and a Windows VPS that do not exist yet. The code and chaos parts of the gate
  pass; the real run stays open (open question 23), with a 48 h simulated soak as the stand-in.

Tests that caught real bugs
- Random fault schedules (hypothesis): the restart / halt bug above.
- Bridge end-to-end test: missing content type.
- Process rule `test_every_spec_question_is_logged`: a SPEC-QUESTION added without an open-questions entry.
- Mutation run: removing deal de-duplication survived the first version of the tests (the order state machine
  hid it, but cash would have been counted twice). The test now also reconciles the balance.

Lessons -> automation
- `make mutate` (tests/mutation/mutations.py): each safety rule is broken once and the named tests must fail;
  runs in CI. `make check` verifies every mutation anchor still matches the code, so the list cannot rot.
- Hypothesis-found failures are pinned with `@example` (the restart schedule is one).
- Schema drift test: tables written from Pydantic models must have exactly the models' columns.
- Rule for CLAUDE.md: state that explains a persisted condition is persisted with it, write-ahead.
