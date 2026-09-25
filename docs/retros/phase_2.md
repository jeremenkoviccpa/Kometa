# Phase 2 retro (2026-09-24)

What broke
- Process-rule tests caught two real inconsistencies: the import-rule test and the AST checker had different
  allowed-import lists (now one source of truth), and a new SPEC-QUESTION was not logged.
- My own SimBroker test had a wrong commission expectation (fixture was per side, not round turn).

What took longest
- The fill simulator: getting "stop first when ambiguous", "no target on the fill bar", gap fills and
  event-jumping right, with hand-built bars for each rule.

Spec assumptions that were wrong or incomplete
- Strategies had no way to exit except SL/TP; added CloseRequest and ModifyStopRequest (tighten only). Both only
  reduce risk, so they are safe under the risk-gate principle.
- Random signal ids (uuid4) would break determinism; ids are uuid5 of strategy, version, bar time and call order.

Tests that caught real bugs
- Future poisoning test proved it can catch lookahead: a deliberately cheating strategy that reads the next bar
  through a private attribute fails it.

Lessons -> automation
- Golden test prints which fields changed; updating it requires AT_UPDATE_GOLDEN=1 and a decisions.md entry.
