# Phase 8 retro (2026-09-25)

What broke
- The previous session ended with the tree red: an interrupted or overlapping `make mutate` left the live-bar
  grace rule mutated in the source (`cutoff = now_ns`). Nothing flagged it as a mutation; the demo and the
  pipeline test failed with symptoms two steps away ("bars must be appended in close_time order", a weekly
  instead of daily halt). A safety rule was silently switched off in the working tree.
- Grafana could never have connected: its datasource logged in as `autotrader_readonly`, "created in migrations",
  a role that did not exist, with the owner's password. The comment was the only evidence and it was false.
- Two mutation anchors went stale when the `demo_only` stage was added, and that stage was never logged as a
  decision.
- The Telegram "/ack from a stranger is ignored" test was vacuous: the stranger's and the owner's /ack arrived in
  the same poll, so the owner's ack hid the bug. `make mutate` found it (third phase running with a missing
  control case).

What took longest
- Deciding what "every order, fill, cancel is audited" means without touching every call site: diffing tracked
  orders at the journal write gave one hook that cannot be forgotten by new code paths.
- Making the dashboards testable: generating them from code and running every panel as Grafana's role on rows
  written by the real audit service and monitor.

Spec assumptions that were wrong or incomplete
- The spec's dashboards assume their data is in Postgres; equity is not audited (too frequent), so it needed
  `account_snapshots`, and the backtest band for "drift" still lives in the lifecycle ledger (open question 30).
- "Every config file is hashed at load time" had no mechanism; now `core.configs.read_config` plus a process rule.
- jsonb is not a faithful store for hashed JSON (it normalizes numbers and key order), so the chain is kept over
  a stored canonical text.

Tests that caught real bugs
- `test_mutation_anchors_still_exist`: showed the three changed sites, one of which was the leaked mutation.
- `make mutate`: the vacuous stranger-/ack test.
- The dashboard check, once tightened to "some row fully non-null", catches a wrong JSON path (verified by
  breaking one on purpose); the first version, "not all values null", would have missed it.

Lessons -> automation
- `scripts/mutate.py` holds an exclusive lock and writes `var/mutate-restore.json` before each edit; the next run
  restores it and `make check` fails while it exists (`test_no_mutation_run_left_the_tree_mutated`).
- `test_config_loaders_hash_what_they_read`: any YAML loader that bypasses `read_config` fails the gate.
- `test_committed_json_is_what_the_generator_writes` and the Postgres panel test: dashboards cannot drift or
  query a path that reads null.
- CI now runs a TimescaleDB service, and the Postgres tests fail instead of skipping when CI lacks the URL.
- 7 new mutations (order changes reported, fills published, no re-report after restart, slippage warning,
  quality warning scope, plus the re-anchored two).
- Rules: infrastructure is proven by running it once (a comment saying a role exists is not evidence); queries
  are tested on rows written by the real writers, requiring a fully non-null row.
