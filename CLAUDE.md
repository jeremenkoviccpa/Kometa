# Project rules
- Source of truth: TRADING_SYSTEM_SPEC.md. Follow phase order in section 20.
- Strategies (owner decision 2026-09-25, replaces "never write a real trading strategy"): trading methods may be
  written into strategies/library/<id>/, only as candidates. Never mark one validated, skip or loosen a gate, tune
  on the holdout, or claim it makes money; validation, shadow and micro decide. demo_ma_cross stays demo_only.
- The risk gate is sacred: nothing reaches the broker without an approved RiskDecision. It can only reduce size.
- No lookahead: strategies see closed bars only. Every strategy and feature passes the future poisoning test.
- Research and learning code must never import risk, execution or allocator, and never edit config/risk.yaml,
  config/validation.yaml or config/promotion.yaml.
- No LLM calls in the live order path. One owner exception (2026-09-26): the Claude tracks in packages/ai, paper
  only (refused under AT_ENV=live), demo_only (never promotable), answers checked in code, every trade through
  the risk gate. No other LLM may reach SIGNALS.
- No secrets in code or logs. Use settings.
- Every change ships with tests. Run `make check` before calling anything done.
- If the spec is unclear, pick the safer option, add a SPEC-QUESTION comment and an entry in docs/open_questions.md.
- Log design decisions in docs/decisions.md.
- Plain, typed Python 3.12. Pydantic models for all data crossing a boundary.
- End every phase with a retro in docs/retros/. Turn recurring mistakes into automated checks first, rules below second.

# Working notes
- Packages are namespace packages: `packages/<name>/src/autotrader/<name>`, imported as `autotrader.<name>`.
- Allowed internal imports are defined in tests/unit/test_import_rules.py (ALLOWED). Widening it is a decision to log.
- Current phase and what is left: docs/progress.md. Update it when a phase item lands.
- Synthetic data (`autotrader.data.synthetic`) is for tests and plumbing only; its data_version starts with `synthetic-`.

# Lessons
(Appended by retros. Each lesson: the rule, why, and the phase it came from.)
- Indicator outputs are aligned to the bar where they become knowable (e.g. swings at the confirmation bar), never
  to the bar they describe. Why: aligning to the described bar is silent lookahead. (Phase 1)
- Never guess a timezone for timestamps from files; require tz-aware data or an explicit `assume_tz`. (Phase 1)
- YAML floats go through `str` before `Decimal`, or binary float error leaks into contract specs. (Phase 1)
- "Vectorized" means no per-row Python loop; verify with a 10-year M1 benchmark before claiming it. (Phase 1)
- Deterministic ids only (uuid5 of stable inputs); never uuid4/time in anything that feeds a result hash. (Phase 2)
- A statistic over "everything tried" must be computed after the last trial is recorded; test the count. (Phase 3)
- Speed up hot loops without changing math: iterate over `.tolist()` floats; prove it with an unchanged golden
  result_hash. (Phase 3)
- Property tests: tag outcomes with hypothesis `event()` and check the distribution; an invariant about approvals
  needs a generator that mostly approves. (Phase 4)
- State that explains a persisted condition (why we halted, what we sent) is persisted too, and written before the
  condition is entered. Why: a restart forgot why reconciliation halted and left RECON_HALT forever. (Phase 5)
- Every safety rule has a mutation in tests/mutation/mutations.py and `make mutate` kills all of them. Why: tests
  that pass on the first run can miss a rule (deal de-duplication survived). (Phase 5)
- Pin every hypothesis-found failure with `@example` before fixing it. (Phase 5)
- Fault injection must target the exact call (`skip=n`); check the scenario reaches the fault it names. (Phase 5)
- Tests of statistical bands use representative samples (quantiles of the reference), never a lucky seed; and a
  "refused" assertion needs a control case that is accepted, or it proves nothing. (Phase 6)
- Metrics written to ledgers must be finite (inf/NaN become null); JSON cannot hold them. (Phase 6)
- Write property-test invariants from the spec's sentence ("a cluster shares one slot"), not from what the code
  does; an invariant derived from the implementation passes its bugs. (Phase 7)
- Tools that edit source in place (mutation runs) lock and leave a restore marker; `make check` fails while one
  exists. Why: an interrupted run left the live-bar grace rule mutated and nothing noticed. (Phase 8)
- Infrastructure config (roles, datasources, compose) is proven by running it once; a comment saying something
  exists is not evidence. Why: Grafana pointed at a role no migration created. (Phase 8)
- Test queries (dashboards, reports) on rows written by the real writers and require a fully non-null row; a
  wrong JSON path returns null, not an error. (Phase 8)
