"""Mutation suite for the safety-critical code (Phase 5 retro).

Each entry breaks one safety rule. `make mutate` applies them one at a time and requires the listed
tests to FAIL; a mutation that survives means the tests do not protect that rule.
tests/unit/test_process_rules.py checks in every `make check` that each `old` text still exists exactly
once, so this list cannot silently rot when the code changes.
"""

from __future__ import annotations

from typing import NamedTuple

OM = "packages/execution/src/autotrader/execution/order_manager.py"
RC = "packages/execution/src/autotrader/execution/reconcile.py"
GATE = "packages/risk/src/autotrader/risk/gate.py"
SIGN = "packages/core/src/autotrader/core/signing.py"
UNIT_EXEC = "tests/unit/test_execution.py"
CHAOS = "tests/integration/test_execution_chaos.py"
RISK = "tests/unit/test_risk_gate.py"
REG = "packages/lifecycle/src/autotrader/lifecycle/registry.py"
EVAL = "packages/lifecycle/src/autotrader/lifecycle/evaluator.py"
BARS = "packages/engine/src/autotrader/engine/live_bars.py"
LIFE = "tests/unit/test_lifecycle.py"
SHADOW_LIFE = "tests/integration/test_shadow_lifecycle.py"
LIVE_BARS = "tests/unit/test_live_bars.py"
ALLOC = "packages/allocator/src/autotrader/allocator/allocator.py"
WEIGHTS = "packages/allocator/src/autotrader/allocator/weights.py"
ALLOC_SVC = "packages/allocator/src/autotrader/allocator/service.py"
RISK_SVC = "packages/risk/src/autotrader/risk/service.py"
ENGINE_SVC = "packages/engine/src/autotrader/engine/service.py"
EXEC_BUS = "packages/execution/src/autotrader/execution/bus_service.py"
ALLOC_T = "tests/unit/test_allocator.py"
BUS_T = "tests/unit/test_bus.py"
API = "packages/api/src/autotrader/api/app.py"
ALERTS = "packages/monitor/src/autotrader/monitor/alerts.py"
AUDIT = "packages/monitor/src/autotrader/monitor/audit.py"
API_T = "tests/unit/test_api.py"
MON_T = "tests/unit/test_monitor.py"
MON_STATE = "packages/monitor/src/autotrader/monitor/state.py"
RISK_SVC_T = "tests/unit/test_bus.py"
OANDA = "packages/execution/src/autotrader/execution/oanda.py"
OANDA_T = "tests/unit/test_oanda.py"
CTRADER = "packages/execution/src/autotrader/execution/ctrader.py"
CTRADER_T = "tests/unit/test_ctrader.py"
QUALITY = "packages/data/src/autotrader/data/quality.py"
DATA_T = "tests/unit/test_data.py"


class Mutation(NamedTuple):
    name: str
    file: str
    old: str
    new: str
    tests: tuple[str, ...]


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "execution: no broker lookup before send",
        OM,
        "if pos is not None or order_id is not None:",
        "if False:",
        (UNIT_EXEC, CHAOS),
    ),
    Mutation(
        "execution: accepts approved > proposed",
        OM,
        "ZERO < lots <= intent.proposed_lots",
        "ZERO < lots",
        (UNIT_EXEC,),
    ),
    Mutation(
        "execution: ignores stale quotes",
        OM,
        "            if q is None:\n",
        "            if q is None and False:\n",
        (UNIT_EXEC,),
    ),
    Mutation(
        "execution: deals applied twice",
        OM,
        "if d.deal_id in self.state.seen_deals:",
        "if False:",
        (UNIT_EXEC,),
    ),
    Mutation(
        "execution: stop loosening allowed",
        OM,
        'if not (sl > t.sl if t.side == "buy" else sl < t.sl):',
        "if False:",
        (UNIT_EXEC,),
    ),
    Mutation(
        "execution: decision sequence not restored",
        OM,
        "self.verifier.last_sequence = max(self.verifier.last_sequence, self.state.last_decision_sequence)",
        "pass",
        (UNIT_EXEC,),
    ),
    Mutation(
        "execution: halt does not close positions",
        OM,
        "        if cmd.close_positions:",
        "        if False:",
        (UNIT_EXEC, CHAOS),
    ),
    Mutation(
        "reconcile: stuck halt clears itself",
        RC,
        "            return report  # already halted and alerted; waits for the owner",
        "            self.risk.clear_recon_halt()\n            return report",
        (UNIT_EXEC,),
    ),
    Mutation(
        "reconcile: external positions not flagged",
        RC,
        "            if not is_system_comment(p.comment):\n                external.append",
        "            if False:\n                external.append",
        (UNIT_EXEC,),
    ),
    Mutation(
        "reconcile: episode forgotten on restart",
        RC,
        "        return self.om.state.recon_episode",
        '        return "clean"',
        (CHAOS,),
    ),
    Mutation(
        "risk: can increase size",
        GATE,
        "lots = ZERO if w.rejected else min(w.lots, intent.proposed_lots)  # can only reduce",
        "lots = ZERO if w.rejected else w.lots + intent.proposed_lots",
        (RISK,),
    ),
    Mutation(
        "risk: trades while halted",
        GATE,
        "        if self.state.halt != HaltState.NORMAL:",
        "        if False:",
        (RISK,),
    ),
    Mutation(
        "signing: expired decisions accepted",
        SIGN,
        "if ensure_utc(now) > d.expires_at:",
        "if False:",
        (RISK, UNIT_EXEC),
    ),
    Mutation(
        "signing: replays accepted",
        SIGN,
        "if d.sequence <= self.last_sequence:",
        "if False:",
        (RISK, UNIT_EXEC),
    ),
    Mutation(
        "lifecycle: any transition allowed",
        REG,
        "legal = (frm, to) in ALLOWED or",
        "legal = True or",
        (LIFE,),
    ),
    Mutation(
        "lifecycle: demo_only may leave shadow",
        REG,
        "if v.info.demo_only and frm == Stage.SHADOW and to not in (Stage.RETIRED, Stage.DEMO_ONLY):",
        "if False:",
        (LIFE,),
    ),
    Mutation(
        "lifecycle: demotion rules ignored",
        EVAL,
        "        if reasons:\n",
        "        if False:\n",
        (LIFE,),
    ),
    Mutation(
        "lifecycle: one-stage demotion closes positions",
        EVAL,
        "if to in (Stage.SHADOW, Stage.RETIRED):",
        "if True:",
        (LIFE,),
    ),
    Mutation(
        "lifecycle: weekly promotion limit ignored",
        EVAL,
        ">= self.cfg.global_.max_promotions_to_live_per_week:",
        ">= 10**9:",
        (LIFE,),
    ),
    Mutation(
        "live bars: no grace for late quotes",
        BARS,
        "cutoff = now_ns - self.grace_ns",
        "cutoff = now_ns",
        (LIVE_BARS,),
    ),
    Mutation(
        "risk: shadow versions may trade",
        GATE,
        "if stage not in trading:",
        "if stage is None:",
        (SHADOW_LIFE,),
    ),
    Mutation(
        "allocator: cap per version instead of per cluster slot",
        WEIGHTS,
        "slot_share = capped_shares(slot_w, cap)",
        "slot_share = capped_shares(slot_w, 1.0)",
        (ALLOC_T,),
    ),
    Mutation(
        "allocator: stage cap waits for the weekly rebalance",
        ALLOC,
        "        return min(rf, self.cfg.stage_limit(stage))",
        "        return rf",
        (ALLOC_T,),
    ),
    Mutation(
        "allocator: micro sized by weight",
        ALLOC,
        "        if stage in (Stage.MICRO, Stage.DEMO_ONLY):\n"
        "            return self.cfg.stage_limit(Stage.MICRO)\n",
        "",
        (ALLOC_T,),
    ),
    Mutation(
        "allocator: shadow signals sized",
        ALLOC_SVC,
        "if not isinstance(msg, SignalEmitted) or msg.shadow:",
        "if not isinstance(msg, SignalEmitted):",
        (BUS_T,),
    ),
    Mutation(
        "risk service: stale account accepted",
        RISK_SVC,
        "if acct is None or now - acct.at > self.max_account_age:",
        "if acct is None:",
        (BUS_T,),
    ),
    Mutation(
        "engine: demoted version keeps publishing",
        ENGINE_SVC,
        "if self.stages.get(key) not in MONEY:",
        "if False:",
        (BUS_T,),
    ),
    Mutation(
        "execution: strategy may close a foreign position",
        EXEC_BUS,
        "if t is None or t.strategy_id != req.strategy_id or t.strategy_version != req.strategy_version:",
        "if t is None:",
        (UNIT_EXEC,),
    ),
    Mutation(
        "api: control endpoints without the owner token",
        API,
        "        if authorization is None or not hmac.compare_digest(authorization.encode(), expected):",
        "        if False:",
        (API_T,),
    ),
    Mutation(
        "alerts: criticals are not repeated",
        ALERTS,
        "due = st.last_sent is None or now - st.last_sent >= REPEAT",
        "due = st.last_sent is None",
        (MON_T,),
    ),
    Mutation(
        "alerts: a stranger can acknowledge",
        ALERTS,
        '            if str((msg.get("chat") or {}).get("id")) != str(self.chat_id):\n'
        "                continue",
        "            pass",
        (MON_T,),
    ),
    Mutation(
        "audit: chain break not alerted",
        AUDIT,
        "        alerts.send("
        'Alert(severity=Severity.CRITICAL, kind="audit_chain_break", message=str(e), at=now))',
        "        pass",
        (MON_T,),
    ),
    Mutation(
        "risk: demo_only trades without a signed stage limit",
        GATE,
        "trading = TRADING_STAGES | ({Stage.DEMO_ONLY} if stage_limit > 0 else set())",
        "trading = TRADING_STAGES | {Stage.DEMO_ONLY}",
        (RISK,),
    ),
    Mutation(
        "audit: order changes not reported",
        OM,
        "        self.journal.save(self.state, self.clock.now())\n        self._report_orders()\n",
        "        self.journal.save(self.state, self.clock.now())\n",
        (UNIT_EXEC,),
    ),
    Mutation(
        "audit: fills not published",
        EXEC_BUS,
        "        om.on_fill = self._fill\n",
        "",
        (UNIT_EXEC,),
    ),
    Mutation(
        "audit: a restart reports old order changes again",
        OM,
        "self._reported: dict[str, _Seen] = {k: _seen(t) for k, t in self.state.orders.items()}",
        "self._reported: dict[str, _Seen] = {}",
        (UNIT_EXEC,),
    ),
    Mutation(
        "alerts: slippage above the model is not warned",
        MON_STATE,
        "        if real <= model:",
        "        if real <= model * 10:",
        (MON_T,),
    ),
    Mutation(
        "alerts: data quality warns about symbols nobody trades",
        QUALITY,
        "if i.symbol in traded and i.severity != Severity.LOW:",
        "if i.severity != Severity.LOW:",
        (DATA_T,),
    ),
    Mutation(
        "risk: daily loss halt ignored",
        GATE,
        '        if loss["day"] is not None and loss["day"] >= lim.daily_loss_halt:',
        "        if False:",
        (RISK,),
    ),
    Mutation(
        "risk: weekly loss halt ignored",
        GATE,
        '        if loss["week"] is not None and loss["week"] >= lim.weekly_loss_halt:',
        "        if False:",
        (RISK,),
    ),
    Mutation(
        "risk: peak drawdown full halt ignored",
        GATE,
        '        if loss["peak"] is not None and loss["peak"] >= lim.peak_drawdown_full_halt:',
        "        if False:",
        (RISK,),
    ),
    Mutation(
        "risk service: trades blind when the calendar is stale",
        RISK_SVC,
        "        elif self.news is not None and events is None:",
        "        elif False:",
        (RISK_SVC_T,),
    ),
    Mutation(
        "risk service: news never reaches the gate",
        RISK_SVC,
        "            events=events,\n",
        "",
        (RISK_SVC_T,),
    ),
    Mutation(
        "oanda: a netting account passes as hedging",
        OANDA,
        'margin_mode="hedging" if a.get("hedgingEnabled") else "netting",',
        'margin_mode="hedging",',
        (OANDA_T,),
    ),
    Mutation(
        "oanda: orders sent without their stop",
        OANDA,
        '            "stopLossOnFill": {"price": str(req.sl), "timeInForce": "GTC"},\n',
        "",
        (OANDA_T,),
    ),
    Mutation(
        "oanda: a rejected token is retried as an outage",
        OANDA,
        "        if r.status_code in (401, 403):",
        "        if False:",
        (OANDA_T,),
    ),
    Mutation(
        "api: a public hub serves data without the token",
        API,
        "    if protect_reads:\n",
        "    if False:\n",
        (API_T,),
    ),
    Mutation(
        "ctrader: a rejected token is retried as an outage",
        CTRADER,
        "        except (PermissionError, BrokerUnavailableError):\n",
        "        except BrokerUnavailableError:\n",
        (CTRADER_T,),
    ),
    Mutation(
        "ctrader: a netting account passes as hedging",
        CTRADER,
        'margin_mode="hedging" if _enum(tr.get("accountType", HEDGED)) == HEDGED else "netting",',
        'margin_mode="hedging",',
        (CTRADER_T,),
    ),
)
