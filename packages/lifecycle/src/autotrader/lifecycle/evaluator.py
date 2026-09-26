"""Evaluator (spec section 10): runs every hour and after every closed trade.

Per version it computes stage metrics from the trade table, applies config/promotion.yaml, writes the
stage change (with a metrics snapshot) and acts on demotions immediately: pending orders of a demoted
version are cancelled; its positions are closed at market if it drops to shadow or retired, and kept
with their stops if it drops one money stage.

Shadow exit: after min_weeks and min_signals, the entry rate and mean R must both sit inside the
backtest's `band` interval -> micro; otherwise the version diverges from its backtest -> retired.
demo_only versions are never promoted (capped at shadow).

Demotion rules (money stages; trades since the version last entered micro from shadow):
  drawdown in R > dd_vs_mc_p95 x Monte Carlo p95 drawdown
  profit factor of the last rolling_pf_window trades < rolling_pf_min
  mean slippage of the last slippage_window fills > slippage_vs_model_max x model slippage
  win rate or mean R of the last drift_window trades outside the backtest drift_interval band
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from autotrader.core.alerts import Alert, AlertSink, Severity
from autotrader.core.broker import ExecutionQuality
from autotrader.core.clock import Clock
from autotrader.core.events import StageChanged
from autotrader.core.models import Stage, Trade
from autotrader.core.profile import BacktestProfile
from autotrader.lifecycle.champion import Record, rollback_due, swap_test
from autotrader.lifecycle.config import PromotionConfig
from autotrader.lifecycle.registry import DEMOTIONS, Registry, VersionState
from autotrader.lifecycle.stats import (
    annualized_sharpe,
    bootstrap_mean_band,
    daily_r,
    inside,
    max_drawdown_r,
    mean,
    profit_factor,
)

Metrics = dict[str, float | int | str | bool | None]
SHADOW_ACCOUNT = "shadow"


def champion_record(trades: Sequence[Trade], since: datetime, now: datetime) -> Record:
    """A version's R per closed trade since `since`, in exit order, for the swap test."""
    ts = sorted(trades, key=lambda t: t.exit_time)
    return Record(r=[t.r_multiple for t in ts], first_at=since, last_at=now)


def stable_seed(key: tuple[str, str]) -> int:
    """A bootstrap seed from the version's key: the same test on the same data gives the same p."""
    return int(hashlib.sha256("|".join(key).encode()).hexdigest()[:8], 16)


class StageData(Protocol):
    """Where the evaluator reads trades, signals and fills (trade journal / execution quality)."""

    def trades(self, strategy_id: str, version: str, since: datetime, *, shadow: bool) -> list[Trade]: ...

    def signal_count(self, strategy_id: str, version: str, since: datetime) -> int: ...

    def fills(self, strategy_id: str, version: str, since: datetime) -> list[ExecutionQuality]: ...


class DemotionActions(Protocol):
    """Execution side of a demotion (execution.OrderManager implements it)."""

    async def cancel_pending_of(self, strategy_id: str, version: str, reason: str) -> None: ...

    async def close_positions_of(self, strategy_id: str, version: str, reason: str) -> None: ...


@dataclass
class MemoryStageData:
    """In-memory StageData for tests and shadow sessions until the trade journal (Phase 9) exists."""

    trade_rows: list[Trade] = field(default_factory=list)
    signal_times: dict[tuple[str, str], list[datetime]] = field(default_factory=dict)
    fill_rows: list[ExecutionQuality] = field(default_factory=list)

    def add_trade(self, t: Trade) -> None:
        self.trade_rows.append(t)

    def add_signal(self, strategy_id: str, version: str, at: datetime) -> None:
        self.signal_times.setdefault((strategy_id, version), []).append(at)

    def trades(self, strategy_id: str, version: str, since: datetime, *, shadow: bool) -> list[Trade]:
        return sorted(
            (
                t
                for t in self.trade_rows
                if t.strategy_id == strategy_id
                and t.strategy_version == version
                and t.entry_time >= since
                and (t.account_id == SHADOW_ACCOUNT) == shadow
            ),
            key=lambda t: t.exit_time,
        )

    def signal_count(self, strategy_id: str, version: str, since: datetime) -> int:
        return sum(1 for t in self.signal_times.get((strategy_id, version), []) if t >= since)

    def fills(self, strategy_id: str, version: str, since: datetime) -> list[ExecutionQuality]:
        return [
            f
            for f in self.fill_rows
            if f.strategy_id == strategy_id and f.strategy_version == version and f.filled_at >= since
        ]


def _rs(trades: Sequence[Trade]) -> list[float]:
    return [t.r_multiple for t in trades]


class Evaluator:
    def __init__(
        self,
        registry: Registry,
        config: PromotionConfig,
        data: StageData,
        clock: Clock,
        alerts: AlertSink,
        actions: DemotionActions | None = None,
    ) -> None:
        self.reg = registry
        self.cfg = config
        self.data = data
        self.clock = clock
        self.alerts = alerts
        self.actions = actions

    async def evaluate_all(self) -> list[StageChanged]:
        out = []
        for v in self.reg.versions():
            ev = await self.evaluate(*v.key)
            if ev is not None:
                out.append(ev)
        out += self.review_challengers()
        return out

    # ------------------------------------------------------------ champion vs challenger (spec 14.8)

    def review_challengers(self) -> list[StageChanged]:
        """Every re-optimized challenger in shadow against its champion in a money stage: the swap test on
        R per closed trade since the challenger entered shadow. Each test is recorded; a win swaps."""
        now = self.clock.now()
        out: list[StageChanged] = []
        shadow = [
            v
            for v in self.reg.versions(Stage.SHADOW)
            if v.info.origin == "learning_reopt" and v.info.parent_version is not None
        ]
        for chal in sorted(shadow, key=lambda v: v.key):
            sid, parent = chal.info.strategy_id, chal.info.parent_version
            if parent is None:
                continue
            champ = self.reg.get(sid, parent)
            if champ.stage not in (Stage.MICRO, Stage.LIVE, Stage.SCALED):
                continue
            since = chal.stage_since
            k = sum(1 for v in shadow if v.info.strategy_id == sid and v.info.parent_version == parent)
            a = champion_record(self.data.trades(sid, parent, since, shadow=False), since, now)
            b = champion_record(self.data.trades(sid, chal.info.version, since, shadow=True), since, now)
            last = self.reg.last_swap(sid)
            verdict = swap_test(
                a, b, now=now, k=k, last_swap=last[0] if last else None, seed=stable_seed(chal.key)
            )
            self.reg.record_swap_test(
                {
                    "at": now.isoformat(),
                    "strategy_id": sid,
                    "champion": parent,
                    "challenger": chal.info.version,
                    "k": k,
                    "swap": verdict.swap,
                    "diff": verdict.diff,
                    "p_value": verdict.p_value,
                    "champion_dd": verdict.champion_dd,
                    "challenger_dd": verdict.challenger_dd,
                    "champion_signals": len(a.r),
                    "challenger_signals": len(b.r),
                    "reasons": verdict.reasons,
                }
            )
            if verdict.swap:
                out += self.reg.swap(
                    sid,
                    parent,
                    chal.info.version,
                    f"{verdict.diff:+.3f}R per signal, p {verdict.p_value:.3f}",
                    {"diff": verdict.diff, "p_value": verdict.p_value},
                )
        return out

    def _maybe_rollback(self, ev: StageChanged) -> list[StageChanged]:
        """A new champion demoted within 4 weeks of its swap: the old champion comes back."""
        if (ev.from_stage, ev.to_stage) not in DEMOTIONS and ev.to_stage != Stage.RETIRED:
            return []
        last = self.reg.last_swap(ev.strategy_id)
        if last is None or last[1] != ev.strategy_version or not rollback_due(last[0], ev.at):
            return []
        return list(self.reg.rollback(ev.strategy_id, f"{ev.strategy_version} demoted: {ev.reason}"))

    async def on_trade_closed(self, trade: Trade) -> StageChanged | None:
        return await self.evaluate(trade.strategy_id, trade.strategy_version)

    async def evaluate(self, strategy_id: str, version: str) -> StageChanged | None:
        v = self.reg.get(strategy_id, version)
        if v.stage == Stage.SHADOW:
            return self._shadow(v)
        if v.stage in (Stage.MICRO, Stage.LIVE, Stage.SCALED):
            ev = await self._money(v)
            if ev is not None:
                self._maybe_rollback(ev)  # its events reach the bus through the registry's callback
            return ev
        return None

    # ------------------------------------------------------------ shadow

    def _shadow(self, v: VersionState) -> StageChanged | None:
        ex = self.cfg.shadow.exit
        now = self.clock.now()
        weeks = (now - v.stage_since) / timedelta(weeks=1)
        signals = self.data.signal_count(*v.key, since=v.stage_since)
        if weeks < ex.min_weeks or signals < ex.min_signals:
            return None
        if v.info.demo_only:
            return None  # capped at shadow forever
        if v.profile is None:
            return None  # cannot compare; registry refuses candidates without a profile anyway
        trades = self.data.trades(*v.key, since=v.stage_since, shadow=True)
        entries_per_week = len(trades) / weeks
        k = max(1, round(weeks))
        rate_band = bootstrap_mean_band([float(x) for x in v.profile.weekly_entries], k, ex.band)
        m: Metrics = {
            "weeks": round(weeks, 2),
            "signals": signals,
            "trades": len(trades),
            "entries_per_week": entries_per_week,
        }
        m["rate_band_lo"], m["rate_band_hi"] = rate_band
        ok = inside(entries_per_week, rate_band)
        if trades:
            r_band = bootstrap_mean_band(v.profile.trade_r, len(trades), ex.band)
            m["avg_r"], m["r_band_lo"], m["r_band_hi"] = mean(_rs(trades)), *r_band
            ok = ok and inside(mean(_rs(trades)), r_band)
        else:
            ok = False
        if ok:
            return self.reg.transition(
                *v.key, Stage.MICRO, "shadow matches backtest", actor="evaluator", metrics=m
            )
        return self.reg.transition(
            *v.key, Stage.RETIRED, "shadow diverges from backtest", actor="evaluator", metrics=m
        )

    # ------------------------------------------------------------ money stages

    def demotion_reasons(
        self, v: VersionState, trades: list[Trade], fills: list[ExecutionQuality]
    ) -> tuple[list[str], Metrics]:
        d = self.cfg.demotion
        prof = v.profile
        rs = _rs(trades)
        reasons: list[str] = []
        m: Metrics = {"trades": len(trades)}
        if prof is not None:
            dd = max_drawdown_r(rs)
            m["dd_r"], m["mc_dd_p95_r"] = dd, prof.mc_dd_p95_r
            if dd > d.dd_vs_mc_p95 * prof.mc_dd_p95_r:
                reasons.append(f"drawdown {dd:.2f}R > {d.dd_vs_mc_p95} x MC p95 {prof.mc_dd_p95_r:.2f}R")
        if len(rs) >= d.rolling_pf_window:
            pf = profit_factor(rs[-d.rolling_pf_window :])
            m["rolling_pf"] = pf
            if pf < d.rolling_pf_min:
                reasons.append(f"rolling PF {pf:.2f} < {d.rolling_pf_min}")
        ratio = (
            self._slippage_ratio(fills[-d.slippage_window :], prof)
            if len(fills) >= d.slippage_window
            else None
        )
        if ratio is not None:
            m["slippage_vs_model"] = ratio
            if ratio > d.slippage_vs_model_max:
                reasons.append(f"slippage {ratio:.2f} x model > {d.slippage_vs_model_max}")
        if prof is not None and len(rs) >= d.drift_window:
            window = rs[-d.drift_window :]
            wins = [1.0 if r > 0 else 0.0 for r in prof.trade_r]
            wr, ar = mean([1.0 if r > 0 else 0.0 for r in window]), mean(window)
            wr_band = bootstrap_mean_band(wins, d.drift_window, d.drift_interval)
            ar_band = bootstrap_mean_band(prof.trade_r, d.drift_window, d.drift_interval)
            m["win_rate"], m["avg_r"] = wr, ar
            if not inside(wr, wr_band):
                reasons.append(f"win rate {wr:.2f} outside backtest band {wr_band[0]:.2f}..{wr_band[1]:.2f}")
            if not inside(ar, ar_band):
                reasons.append(f"avg R {ar:.2f} outside backtest band {ar_band[0]:.2f}..{ar_band[1]:.2f}")
        return reasons, m

    @staticmethod
    def _slippage_ratio(fills: Sequence[ExecutionQuality], prof: BacktestProfile | None) -> float | None:
        if prof is None:
            return None
        model = [prof.model_slippage.get(f.symbol, 0.0) for f in fills]
        actual = [float(f.slippage) for f in fills if f.slippage is not None]
        if not actual or mean(model) <= 0:
            return None
        return mean(actual) / mean(model)

    async def _money(self, v: VersionState) -> StageChanged | None:
        since = v.money_since() or v.stage_since
        trades = self.data.trades(*v.key, since=since, shadow=False)
        fills = self.data.fills(*v.key, since=since)
        reasons, m = self.demotion_reasons(v, trades, fills)
        if reasons:
            return await self._demote(v, "; ".join(reasons), m)
        stage_trades = [t for t in trades if t.entry_time >= v.stage_since]
        if v.stage == Stage.MICRO:
            return self._micro_exit(v, stage_trades, fills, m)
        ex = self.cfg.live.exit
        # rolling: the last `min_trades` money trades (the same window the live exit is judged on)
        sharpe = annualized_sharpe(daily_r(trades[-ex.min_trades :]))
        m["rolling_sharpe"] = sharpe
        if v.stage == Stage.LIVE and len(stage_trades) >= ex.min_trades and sharpe >= ex.min_rolling_sharpe:
            return self.reg.transition(*v.key, Stage.SCALED, "live targets met", actor="evaluator", metrics=m)
        if v.stage == Stage.SCALED and sharpe < ex.min_rolling_sharpe:
            return await self._demote(
                v, f"rolling Sharpe {sharpe:.2f} slipped below {ex.min_rolling_sharpe}", m
            )
        return None

    def _micro_exit(
        self, v: VersionState, trades: list[Trade], fills: list[ExecutionQuality], m: Metrics
    ) -> StageChanged | None:
        ex = self.cfg.micro.exit
        if len(trades) < ex.min_trades or v.profile is None:
            return None
        ratio = self._slippage_ratio(fills, v.profile)
        band = bootstrap_mean_band(v.profile.trade_r, len(trades), ex.band)
        avg = mean(_rs(trades))
        m.update({"slippage_vs_model": ratio, "avg_r": avg, "r_band_lo": band[0], "r_band_hi": band[1]})
        if (ratio is not None and ratio > ex.max_slippage_vs_model) or not inside(avg, band):
            return None  # not ready; demotion rules decide whether it has to go
        week_ago = self.clock.now() - timedelta(days=7)
        if self.reg.promotions_to(Stage.LIVE, week_ago) >= self.cfg.global_.max_promotions_to_live_per_week:
            m["waiting"] = "weekly promotion limit"
            return None
        return self.reg.transition(*v.key, Stage.LIVE, "micro targets met", actor="evaluator", metrics=m)

    async def _demote(self, v: VersionState, reason: str, m: Metrics) -> StageChanged:
        d = self.cfg.demotion
        window_start = self.clock.now() - timedelta(days=round(30.44 * d.retire_window_months))
        if v.stage == Stage.SCALED:
            to = Stage.LIVE
        elif v.stage == Stage.LIVE:
            to = Stage.MICRO
        elif v.demotions_since(window_start) + 1 >= d.retire_after_demotions:
            to = Stage.RETIRED
        else:
            # SPEC-QUESTION: first demotion while in micro goes back to shadow (no money) rather than staying
            to = Stage.SHADOW
        ev = self.reg.transition(*v.key, to, reason, actor="evaluator", metrics=m)
        await self._act(v, to, reason)
        return ev

    async def _act(self, v: VersionState, to: Stage, reason: str) -> None:
        if self.actions is None:
            return
        try:
            await self.actions.cancel_pending_of(*v.key, reason)
            if to in (Stage.SHADOW, Stage.RETIRED):
                await self.actions.close_positions_of(*v.key, reason)
        except Exception as e:  # the stage change stands; execution must retry, the owner must know
            self.alerts.send(
                Alert(
                    severity=Severity.CRITICAL,
                    kind="demotion_action_failed",
                    message=f"{v.key}: {type(e).__name__}: {e}",
                    at=self.clock.now(),
                )
            )
