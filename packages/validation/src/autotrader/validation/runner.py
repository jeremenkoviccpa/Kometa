"""Full validation of one strategy version (spec section 9).

Order: walk-forward (tune on train, measure on test) -> out-of-sample metrics,
deflated Sharpe, block-bootstrap Monte Carlo, suspicious-return scan ->
parameter stability -> cross-market -> locked holdout (only if everything
else passed, so holdout attempts are not wasted). Every backtest is recorded
in the trial registry before its result is used.

Returns for Sharpe/DSR are daily R returns: each trade contributes
risk_fraction x R on the day it closes, days without trades count as zero.
This keeps trials comparable regardless of window length or sizing noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl

from autotrader.core.hashing import hash_obj
from autotrader.core.models import Instrument
from autotrader.core.series import BarsArray, from_ns, to_ns
from autotrader.core.timeutil import ensure_utc
from autotrader.engine.backtest import BacktestConfig, BacktestResult, run_backtest
from autotrader.engine.costs import rollover_times
from autotrader.engine.simbroker import TradeRecord
from autotrader.strategies_api.base import Strategy
from autotrader.strategies_api.loader import LoadedStrategy
from autotrader.strategies_api.manifest import ParamValue, StrategyManifest
from autotrader.validation.config import ValidationConfig
from autotrader.validation.dsr import DSRResult, deflated_sharpe, sharpe
from autotrader.validation.inputs import EngineInputs, prepare
from autotrader.validation.montecarlo import MonteCarloResult, monte_carlo
from autotrader.validation.store import HoldoutLock, HoldoutRefusedError, Trial, TrialKind, TrialRegistry

DAYS_PER_MONTH = 30.4375


# ---------------------------------------------------------------- helpers


def add_months(t: datetime, months: float) -> datetime:
    return t + timedelta(days=DAYS_PER_MONTH * months)


def _slice(b: BarsArray, start_ns: int, end_ns: int) -> BarsArray:
    lo = int(np.searchsorted(b.open_time, start_ns, side="left"))
    hi = int(np.searchsorted(b.close_time, end_ns, side="right"))
    return b.slice(lo, max(lo, hi))


def slice_inputs(
    inp: EngineInputs, start_ns: int, end_ns: int, symbols: Sequence[str] | None = None
) -> EngineInputs:
    syms = list(symbols) if symbols is not None else list(inp.m1)
    return replace(
        inp,
        m1={s: _slice(inp.m1[s], start_ns, end_ns) for s in syms},
        series={k: _slice(v, start_ns, end_ns) for k, v in inp.series.items() if k[0] in syms},
        instruments={s: inp.instruments[s] for s in syms},
    )


def with_manifest(cls: type[Strategy], **changes: Any) -> type[Strategy]:
    m = cls.manifest.model_copy(update=changes)
    return type(cls.__name__, (cls,), {"manifest": m})


def profit_factor(r: Sequence[float] | np.ndarray) -> float:
    a = np.asarray(r, dtype=np.float64)
    wins, losses = float(a[a > 0].sum()), float(-a[a < 0].sum())
    if losses > 0:
        return wins / losses
    return math.inf if wins > 0 else 0.0


def daily_r_returns(
    trades: Sequence[TradeRecord], start_ns: int, end_ns: int, risk_fraction: float
) -> np.ndarray:
    days = np.asarray([t for t, _ in rollover_times(from_ns(start_ns), from_ns(end_ns))], dtype=np.int64)
    out = np.zeros(days.size + 1)
    for tr in trades:
        out[int(np.searchsorted(days, tr.exit_time_ns, side="left"))] += risk_fraction * tr.r_multiple
    return out


def warmup_pad(cls: type[Strategy]) -> timedelta:
    need = cls().warmup()
    minutes = max((n * tf.minutes for (_, tf), n in need.items()), default=0)
    return timedelta(minutes=minutes) * 1.5 + timedelta(days=3)


def param_candidates(manifest: StrategyManifest, budget: int, seed: int) -> list[dict[str, ParamValue]]:
    rng = np.random.default_rng(seed)
    base = manifest.param_values()
    tunable = {k: p for k, p in manifest.params.items() if p.tunable}
    out = [base]
    attempts = 0
    while len(out) < budget and tunable and attempts < budget * 50:
        attempts += 1
        c = dict(base)
        for k, p in tunable.items():
            lo, hi = float(p.min or 0.0), float(p.max or 0.0)
            v = rng.uniform(lo, hi)
            c[k] = round(v) if isinstance(p.value, int) and not isinstance(p.value, bool) else round(v, 6)
        if c not in out:
            out.append(c)
    return out


def objective(r: np.ndarray, min_trades: int) -> float:
    """t-statistic of mean R: rewards edge and consistency, not a few lucky trades."""
    if r.size < min_trades:
        return -math.inf
    sd = float(r.std(ddof=1))
    return -math.inf if sd == 0 else float(r.mean()) / sd * math.sqrt(r.size)


def _overlaps(t: TradeRecord, windows: Sequence[tuple[int, int]]) -> bool:
    return any(t.entry_time_ns <= e and t.exit_time_ns >= s for s, e in windows)


# ---------------------------------------------------------------- report model


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    threshold: float
    passed: bool
    op: str  # ">=", "<=", "=="


@dataclass(frozen=True)
class WindowResult:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    params: dict[str, ParamValue]
    train_score: float
    test_trades: int
    test_sharpe: float
    test_pf: float


@dataclass(frozen=True)
class VariantResult:
    label: str
    params: dict[str, ParamValue]
    trades: int
    profit_factor: float
    passed: bool


@dataclass
class ValidationReport:
    strategy_id: str
    version: str
    family: str
    code_hash: str
    config_hash: str
    data_versions: dict[str, str]
    synthetic: bool
    research_window: tuple[datetime, datetime]
    holdout_window: tuple[datetime, datetime]
    holdout_epoch: str
    windows: list[WindowResult]
    final_params: dict[str, ParamValue]
    oos_trades: list[TradeRecord]
    oos_daily_returns: np.ndarray
    risk_fraction: float
    dsr: DSRResult | None
    monte_carlo: MonteCarloResult | None
    monthly_returns: dict[str, float]
    stability: list[VariantResult]
    cross_market: list[VariantResult]
    holdout: VariantResult | None
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded_trades: int = 0

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def eligible_for_promotion(self) -> bool:
        return self.passed and not self.synthetic


@dataclass(frozen=True)
class HoldoutEpoch:
    epoch: str
    start: datetime
    end: datetime


# ---------------------------------------------------------------- runner


class Validator:
    def __init__(
        self,
        cfg: ValidationConfig,
        registry: TrialRegistry,
        holdout_lock: HoldoutLock,
        *,
        config_hash: str = "",
        risk_fraction: float = 0.005,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.lock = holdout_lock
        self.config_hash = config_hash
        self.bt = BacktestConfig(risk_fraction=risk_fraction)
        self.seed = seed

    # one backtest + one trial record
    def _run(
        self,
        cls: type[Strategy],
        inp: EngineInputs,
        params: Mapping[str, ParamValue],
        kind: TrialKind,
        start_ns: int,
        end_ns: int,
        code_hash: str,
        count_from_ns: int | None = None,
    ) -> tuple[BacktestResult, list[TradeRecord], np.ndarray]:
        res = run_backtest(
            cls, inp.m1, inp.series, inp.instruments, inp.cost_model, params=params, config=self.bt
        )
        lo = count_from_ns if count_from_ns is not None else start_ns
        trades = [t for t in res.trades if lo <= t.entry_time_ns < end_ns]
        rets = daily_r_returns(trades, lo, end_ns, self.bt.risk_fraction)
        m = cls.manifest
        self.registry.record(
            Trial(
                family=m.family,
                strategy_id=m.id,
                version=m.version,
                params_hash=hash_obj(dict(params)),
                kind=kind,
                data_version=hash_obj(inp.data_versions),
                window_start=from_ns(lo).isoformat(),
                window_end=from_ns(end_ns).isoformat(),
                sharpe=sharpe(rets),
                trades=len(trades),
                code_hash=code_hash,
                note=",".join(m.symbols),
            )
        )
        return res, trades, rets

    def validate(
        self,
        strategy: LoadedStrategy,
        frames: Mapping[str, pl.DataFrame],
        instruments: Mapping[str, Instrument],
        epoch: HoldoutEpoch,
        *,
        universe: Sequence[str] | None = None,
        quality_exclusions: Sequence[tuple[datetime, datetime]] = (),
        synthetic: bool = False,
    ) -> ValidationReport:
        cfg, th = self.cfg, self.cfg.thresholds
        cls, man = strategy.cls, strategy.manifest
        universe = list(universe or sorted(frames))[: max(cfg.cross_market.of_pairs, len(man.symbols))]
        for s in man.symbols:
            if s not in universe:
                universe.insert(0, s)
        hs, he = ensure_utc(epoch.start), ensure_utc(epoch.end)

        # research data ends where the holdout starts; the holdout is prepared separately
        research = {s: frames[s].filter(pl.col("open_time") < hs) for s in universe}
        uni_manifest = man.model_copy(update={"symbols": tuple(universe)})
        rin = prepare(research, uni_manifest, instruments, synthetic=synthetic)
        r_start = max(int(rin.m1[s].open_time[0]) for s in man.symbols)
        r_end = min(int(rin.m1[s].close_time[-1]) for s in man.symbols)
        pad_ns = int(warmup_pad(cls).total_seconds() * 1e9)
        exclusions = [(to_ns(ensure_utc(a)), to_ns(ensure_utc(b))) for a, b in quality_exclusions]

        # ---- walk-forward
        windows: list[WindowResult] = []
        oos: list[TradeRecord] = []
        oos_rets: list[np.ndarray] = []
        wf = cfg.walk_forward
        t0 = from_ns(r_start)
        k = 0
        while True:
            tr_s, tr_e = t0, add_months(t0, wf.train_years * 12)
            te_s, te_e = tr_e, add_months(tr_e, wf.test_months)
            if to_ns(te_e) > r_end:
                break
            best, best_score = None, -math.inf
            train_in = slice_inputs(rin, to_ns(tr_s), to_ns(tr_e), man.symbols)
            for c in param_candidates(man, wf.search_budget, self.seed + k):
                _, trades, _ = self._run(
                    cls, train_in, c, "wf_train", to_ns(tr_s), to_ns(tr_e), strategy.code_hash
                )
                score = objective(np.asarray([t.r_multiple for t in trades]), wf.min_train_trades)
                if score > best_score:
                    best, best_score = c, score
            chosen = best or man.param_values()
            test_in = slice_inputs(rin, to_ns(te_s) - pad_ns, to_ns(te_e), man.symbols)
            _, trades, rets = self._run(
                cls,
                test_in,
                chosen,
                "wf_test",
                to_ns(te_s) - pad_ns,
                to_ns(te_e),
                strategy.code_hash,
                to_ns(te_s),
            )
            oos += trades
            oos_rets.append(rets)
            r = np.asarray([t.r_multiple for t in trades])
            windows.append(
                WindowResult(
                    tr_s, tr_e, te_s, te_e, chosen, best_score, len(trades), sharpe(rets), profit_factor(r)
                )
            )
            t0 = add_months(t0, wf.step_months)
            k += 1

        warnings: list[str] = []
        if not windows:
            warnings.append("research data too short for a single walk-forward window")
        for a, b in pairwise(windows):
            for p, v in b.params.items():
                old = a.params.get(p)
                if (
                    isinstance(v, (int, float))
                    and isinstance(old, (int, float))
                    and old
                    and abs(v / old - 1) > 0.5
                ):
                    warnings.append(f"param {p} jumped {old} -> {v} at {b.test_start.date()}")

        excluded = [t for t in oos if _overlaps(t, exclusions)]
        oos_used = [t for t in oos if not _overlaps(t, exclusions)]
        r_oos = np.asarray([t.r_multiple for t in oos_used])
        daily = np.concatenate(oos_rets) if oos_rets else np.zeros(0)
        final_params = windows[-1].params if windows else man.param_values()

        # ---- statistics on the concatenated out-of-sample record
        mc = None
        if r_oos.size:
            bl = cfg.monte_carlo.block_length
            mc = monte_carlo(
                r_oos,
                runs=cfg.monte_carlo.runs,
                risk_fraction=0.005,
                block_length=None if bl == "auto" else int(bl),
                method=cfg.monte_carlo.method,
                seed=self.seed,
            )
        monthly = self._monthly(oos_used)

        # ---- stability: every tunable param +-shift, one at a time, on the full research range
        full_in = slice_inputs(rin, r_start, r_end, man.symbols)
        stability: list[VariantResult] = []
        for pname, spec in man.params.items():
            if not spec.tunable:
                continue
            for sgn in (+1, -1):
                v0 = final_params[pname]
                if not isinstance(v0, (int, float)) or isinstance(v0, bool):
                    raise TypeError(f"tunable param {pname} is not numeric")
                v = min(
                    max(v0 * (1 + sgn * cfg.stability.param_shift), float(spec.min or v0)),
                    float(spec.max or v0),
                )
                v = round(v) if isinstance(spec.value, int) else v
                variant = {**final_params, pname: v}
                _, trades, _ = self._run(
                    cls, full_in, variant, "stability", r_start, r_end, strategy.code_hash
                )
                pf = profit_factor([t.r_multiple for t in trades])
                stability.append(
                    VariantResult(
                        f"{pname} {'+' if sgn > 0 else '-'}{cfg.stability.param_shift:.0%}",
                        variant,
                        len(trades),
                        pf,
                        pf >= th.min_stability_profit_factor,
                    )
                )

        # ---- cross-market: final params on each symbol of the universe on its own
        cross: list[VariantResult] = []
        for s in universe[: cfg.cross_market.of_pairs]:
            one = with_manifest(cls, symbols=(s,))
            s_in = slice_inputs(rin, int(rin.m1[s].open_time[0]), int(rin.m1[s].close_time[-1]), [s])
            _, trades, _ = self._run(
                one,
                s_in,
                final_params,
                "cross_market",
                int(s_in.m1[s].open_time[0]),
                int(s_in.m1[s].close_time[-1]),
                strategy.code_hash,
            )
            pf = profit_factor([t.r_multiple for t in trades])
            cross.append(
                VariantResult(s, final_params, len(trades), pf, pf >= th.min_stability_profit_factor)
            )

        # ---- deflated Sharpe, after every pre-holdout trial of this run is in the registry
        dsr = deflated_sharpe(daily, self.registry.family_sharpes(man.family)) if daily.size > 2 else None

        # ---- checks before the holdout
        checks = [
            Check("oos_trades", float(r_oos.size), th.min_oos_trades, r_oos.size >= th.min_oos_trades, ">="),
            Check(
                "oos_profit_factor",
                profit_factor(r_oos),
                th.min_profit_factor,
                profit_factor(r_oos) >= th.min_profit_factor,
                ">=",
            ),
            Check(
                "deflated_sharpe_prob",
                dsr.probability if dsr else 0.0,
                th.min_deflated_sharpe_prob,
                bool(dsr and dsr.probability >= th.min_deflated_sharpe_prob),
                ">=",
            ),
            Check(
                "mc_dd_p95_at_0.5pct",
                mc.dd_p95 if mc else 1.0,
                th.max_mc_dd_p95_at_half_percent_risk,
                bool(mc and mc.dd_p95 <= th.max_mc_dd_p95_at_half_percent_risk),
                "<=",
            ),
            Check(
                "stability_variants_passing",
                float(sum(v.passed for v in stability)),
                float(len(stability)),
                all(v.passed for v in stability),
                "==",
            ),
            Check(
                "cross_market_passing",
                float(sum(v.passed for v in cross)),
                cfg.cross_market.min_pairs_passing,
                sum(v.passed for v in cross) >= cfg.cross_market.min_pairs_passing,
                ">=",
            ),
            Check(
                "max_monthly_return",
                max(monthly.values(), default=0.0),
                th.suspicious_monthly_return,
                max(monthly.values(), default=0.0) <= th.suspicious_monthly_return,
                "<=",
            ),
        ]
        if max(monthly.values(), default=0.0) > th.suspicious_monthly_return:
            warnings.append("monthly return above the suspicious threshold: probable bug, promotion blocked")

        # ---- locked holdout, only when everything else passed
        holdout: VariantResult | None = None
        if all(c.passed for c in checks):
            try:
                self.lock.open(man.family, man.id, man.version)
            except HoldoutRefusedError as e:
                warnings.append(str(e))
                checks.append(Check("holdout_profit_factor", 0.0, th.min_holdout_profit_factor, False, ">="))
            else:
                full = {s: frames[s].filter(pl.col("open_time") < he) for s in man.symbols}
                h_in_all = prepare(full, man, instruments, spread_frames=research, synthetic=synthetic)
                h_in = slice_inputs(h_in_all, to_ns(hs) - pad_ns, to_ns(he), man.symbols)
                _, trades, _ = self._run(
                    cls,
                    h_in,
                    final_params,
                    "holdout",
                    to_ns(hs) - pad_ns,
                    to_ns(he),
                    strategy.code_hash,
                    to_ns(hs),
                )
                pf = profit_factor([t.r_multiple for t in trades])
                holdout = VariantResult(
                    "holdout", final_params, len(trades), pf, pf >= th.min_holdout_profit_factor
                )
                checks.append(
                    Check("holdout_profit_factor", pf, th.min_holdout_profit_factor, holdout.passed, ">=")
                )
        else:
            checks.append(Check("holdout_profit_factor", 0.0, th.min_holdout_profit_factor, False, ">="))
            warnings.append("holdout not opened: earlier checks failed (attempt not consumed)")

        return ValidationReport(
            strategy_id=man.id,
            version=man.version,
            family=man.family,
            code_hash=strategy.code_hash,
            config_hash=self.config_hash,
            data_versions={s: rin.data_versions[s] for s in universe},
            synthetic=synthetic,
            research_window=(from_ns(r_start), from_ns(r_end)),
            holdout_window=(hs, he),
            holdout_epoch=epoch.epoch,
            windows=windows,
            final_params=dict(final_params),
            oos_trades=oos_used,
            oos_daily_returns=daily,
            risk_fraction=self.bt.risk_fraction,
            dsr=dsr,
            monte_carlo=mc,
            monthly_returns=monthly,
            stability=stability,
            cross_market=cross,
            holdout=holdout,
            checks=checks,
            warnings=warnings,
            excluded_trades=len(excluded),
        )

    def _monthly(self, trades: Sequence[TradeRecord]) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in trades:
            key = from_ns(t.exit_time_ns).strftime("%Y-%m")
            out[key] = out.get(key, 0.0) + self.bt.risk_fraction * t.r_multiple
        return dict(sorted(out.items()))


def default_epoch(data_end: datetime, months: float, name: str | None = None) -> HoldoutEpoch:
    """Convenience for tests and first setup; production epochs are fixed by owner command."""
    end = ensure_utc(data_end)
    start = add_months(end, -months)
    return HoldoutEpoch(name or f"epoch-{start.date()}", start, end)
