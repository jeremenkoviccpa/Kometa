"""A historical trade journal: a strategy replayed over past data, each signal that became a trade recorded
with the same market snapshot the live journal takes (spec 14.2) and the trade's real outcome, so meta
models (L3) can learn from years of signals instead of the few a demo has made.

The snapshot sees only bars closed at the signal time (features.snapshot drops the rest itself), from a
full set of timeframes resampled from the same M1 data, whatever timeframes the strategy itself uses.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl

from autotrader.core.models import Instrument, Timeframe
from autotrader.core.series import BarsArray, from_ns
from autotrader.engine.backtest import BacktestConfig, run_backtest
from autotrader.learning.features import snapshot
from autotrader.learning.journal import KEEP, JournalEntry, Outcome
from autotrader.strategies_api.base import Strategy
from autotrader.strategies_api.manifest import ParamValue
from autotrader.validation.inputs import prepare
from autotrader.validation.runner import with_manifest

LABEL = {"target": 1, "stop": -1}  # anything else (a close, the end of the data) timed out: 0


def _last(b: BarsArray, now_ns: int, n: int) -> BarsArray:
    j = int(np.searchsorted(b.close_time, now_ns, side="right"))
    i = max(0, j - n)
    return BarsArray(**{f: getattr(b, f)[i:j] for f in b.__dataclass_fields__})


def journal_from_backtest(
    cls: type[Strategy],
    frame: pl.DataFrame,
    symbol: str,
    instrument: Instrument,
    *,
    params: Mapping[str, ParamValue] | None = None,
    risk_fraction: float = 0.005,
) -> list[JournalEntry]:
    """Every trade of a backtest as a journal entry: features at the signal, the real R, the exit as label."""
    strat = with_manifest(cls, symbols=[symbol])
    inp = prepare({symbol: frame}, strat.manifest, {symbol: instrument})
    res = run_backtest(
        strat, inp.m1, inp.series, inp.instruments, inp.cost_model, params=params,
        config=BacktestConfig(risk_fraction=risk_fraction),
    )  # fmt: skip
    # the features' own timeframes, from the same M1, whatever the strategy subscribes to
    full = with_manifest(cls, symbols=[symbol], timeframes=[Timeframe.M1, *KEEP])
    bars = {
        tf: b for (_, tf), b in prepare({symbol: frame}, full.manifest, {symbol: instrument}).series.items()
    }
    out: list[JournalEntry] = []
    for t in sorted(res.trades, key=lambda t: t.entry_time_ns):
        now = from_ns(t.entry_time_ns)
        seen = {tf: _last(bars[tf], t.entry_time_ns, n) for tf, n in KEEP.items()}
        m5 = seen[Timeframe.M5]
        # the spread quoted on the last closed bar: the cost model is estimated from all the data (the future)
        quoted = float(m5.ask_c[-1] - m5.bid_c[-1]) if len(m5) else 0.0
        feats = snapshot(seen, now, spread=quoted)
        done = [e.outcome for e in out[-20:] if e.outcome is not None]
        risk = abs(t.entry_price - t.stop_price)
        out.append(
            JournalEntry(
                signal_id=t.signal_id,
                strategy_id=cls.manifest.id,
                strategy_version=cls.manifest.version,
                symbol=symbol,
                side=t.side,
                timeframe=min(cls.manifest.timeframes, key=lambda x: x.minutes),
                created_at=now,
                entry=t.entry_price,
                stop=t.stop_price,
                target=None,
                shadow=False,
                status="approved",
                strategy_win_rate_20=sum(o.label == 1 for o in done) / len(done) if done else None,
                features=feats,
                mae_r=t.mae_r,
                mfe_r=t.mfe_r,
                outcome=Outcome(
                    label=LABEL.get(t.exit_reason, 0),
                    r=t.r_multiple,
                    mae_r=t.mae_r,
                    mfe_r=t.mfe_r,
                    minutes=int((t.exit_time_ns - t.entry_time_ns) // 60_000_000_000),
                    resolved_at=from_ns(t.exit_time_ns),
                ),
                trade_r=t.r_multiple if risk > 0 else None,
            )
        )
    return out
