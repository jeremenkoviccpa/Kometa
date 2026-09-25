"""Deterministic synthetic M1 bid/ask bars for CI, golden tests and Phases 1-3.

Regime-switching random walk with session-dependent volatility, spread by hour
of week (wider in Asia and at the 17:00 New York rollover), weekend closure,
and optional injected faults whose ground truth is returned for the quality
tests. Never used for promotion decisions (see versioning.SYNTHETIC_PREFIX).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl

from autotrader.core.models import Instrument
from autotrader.core.timeutil import ensure_utc
from autotrader.data.market_hours import FX_HOURS, MarketHours
from autotrader.data.schema import validate_frame

_DEFAULT_START = datetime(2020, 1, 6, tzinfo=UTC)


@dataclass(frozen=True)
class SyntheticSpec:
    symbol: str = "SYNTH"
    start: datetime = _DEFAULT_START
    days: int = 30
    start_price: float = 100.0
    pip_size: float = 0.01
    annual_vol: float = 0.10
    base_spread_pips: float = 1.0
    hours: MarketHours = field(default_factory=lambda: FX_HOURS)
    # per-minute drift in units of sigma during trending regimes; 0 gives a driftless (no-edge) walk
    trend_strength: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class InjectedFault:
    kind: str  # matches quality.IssueType values
    start: datetime
    end: datetime


_MINUTES_PER_YEAR = 252 * 1440


def _session_vol_mult(hour_utc: np.ndarray) -> np.ndarray:
    # Asia quiet, London and NY active, overlap busiest
    mult = np.full(hour_utc.shape, 0.6)
    mult[(hour_utc >= 7) & (hour_utc < 16)] = 1.2
    mult[(hour_utc >= 12) & (hour_utc < 16)] = 1.6
    mult[(hour_utc >= 16) & (hour_utc < 21)] = 1.0
    return mult


def _spread_mult(hour_utc: np.ndarray, ny_minute_of_day: np.ndarray) -> np.ndarray:
    mult = np.where((hour_utc >= 22) | (hour_utc < 6), 1.8, 1.0)
    rollover = (ny_minute_of_day >= 16 * 60 + 55) & (ny_minute_of_day < 17 * 60 + 10)
    return np.where(rollover, mult * 3.0, mult)


def generate(spec: SyntheticSpec) -> pl.DataFrame:
    rng = np.random.default_rng(spec.seed)
    start = ensure_utc(spec.start)
    times = pl.datetime_range(
        start,
        start + timedelta(days=spec.days),
        "1m",
        closed="left",
        time_unit="us",
        time_zone="UTC",
        eager=True,
    )
    frame = pl.DataFrame({"open_time": times}).filter(spec.hours.is_open_expr())
    n = frame.height
    if n == 0:
        raise ValueError("synthetic range contains no open market minutes")

    ny = frame.select(pl.col("open_time").dt.convert_time_zone("America/New_York").alias("ny"))["ny"]
    hour_utc = frame["open_time"].dt.hour().to_numpy().astype(np.int64)
    ny_mod = (ny.dt.hour().cast(pl.Int64) * 60 + ny.dt.minute().cast(pl.Int64)).to_numpy()

    # regimes switch per trading day: drift in {-1, 0, +1} x vol multiplier in {0.7, 1.0, 2.0}
    day_idx = ((frame["open_time"] - start).dt.total_minutes().to_numpy() // 1440).astype(np.int64)
    n_days = int(day_idx.max()) + 1
    regime_drift = rng.choice([-1.0, 0.0, 1.0], size=n_days, p=[0.25, 0.5, 0.25])
    regime_vol = rng.choice([0.7, 1.0, 2.0], size=n_days, p=[0.3, 0.55, 0.15])

    sigma = spec.annual_vol / np.sqrt(_MINUTES_PER_YEAR) * _session_vol_mult(hour_utc) * regime_vol[day_idx]
    drift = regime_drift[day_idx] * sigma * spec.trend_strength
    rets = drift + sigma * rng.standard_t(df=4, size=n) / np.sqrt(2.0)  # fat tails, unit variance
    close = spec.start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[spec.start_price, close[:-1]]
    wick = np.abs(rng.normal(0, 1, size=(2, n))) * sigma * close * 0.8
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]

    tick = spec.pip_size / 10
    spread = spec.base_spread_pips * spec.pip_size * _spread_mult(hour_utc, ny_mod)
    spread *= np.exp(rng.normal(0, 0.15, size=n))
    spread = np.maximum(np.round(spread / tick) * tick, tick)

    def q(x: np.ndarray) -> np.ndarray:
        return np.round(x / tick) * tick

    bid_o, bid_h, bid_l, bid_c = q(open_), q(high), q(low), q(close)
    df = frame.with_columns(
        pl.Series("bid_o", bid_o),
        pl.Series("bid_h", bid_h),
        pl.Series("bid_l", bid_l),
        pl.Series("bid_c", bid_c),
        pl.Series("ask_o", q(bid_o + spread)),
        pl.Series("ask_h", q(bid_h + spread)),
        pl.Series("ask_l", q(bid_l + spread)),
        pl.Series("ask_c", q(bid_c + spread)),
        pl.Series(
            "volume", np.maximum(1.0, np.round(rng.gamma(2.0, 20.0, size=n) * _session_vol_mult(hour_utc)))
        ),
    )
    return validate_frame(df)


def inject_faults(df: pl.DataFrame, seed: int = 7) -> tuple[pl.DataFrame, list[InjectedFault]]:
    """Inject one fault of every kind at well-separated in-session locations."""
    rng = np.random.default_rng(seed)
    n = df.height
    if n < 2000:
        raise ValueError("need at least 2000 bars to inject faults")
    slots = np.sort(rng.choice(np.arange(200, n - 200, 250), size=6, replace=False))
    t = df["open_time"]
    cols = {c: df[c].to_numpy().copy() for c in df.columns if c != "open_time"}
    faults: list[InjectedFault] = []
    drop = np.zeros(n, dtype=bool)
    dup_at: int | None = None

    # 1. ask below bid
    i = int(slots[0])
    cols["ask_c"][i] = cols["bid_c"][i] - 0.05
    cols["ask_l"][i] = min(cols["ask_l"][i], cols["ask_c"][i])
    faults.append(InjectedFault("ask_below_bid", t[i], t[i]))
    # 2. zero spread
    i = int(slots[1])
    for side in ("o", "h", "l", "c"):
        cols[f"ask_{side}"][i] = cols[f"bid_{side}"][i]
    faults.append(InjectedFault("zero_spread", t[i], t[i]))
    # 3. reverting spike: high jumps 50x typical range for one bar
    i = int(slots[2])
    rng_typ = float(np.median(cols["bid_h"][i - 50 : i] - cols["bid_l"][i - 50 : i]))
    for side in ("bid", "ask"):
        cols[f"{side}_h"][i] += 50 * rng_typ
    faults.append(InjectedFault("spike", t[i], t[i]))
    # 4. stale quotes: 15 bars frozen at the previous close
    i = int(slots[3])
    for k in range(i, i + 15):
        for side in ("bid", "ask"):
            v = cols[f"{side}_c"][i - 1]
            for f in ("o", "h", "l", "c"):
                cols[f"{side}_{f}"][k] = v
    faults.append(InjectedFault("stale", t[i], t[i + 14]))
    # 5. gap: drop 45 bars
    i = int(slots[4])
    drop[i : i + 45] = True
    faults.append(InjectedFault("gap", t[i], t[i + 44]))
    # 6. duplicate timestamp
    dup_at = int(slots[5])
    faults.append(InjectedFault("duplicate", t[dup_at], t[dup_at]))

    out = pl.DataFrame({"open_time": t, **cols}).filter(~pl.Series(drop))
    dup_row = out.filter(pl.col("open_time") == t[dup_at])
    out = pl.concat([out, dup_row]).sort("open_time", maintain_order=True)
    return validate_frame(out), faults


def synthetic_instrument(symbol: str = "SYNTH", pip_size: float = 0.01) -> Instrument:
    """Instrument spec matching `SyntheticSpec` defaults (USD quoted, 1000 units per lot)."""
    return Instrument(
        symbol=symbol,
        asset_class="fx",
        base="SYN",
        quote="USD",
        contract_size=Decimal(1000),
        pip_size=Decimal(str(pip_size)),
        tick_size=Decimal(str(pip_size / 10)),
        min_lot=Decimal("0.01"),
        lot_step=Decimal("0.01"),
        max_lot=Decimal(100),
        commission_per_lot=Decimal("0.7"),
        swap_long=Decimal(-2),
        swap_short=Decimal("-0.5"),
        swap_mode="points",
        triple_swap_weekday=2,
    )
