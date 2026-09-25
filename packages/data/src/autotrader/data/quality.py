"""Data quality checks on M1 bid/ask bars (spec section 7). Flag and store, never fix."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

import numpy as np
import polars as pl

from autotrader.core.alerts import Alert
from autotrader.core.alerts import Severity as AlertSeverity
from autotrader.core.indicators import atr
from autotrader.data.market_hours import MarketHours


class IssueType(StrEnum):
    GAP = "gap"
    ASK_BELOW_BID = "ask_below_bid"
    ZERO_SPREAD = "zero_spread"
    SPIKE = "spike"
    DUPLICATE = "duplicate"
    STALE = "stale"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class QualityIssue:
    symbol: str
    start: datetime
    end: datetime
    issue_type: IssueType
    severity: Severity
    detail: str = ""


@dataclass(frozen=True)
class QualityConfig:
    max_gap_minutes: int = 3  # SPEC-QUESTION: tolerance for broker micro-gaps; 3 chosen as safe default
    spike_atr_mult: float = 8.0
    spike_revert_bars: int = 3
    spike_revert_atr: float = 2.0
    stale_minutes: int = 10
    atr_period: int = 14


def _gap_severity(minutes: float) -> Severity:
    if minutes >= 60:
        return Severity.HIGH
    return Severity.MEDIUM if minutes >= 15 else Severity.LOW


def check_bars(
    df: pl.DataFrame, symbol: str, hours: MarketHours, cfg: QualityConfig | None = None
) -> list[QualityIssue]:
    cfg = cfg or QualityConfig()
    issues: list[QualityIssue] = []
    if df.height == 0:
        return issues
    t = df["open_time"]

    # duplicates
    dup = df.filter(pl.col("open_time").is_duplicated()).select("open_time").unique()
    issues += [
        QualityIssue(symbol, ts, ts, IssueType.DUPLICATE, Severity.HIGH)
        for ts in dup["open_time"].sort().to_list()
    ]
    d = df.unique(subset="open_time", keep="first", maintain_order=True).sort("open_time")
    t = d["open_time"]

    # ask below bid (any field), zero spread (all fields equal)
    below = d.filter(
        (pl.col("ask_o") < pl.col("bid_o"))
        | (pl.col("ask_h") < pl.col("bid_h"))
        | (pl.col("ask_l") < pl.col("bid_l"))
        | (pl.col("ask_c") < pl.col("bid_c"))
    )["open_time"]
    issues += [QualityIssue(symbol, ts, ts, IssueType.ASK_BELOW_BID, Severity.HIGH) for ts in below.to_list()]
    zero = d.filter(pl.col("ask_c") == pl.col("bid_c"))["open_time"]
    issues += [QualityIssue(symbol, ts, ts, IssueType.ZERO_SPREAD, Severity.MEDIUM) for ts in zero.to_list()]

    # gaps: consecutive bars more than max_gap apart, counting only minutes the market was open
    if d.height > 1:
        step = d.select(pl.col("open_time").diff().dt.total_minutes().alias("m"))["m"].to_numpy()
        for gi in np.nonzero(step > cfg.max_gap_minutes)[0].tolist():
            prev_t, next_t = t[gi - 1], t[gi]
            missing = pl.DataFrame(
                {
                    "open_time": pl.datetime_range(
                        prev_t + timedelta(minutes=1),
                        next_t,
                        "1m",
                        closed="left",
                        time_zone="UTC",
                        eager=True,
                    )
                }
            ).filter(hours.is_open_expr())
            if missing.height > cfg.max_gap_minutes:
                issues.append(
                    QualityIssue(
                        symbol,
                        missing["open_time"][0],
                        missing["open_time"][-1],
                        IssueType.GAP,
                        _gap_severity(missing.height),
                        f"{missing.height} open-market minutes missing",
                    )
                )

    # reverting spikes: excursion from previous close > k * ATR, back within m * ATR in N bars
    h, lo, c = d["bid_h"].to_numpy(), d["bid_l"].to_numpy(), d["bid_c"].to_numpy()
    a = atr(h, lo, c, cfg.atr_period)
    if d.height > 1:
        prev, ref = c[:-1], a[:-1]
        excursion = np.maximum(h[1:] - prev, prev - lo[1:])
        with np.errstate(invalid="ignore"):
            candidates = np.flatnonzero((ref > 0) & (excursion > cfg.spike_atr_mult * ref)) + 1
        for i in candidates.tolist():
            p, r = c[i - 1], a[i - 1]
            window = c[i : i + 1 + cfg.spike_revert_bars]
            if np.any(np.abs(window - p) <= cfg.spike_revert_atr * r):
                x = max(h[i] - p, p - lo[i]) / r
                issues.append(
                    QualityIssue(symbol, t[i], t[i], IssueType.SPIKE, Severity.HIGH, f"{x:.1f} x ATR")
                )

    # stale quotes: runs of flat bars equal to the previous close
    flat = (
        (d["bid_o"] == d["bid_c"]) & (d["bid_h"] == d["bid_c"]) & (d["bid_l"] == d["bid_c"])
    ).to_numpy() & np.r_[False, c[1:] == c[:-1]]
    edges = np.flatnonzero(np.diff(np.r_[False, flat, False].astype(np.int8)))
    for s0, s1 in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
        if s1 - s0 >= cfg.stale_minutes:
            issues.append(
                QualityIssue(symbol, t[s0], t[s1 - 1], IssueType.STALE, Severity.MEDIUM, f"{s1 - s0} bars")
            )
    return issues


def high_severity_windows(issues: list[QualityIssue]) -> list[tuple[datetime, datetime]]:
    """Windows that validation must exclude from pass thresholds."""
    return [(i.start, i.end) for i in issues if i.severity == Severity.HIGH]


def quality_alerts(issues: list[QualityIssue], traded: set[str], at: datetime) -> list[Alert]:
    """Spec section 16: a data quality issue on a traded symbol is a warning. One alert per symbol;
    low-severity issues and symbols nobody trades only go to the stored report."""
    by_symbol: dict[str, list[QualityIssue]] = {}
    for i in issues:
        if i.symbol in traded and i.severity != Severity.LOW:
            by_symbol.setdefault(i.symbol, []).append(i)
    out = []
    for symbol, found in sorted(by_symbol.items()):
        counts = Counter(i.issue_type.value for i in found)
        worst = max(found, key=lambda i: (i.severity == Severity.HIGH, i.end))
        out.append(
            Alert(
                severity=AlertSeverity.WARNING,
                kind="data_quality",
                message=f"{symbol}: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())),
                at=at,
                details={"symbol": symbol, "worst": f"{worst.issue_type.value} {worst.start:%Y-%m-%d %H:%M}"},
            )
        )
    return out
