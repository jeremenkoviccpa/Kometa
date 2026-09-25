"""Geometry building blocks for chart patterns (spec section 6).

Flags, triangles and double tops are strategies and are NOT built here; these
helpers give them lines through swing points, convergence and breakouts.
x is a bar index, y a price. All functions are pure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autotrader.core.indicators._base import as_f64


@dataclass(frozen=True)
class Line:
    slope: float
    intercept: float
    r2: float  # goodness of fit, 1.0 for two points
    n: int

    def at(self, x: float) -> float:
        return self.slope * x + self.intercept


def fit_line(xs: object, ys: object) -> Line:
    """Least-squares line through points (e.g. the last swing highs)."""
    x, y = as_f64(xs), as_f64(ys)
    if x.size != y.size or x.size < 2:
        raise ValueError("need at least two points of equal length")
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx == 0.0:
        raise ValueError("points share one x; line is vertical")
    slope = float(((x - xm) * (y - ym)).sum()) / sxx
    intercept = float(ym - slope * xm)
    resid = y - (slope * x + intercept)
    sst = float(((y - ym) ** 2).sum())
    r2 = 1.0 if sst == 0.0 else 1.0 - float((resid**2).sum()) / sst
    return Line(slope, intercept, r2, int(x.size))


@dataclass(frozen=True)
class Convergence:
    gap_start: float  # upper - lower at x_start
    gap_end: float  # upper - lower at x_end
    converging: bool
    apex_x: float | None  # where the lines meet; None if parallel


def convergence(upper: Line, lower: Line, x_start: float, x_end: float) -> Convergence:
    g0 = upper.at(x_start) - lower.at(x_start)
    g1 = upper.at(x_end) - lower.at(x_end)
    ds = upper.slope - lower.slope
    apex = None if ds == 0.0 else (lower.intercept - upper.intercept) / ds
    return Convergence(g0, g1, bool(abs(g1) < abs(g0)), apex)


def breakout(close: float, line: Line, x: float, min_distance: float = 0.0) -> int:
    """+1 if the close is above the line by more than min_distance, -1 if below, else 0."""
    v = line.at(x)
    if close > v + min_distance:
        return 1
    if close < v - min_distance:
        return -1
    return 0


def last_points(values: object, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the last n finite entries (e.g. of a swings array)."""
    a = as_f64(values)
    idx = np.flatnonzero(~np.isnan(a))[-n:]
    return idx.astype(np.float64), a[idx]
