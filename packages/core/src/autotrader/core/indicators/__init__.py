"""Indicator and pattern library (spec section 6).

Every indicator exists as a vectorized function (research, backtests) and an
incremental class with `update()` (live). tests/unit/test_indicator_parity.py
proves both agree. Output at index i depends only on inputs [0, i].
"""

from autotrader.core.indicators.bands import Bollinger, Donchian, Keltner, bollinger, donchian, keltner
from autotrader.core.indicators.candles import (
    CandleTracker,
    Reversals,
    ReversalTracker,
    anatomy,
    patterns,
    reversals,
)
from autotrader.core.indicators.candlestick import (
    PATTERNS,
    CandlestickTracker,
    candlestick_patterns,
    last_bar_patterns,
)
from autotrader.core.indicators.geometry import (
    Convergence,
    Line,
    breakout,
    convergence,
    fit_line,
    last_points,
)
from autotrader.core.indicators.levels import Level, LevelTracker, levels
from autotrader.core.indicators.ma import EMA, SMA, WMA, Slope, ema, slope, sma, wma
from autotrader.core.indicators.oscillators import MACD, RSI, Stochastic, macd, rsi, stochastic
from autotrader.core.indicators.sessions import DEFAULT_SESSIONS, EventIndex, in_session, sessions_of
from autotrader.core.indicators.structure import SwingDetector, TrendState, swings, trend_state
from autotrader.core.indicators.volatility import (
    ATR,
    ATRPercentile,
    PercentileRank,
    RollingStd,
    TrueRange,
    Wilder,
    atr,
    atr_percentile,
    percentile_rank,
    rolling_std,
    true_range,
    wilder,
)

__all__ = [
    "ATR",
    "DEFAULT_SESSIONS",
    "EMA",
    "MACD",
    "PATTERNS",
    "RSI",
    "SMA",
    "WMA",
    "ATRPercentile",
    "Bollinger",
    "CandleTracker",
    "CandlestickTracker",
    "Convergence",
    "Donchian",
    "EventIndex",
    "Keltner",
    "Level",
    "LevelTracker",
    "Line",
    "PercentileRank",
    "ReversalTracker",
    "Reversals",
    "RollingStd",
    "Slope",
    "Stochastic",
    "SwingDetector",
    "TrendState",
    "TrueRange",
    "Wilder",
    "anatomy",
    "atr",
    "atr_percentile",
    "bollinger",
    "breakout",
    "candlestick_patterns",
    "convergence",
    "donchian",
    "ema",
    "fit_line",
    "in_session",
    "keltner",
    "last_bar_patterns",
    "last_points",
    "levels",
    "macd",
    "patterns",
    "percentile_rank",
    "reversals",
    "rolling_std",
    "rsi",
    "sessions_of",
    "slope",
    "sma",
    "stochastic",
    "swings",
    "trend_state",
    "true_range",
    "wilder",
    "wma",
]
