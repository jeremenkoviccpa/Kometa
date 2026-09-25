"""Cost model: spread, slippage, commission, swap, currency conversion (spec section 8).

Fill prices are built from the BID series plus a modelled spread, so the same
data can be re-costed with a broker's own spread statistics (and later with
L8-calibrated cost models) without touching the price history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from autotrader.core.models import Instrument
from autotrader.core.series import NS_PER_MINUTE, to_ns

NS_PER_HOUR = 60 * NS_PER_MINUTE
_MONDAY_EPOCH_OFFSET_H = 72  # 1970-01-01 was a Thursday; Monday 00:00 is 72h earlier (mod 168)


def hour_of_week(ns: int) -> int:
    """0 = Monday 00:00 UTC."""
    return int((ns // NS_PER_HOUR + _MONDAY_EPOCH_OFFSET_H) % 168)


def hour_of_week_array(ns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return (ns // NS_PER_HOUR + _MONDAY_EPOCH_OFFSET_H) % 168


@dataclass(frozen=True)
class InstrumentCosts:
    """Float view of an Instrument for hot loops."""

    symbol: str
    quote: str
    contract_size: float
    tick_size: float
    min_lot: float
    lot_step: float
    max_lot: float
    commission_per_lot_side: float  # account currency
    swap_long: float
    swap_short: float
    swap_mode: str
    triple_swap_weekday: int

    @staticmethod
    def from_instrument(i: Instrument) -> InstrumentCosts:
        return InstrumentCosts(
            symbol=i.symbol,
            quote=i.quote,
            contract_size=float(i.contract_size),
            tick_size=float(i.tick_size),
            min_lot=float(i.min_lot),
            lot_step=float(i.lot_step),
            max_lot=float(i.max_lot),
            commission_per_lot_side=float(i.commission_per_lot) / 2.0,
            swap_long=float(i.swap_long),
            swap_short=float(i.swap_short),
            swap_mode=i.swap_mode,
            triple_swap_weekday=i.triple_swap_weekday,
        )


class StaticRates:
    """Fixed conversion rates. Missing pairs raise: never guess a rate.

    SPEC-QUESTION: switch to historical rates from the conversion pairs' own bars.
    """

    def __init__(self, rates: Mapping[tuple[str, str], float] | None = None) -> None:
        self._r = dict(rates or {})

    def rate(self, frm: str, to: str) -> float:
        if frm == to:
            return 1.0
        if (frm, to) in self._r:
            return self._r[(frm, to)]
        if (to, frm) in self._r:
            return 1.0 / self._r[(to, frm)]
        raise KeyError(f"no conversion rate {frm}->{to}")


@dataclass
class SpreadModel:
    """Median spread per hour of week, widened near high-impact news and at rollover."""

    median_by_hour: Mapping[str, npt.NDArray[np.float64]]  # symbol -> 168 floats
    news_ns: Mapping[str, npt.NDArray[np.int64]] = field(default_factory=dict)  # symbol -> sorted event times
    rollover_ns: npt.NDArray[np.int64] = field(default_factory=lambda: np.empty(0, np.int64))
    widen_mult: float = 3.0
    window_ns: int = 2 * NS_PER_MINUTE

    def _near(self, arr: npt.NDArray[np.int64], t: int) -> bool:
        if arr.size == 0:
            return False
        i = int(np.searchsorted(arr, t))
        return (i < arr.size and arr[i] - t <= self.window_ns) or (i > 0 and t - arr[i - 1] <= self.window_ns)

    def median(self, symbol: str, t: int) -> float:
        return float(self.median_by_hour[symbol][hour_of_week(t)])

    def spread(self, symbol: str, t: int) -> float:
        s = self.median(symbol, t)
        news = self.news_ns.get(symbol)
        if (news is not None and self._near(news, t)) or self._near(self.rollover_ns, t):
            s *= self.widen_mult
        return s

    def spread_array(self, symbol: str, t: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        out = self.median_by_hour[symbol][hour_of_week_array(t)].astype(np.float64)
        for arr in (self.news_ns.get(symbol, np.empty(0, np.int64)), self.rollover_ns):
            if arr.size and t.size:
                i = np.clip(np.searchsorted(arr, t), 1, arr.size) - 1
                j = np.clip(np.searchsorted(arr, t), 0, arr.size - 1)
                near = (np.abs(arr[i] - t) <= self.window_ns) | (np.abs(arr[j] - t) <= self.window_ns)
                out = np.where(near, out * self.widen_mult, out)
        return out


DEFAULT_SLIPPAGE_MULT = 0.2  # times the median spread, always adverse


@dataclass(frozen=True)
class CostModel:
    spreads: SpreadModel
    slippage_mult: float = DEFAULT_SLIPPAGE_MULT
    slippage_mult_by_symbol: Mapping[str, float] = field(default_factory=dict)

    def slippage(self, symbol: str, t: int) -> float:
        return self.slippage_mult_by_symbol.get(symbol, self.slippage_mult) * self.spreads.median(symbol, t)


def swap_per_night(inst: InstrumentCosts, side: str, lots: float, price: float, to_account: float) -> float:
    """Signed swap for one night in account currency (positive = credit)."""
    rate = inst.swap_long if side == "buy" else inst.swap_short
    if inst.swap_mode == "points":
        return rate * inst.tick_size * inst.contract_size * lots * to_account
    if inst.swap_mode == "money":
        return rate * lots
    # percent per year of notional, ACT/360
    return rate / 100.0 / 360.0 * price * inst.contract_size * lots * to_account


def rollover_times(
    start: datetime, end: datetime, tz: str = "America/New_York", at: time = time(17, 0)
) -> list[tuple[int, int]]:
    """(ns, local weekday) for each Mon-Fri rollover in [start, end]."""
    zone = ZoneInfo(tz)
    d = start.astimezone(zone).date() - timedelta(days=1)
    last = end.astimezone(zone).date() + timedelta(days=1)
    out: list[tuple[int, int]] = []
    while d <= last:
        if d.weekday() < 5:
            ts = datetime.combine(d, at, tzinfo=zone)
            if start <= ts <= end:
                out.append((to_ns(ts), d.weekday()))
        d += timedelta(days=1)
    return out


def news_by_symbol(
    symbols_ccys: Mapping[str, Sequence[str]], events_ns_by_ccy: Mapping[str, Sequence[int]]
) -> dict[str, npt.NDArray[np.int64]]:
    out: dict[str, npt.NDArray[np.int64]] = {}
    for sym, ccys in symbols_ccys.items():
        ts = sorted({t for c in ccys for t in events_ns_by_ccy.get(c, ())})
        out[sym] = np.asarray(ts, dtype=np.int64)
    return out
