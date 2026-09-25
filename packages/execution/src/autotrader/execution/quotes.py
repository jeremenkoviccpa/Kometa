"""Latest quote per symbol, with age checks: a dropped feed must block market entries, not price them."""

from __future__ import annotations

from datetime import datetime

from autotrader.core.broker import Quote
from autotrader.core.timeutil import ensure_utc


class QuoteBook:
    def __init__(self) -> None:
        self._last: dict[str, Quote] = {}

    def update(self, q: Quote) -> None:
        prev = self._last.get(q.symbol)
        if prev is None or q.time >= prev.time:  # never step back to an older quote
            self._last[q.symbol] = q

    def get(self, symbol: str) -> Quote | None:
        return self._last.get(symbol)

    def fresh(self, symbol: str, now: datetime, max_age_s: float) -> Quote | None:
        q = self._last.get(symbol)
        if q is None or (ensure_utc(now) - q.time).total_seconds() > max_age_s:
            return None
        return q
