"""Currency conversion from live quotes. A missing pair raises: a rate is never guessed."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from autotrader.core.broker import Quote


def rate_from_quotes(quotes: Mapping[str, Quote], frm: str, to: str) -> Decimal:
    """Mid rate converting an amount in `frm` into `to`, from the FRMTO or TOFRM quote."""
    if frm == to:
        return Decimal(1)
    q = quotes.get(frm + to)
    if q is not None:
        return (q.bid + q.ask) / 2
    q = quotes.get(to + frm)
    if q is not None:
        return 2 / (q.bid + q.ask)
    raise KeyError(f"no quote to convert {frm} -> {to}")
