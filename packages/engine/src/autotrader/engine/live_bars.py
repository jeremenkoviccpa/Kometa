"""Live bar builder (spec section 7): quotes -> M1 bid and ask bars -> subscribed timeframes.

- An M1 bar covers [minute, minute + 1m): first/max/min/last bid and ask of its quotes; volume is the
  tick count. A minute without quotes has no bar, exactly like historical data with a gap.
- Higher timeframes are aggregated from closed M1 bars with `core.timeframes.bucket_open_ns`, the
  same alignment the historical resampler uses; a bucket closes at bucket open + timeframe.
- Nothing is emitted before `close_time + grace`, so a quote that arrives slightly late still lands
  in its bar. A quote older than an already emitted bar is dropped and counted (`late_quotes`).
- Everything that closes at the same time is emitted as one batch, ordered like the backtest loop:
  by close time, then symbol order of the subscription, then shorter timeframe first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from autotrader.core.broker import Quote
from autotrader.core.events import BarClosed
from autotrader.core.models import Bar, Timeframe
from autotrader.core.series import NS_PER_MINUTE, from_ns, to_ns
from autotrader.core.timeframes import DEFAULT_DAY_CLOSE, DEFAULT_DAY_TZ, bucket_open_ns


@dataclass
class _Agg:
    open_ns: int
    close_ns: int
    bid_o: float
    bid_h: float
    bid_l: float
    bid_c: float
    ask_o: float
    ask_h: float
    ask_l: float
    ask_c: float
    volume: float

    @classmethod
    def first(cls, open_ns: int, close_ns: int, bid: float, ask: float, volume: float = 1.0) -> _Agg:
        return cls(open_ns, close_ns, bid, bid, bid, bid, ask, ask, ask, ask, volume)

    def add_quote(self, bid: float, ask: float) -> None:
        self.bid_h, self.bid_l, self.bid_c = max(self.bid_h, bid), min(self.bid_l, bid), bid
        self.ask_h, self.ask_l, self.ask_c = max(self.ask_h, ask), min(self.ask_l, ask), ask
        self.volume += 1

    def add_bar(self, b: _Agg) -> None:
        self.bid_h, self.bid_l, self.bid_c = max(self.bid_h, b.bid_h), min(self.bid_l, b.bid_l), b.bid_c
        self.ask_h, self.ask_l, self.ask_c = max(self.ask_h, b.ask_h), min(self.ask_l, b.ask_l), b.ask_c
        self.volume += b.volume

    def bar(self, symbol: str, tf: Timeframe) -> Bar:
        return Bar(
            symbol=symbol,
            timeframe=tf,
            open_time=from_ns(self.open_ns),
            close_time=from_ns(self.close_ns),
            bid_o=self.bid_o,
            bid_h=self.bid_h,
            bid_l=self.bid_l,
            bid_c=self.bid_c,
            ask_o=self.ask_o,
            ask_h=self.ask_h,
            ask_l=self.ask_l,
            ask_c=self.ask_c,
            volume=self.volume,
        )


class LiveBarBuilder:
    def __init__(
        self,
        subscriptions: Iterable[tuple[str, Timeframe]],
        *,
        grace: timedelta = timedelta(seconds=1),
        day_close: str = DEFAULT_DAY_CLOSE,
        tz: str = DEFAULT_DAY_TZ,
        on_m1: Callable[[Bar], None] | None = None,
    ) -> None:
        subs = list(dict.fromkeys(subscriptions))
        self.order = {k: i for i, k in enumerate(subs)}
        self.symbols = list(dict.fromkeys(s for s, _ in subs))
        self.htf = {s: [tf for (x, tf) in subs if x == s and tf != Timeframe.M1] for s in self.symbols}
        self.grace_ns = int(grace.total_seconds() * 1e9)
        self.day_close, self.tz = day_close, tz
        self.on_m1 = on_m1
        self._building: dict[str, _Agg] = {}
        self._done_m1: dict[str, list[_Agg]] = {s: [] for s in self.symbols}
        self._htf: dict[tuple[str, Timeframe], _Agg] = {}
        self._emitted_until: dict[str, int] = dict.fromkeys(self.symbols, 0)  # close ns of last M1 out
        self.late_quotes = 0
        self._next_due = 1 << 62  # earliest close_ns of anything not yet emitted (fast path)

    def on_quote(self, q: Quote) -> list[BarClosed]:
        """Close whatever is due at this quote's time, then add the quote."""
        if q.symbol not in self._done_m1:
            return self.on_time(q.time)
        t = to_ns(q.time)
        out = self._on_time_ns(t)
        minute = t - t % NS_PER_MINUTE
        if minute + NS_PER_MINUTE <= self._emitted_until[q.symbol]:
            self.late_quotes += 1
            return out
        bid, ask = float(q.bid), float(q.ask)
        cur = self._building.get(q.symbol)
        if cur is not None and cur.open_ns == minute:
            cur.add_quote(bid, ask)
        elif cur is not None and minute < cur.open_ns:
            self.late_quotes += 1  # older than the bar being built for this symbol
        else:
            if cur is not None:
                self._done_m1[q.symbol].append(cur)
            self._building[q.symbol] = _Agg.first(minute, minute + NS_PER_MINUTE, bid, ask)
            self._next_due = min(self._next_due, minute + NS_PER_MINUTE)
        return out

    def on_time(self, now: datetime) -> list[BarClosed]:
        return self._on_time_ns(to_ns(now))

    def _on_time_ns(self, now_ns: int) -> list[BarClosed]:
        cutoff = now_ns - self.grace_ns
        if cutoff < self._next_due:
            return []
        events: list[tuple[int, int, BarClosed]] = []
        for s in self.symbols:
            cur = self._building.get(s)
            if cur is not None and cur.close_ns <= cutoff:
                self._done_m1[s].append(cur)
                del self._building[s]
            due = [b for b in self._done_m1[s] if b.close_ns <= cutoff]
            self._done_m1[s] = [b for b in self._done_m1[s] if b.close_ns > cutoff]
            for b in due:
                self._emitted_until[s] = b.close_ns
                bar = b.bar(s, Timeframe.M1)
                if self.on_m1 is not None:
                    self.on_m1(bar)
                if (s, Timeframe.M1) in self.order:
                    events.append(self._event(s, Timeframe.M1, b))
                for tf in self.htf[s]:
                    self._add_to_bucket(s, tf, b, events)
            for tf in self.htf[s]:
                agg = self._htf.get((s, tf))
                if agg is not None and agg.close_ns <= cutoff:
                    events.append(self._event(s, tf, agg))
                    del self._htf[(s, tf)]
        pending = [b.close_ns for b in self._building.values()]
        pending += [b.close_ns for bs in self._done_m1.values() for b in bs]
        pending += [a.close_ns for a in self._htf.values()]
        self._next_due = min(pending, default=1 << 62)
        events.sort(key=lambda e: (e[0], e[1]))
        return [e for _, _, e in events]

    def _add_to_bucket(
        self, s: str, tf: Timeframe, b: _Agg, events: list[tuple[int, int, BarClosed]]
    ) -> None:
        agg = self._htf.get((s, tf))
        if agg is not None and agg.open_ns <= b.open_ns < agg.close_ns:
            agg.add_bar(b)  # buckets are contiguous: still inside the current one
            return
        start = bucket_open_ns(b.open_ns, tf, day_close=self.day_close, tz=self.tz)
        if agg is not None and agg.open_ns != start:
            events.append(self._event(s, tf, agg))  # the previous bucket ended before this bar
            agg = None
        if agg is None:
            self._htf[(s, tf)] = _Agg(
                start,
                start + tf.minutes * NS_PER_MINUTE,
                b.bid_o,
                b.bid_h,
                b.bid_l,
                b.bid_c,
                b.ask_o,
                b.ask_h,
                b.ask_l,
                b.ask_c,
                b.volume,
            )
        else:
            agg.add_bar(b)

    def _event(self, s: str, tf: Timeframe, agg: _Agg) -> tuple[int, int, BarClosed]:
        bar = agg.bar(s, tf)
        return (
            agg.close_ns,
            self.order[(s, tf)],
            BarClosed(at=bar.close_time, symbol=s, timeframe=tf, bar=bar),
        )
