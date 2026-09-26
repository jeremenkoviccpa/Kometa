"""Live strategy runner: the same strategy code, MarketView and request checks as the backtest.

Only the clock and the data feed differ (spec section 1). Bars arrive as `BarClosed` batches from the
LiveBarBuilder; a batch is applied exactly like one backtest time step: every bar that closes at T is
appended first, `now` becomes T, then `on_bar` runs for each in subscription order. Signal ids depend
only on strategy, version, T and call order, so live and backtest produce identical ids.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import groupby

from autotrader.core.events import BarClosed
from autotrader.core.models import Bar, Timeframe
from autotrader.core.series import BarsArray, to_ns
from autotrader.engine.context import EngineContext
from autotrader.engine.market import EngineMarketView, NewsFn, SeriesBuffer
from autotrader.engine.requests import StrategyError, check_requests
from autotrader.strategies_api.base import FillView, PendingView, PositionView, Request, Strategy
from autotrader.strategies_api.manifest import ParamValue


def _row(bar: Bar) -> dict[str, float | int]:
    return {
        "open_time": to_ns(bar.open_time),
        "close_time": to_ns(bar.close_time),
        "bid_o": bar.bid_o,
        "bid_h": bar.bid_h,
        "bid_l": bar.bid_l,
        "bid_c": bar.bid_c,
        "ask_o": bar.ask_o,
        "ask_h": bar.ask_h,
        "ask_l": bar.ask_l,
        "ask_c": bar.ask_c,
        "volume": bar.volume,
    }


class LiveRunner:
    def __init__(
        self,
        strategy_cls: type[Strategy],
        *,
        spread_fn: Callable[[str, int], float],
        positions_fn: Callable[[str, str | None], list[PositionView]],
        pending_fn: Callable[[str, str | None], list[PendingView]],
        params: Mapping[str, ParamValue] | None = None,
        history: Mapping[tuple[str, Timeframe], BarsArray] | None = None,
        seed: int = 0,
        news_fn: NewsFn | None = None,
    ) -> None:
        self.manifest = strategy_cls.manifest
        self.params = self.manifest.param_values(dict(params or {}))
        tfs = sorted(self.manifest.timeframes, key=lambda t: t.minutes)
        self.subs = [(s, tf) for s in self.manifest.symbols for tf in tfs]
        self.order = {k: i for i, k in enumerate(self.subs)}
        self.market = EngineMarketView(spread_fn, news_fn)
        for key in self.subs:
            buf = SeriesBuffer()
            hist = (history or {}).get(key)
            if hist is not None:
                for i in range(len(hist)):
                    buf.append({k: getattr(hist, k)[i].item() for k in _row_keys()})
                if len(hist):
                    self.market.set_now(int(hist.close_time[-1]))
            self.market.add_series(*key, buf)
        self.strategy = strategy_cls()
        self.ctx = EngineContext(
            self.manifest.id,
            self.manifest.version,
            self.market,
            self.params,
            positions_fn,
            pending_fn,
            seed=seed,
        )
        self.warmup = self.strategy.warmup()
        self._seen: set[str] = set()

    def warmed(self) -> bool:
        return all(self.market.series(s, tf).visible >= n for (s, tf), n in self.warmup.items())

    def on_bars(self, events: Sequence[BarClosed]) -> list[Request]:
        """Apply closed bars; returns the strategy's checked requests in call order."""
        out: list[Request] = []
        mine = [e for e in events if (e.symbol, e.timeframe) in self.order]
        mine.sort(key=lambda e: (to_ns(e.at), self.order[(e.symbol, e.timeframe)]))
        for _, group_it in groupby(mine, key=lambda e: to_ns(e.at)):
            group = list(group_it)
            for e in group:
                self.market.series(e.symbol, e.timeframe).append(_row(e.bar))
            self.market.set_now(to_ns(group[0].at))
            if not self.warmed():
                continue
            for e in group:
                out += check_requests(
                    self.strategy.on_bar(self.ctx, e), self.manifest, self.market.now, self._seen
                )
        return out

    def on_fill(self, fill: FillView) -> list[Request]:
        return list(
            check_requests(self.strategy.on_fill(self.ctx, fill), self.manifest, self.market.now, self._seen)
        )


def _row_keys() -> tuple[str, ...]:
    return (
        "open_time",
        "close_time",
        "bid_o",
        "bid_h",
        "bid_l",
        "bid_c",
        "ask_o",
        "ask_h",
        "ask_l",
        "ask_c",
        "volume",
    )


__all__ = ["LiveRunner", "StrategyError"]
