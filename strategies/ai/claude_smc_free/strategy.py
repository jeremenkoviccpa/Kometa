"""claude_smc_free: a placeholder, so the track has a registry entry, a stage and a switch in the hub. Its
trades come from the ai-trader service (packages/ai/src/autotrader/ai/trader.py), which publishes signals
under this id while the owner has the track trading. This class never signals.
"""

from __future__ import annotations

from autotrader.core.events import BarClosed
from autotrader.strategies_api import Request, Strategy, StrategyContext


class ClaudeSmcFree(Strategy):
    def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]:
        return []
