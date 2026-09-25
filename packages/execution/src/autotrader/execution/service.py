"""The execution service: startup checks, then the periodic jobs around the order manager.

Startup refuses (fail closed) when:
- clock skew against the broker server time exceeds `max_clock_skew_seconds` (spec section 18);
- the account's trade mode does not fit the environment: live needs a real account, every other
  environment (dev, ci, paper) needs a demo account, so a misconfigured paper run cannot trade money;
- the account is not a hedging account (the order manager keeps one position per order);
- the broker account id differs from the configured one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from autotrader.core.alerts import Severity
from autotrader.core.broker import AccountInfo
from autotrader.core.clock import Clock
from autotrader.execution.adapter import BrokerAdapter, BrokerUnavailableError
from autotrader.execution.order_manager import OrderManager
from autotrader.execution.reconcile import Reconciler, ReconReport
from autotrader.execution.watchdog import Watchdog

log = logging.getLogger(__name__)


class StartupRefusedError(Exception):
    pass


async def startup_checks(
    adapter: BrokerAdapter,
    clock: Clock,
    *,
    env: str,
    expected_account_id: str | None,
    max_skew_seconds: float,
) -> AccountInfo:
    acct = await adapter.account()
    skew = abs((acct.server_time - clock.now()).total_seconds())
    if skew > max_skew_seconds:
        raise StartupRefusedError(f"clock skew {skew:.1f}s against broker server time")
    wanted = "real" if env == "live" else "demo"
    if acct.trade_mode != wanted:
        raise StartupRefusedError(f"env {env} needs a {wanted} account, broker reports {acct.trade_mode}")
    if acct.margin_mode != "hedging":
        raise StartupRefusedError(f"account margin mode {acct.margin_mode}; a hedging account is required")
    if expected_account_id is not None and acct.account_id != expected_account_id:
        raise StartupRefusedError(f"broker account {acct.account_id} is not the configured account")
    return acct


@dataclass
class CycleResult:
    recon: ReconReport | None
    lost_services: list[str]
    expired: int


class ExecutionService:
    def __init__(self, om: OrderManager, recon: Reconciler, watchdog: Watchdog) -> None:
        self.om = om
        self.recon = recon
        self.watchdog = watchdog
        self._last_recon: datetime | None = None

    async def cycle(self) -> CycleResult:
        """One pass of the periodic jobs. Called every few seconds; reconciliation runs every 60."""
        now = self.om.clock.now()
        expired = 0
        try:
            await self.om.sync_deals()
            expired = await self.om.expire_pending()
            for t in self.om.state.active():
                if t.state == "open" and not t.stop_confirmed:
                    await self.om.confirm_stop(t)
        except BrokerUnavailableError as e:
            log.warning("execution cycle: %s", e)
        lost = await self.watchdog.check()
        report = None
        due = self._last_recon is None or (
            (now - self._last_recon).total_seconds() >= self.om.cfg.reconcile_interval_seconds
        )
        if due:
            report = await self.recon.run_once()
            self._last_recon = now
        return CycleResult(report, lost, expired)

    async def run(
        self, stop: asyncio.Event, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    ) -> None:
        await self.om.initialize()
        while not stop.is_set():
            try:
                await self.cycle()
            except Exception as e:  # never die silently: alert and keep the loop (stops stay at the broker)
                log.exception("execution cycle failed")
                self.om.alert(Severity.CRITICAL, "execution_error", f"cycle failed: {type(e).__name__}")
            await sleep(min(5.0, self.om.cfg.reconcile_interval_seconds))
