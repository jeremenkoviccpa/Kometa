"""The owner's Claude tracks: Claude is asked only while a track is switched on and within its budget, and
whatever it answers must pass the method's hard rules in code before a signal exists. Claude is faked here;
every refusal has an accepted control (the same answer with the one thing changed)."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from autotrader.ai.decision import CHECKLIST, AiDecision, check
from autotrader.ai.trader import FREE_ID, JUDGE_ID, AiConfig, AiTraderService
from autotrader.core.broker import Quote
from autotrader.core.bus import SIGNALS, InMemoryBus
from autotrader.core.clock import SimClock
from autotrader.core.events import AccountUpdate, Exposure, SignalEmitted
from autotrader.core.models import Signal, Stage
from autotrader.strategies_api.loader import load_strategy

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
ALL_OK = dict.fromkeys(CHECKLIST, True)


def take(side: str = "buy", stop: float = 2990.0, target: float = 3040.0, **kw: Any) -> dict[str, Any]:
    return {
        "action": "take",
        "side": side,
        "stop": stop,
        "target": target,
        "confidence": 0.7,
        "reasoning": "bullish D1/H4, sweep of 2995 lows at the H1 OB, CHOCH, FVG retest, engulfing",
        "checklist": ALL_OK,
        **kw,
    }


# ---------------------------------------------------------------- the hard rules


def test_a_sound_trade_passes_at_the_live_price() -> None:
    res = check(AiDecision(**take()), 2999.8, 3000.0, min_rr=3.0, max_stop=15.0)
    assert not isinstance(res, str)
    assert (res.side, res.entry, res.stop, res.target) == ("buy", 3000.0, 2990.0, 3040.0)
    assert res.rr == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("answer", "kw", "why"),
    [
        (take(stop=3001.0), {}, "wrong side of the price"),
        (take(target=2999.0), {}, "target is on the wrong side"),
        (take(target=3025.0), {}, "below 3"),  # 2.5R at the ask
        (take(stop=2980.0, target=3100.0), {}, "too wide"),  # 20 > 15
        (take(checklist={**ALL_OK, "choch_mss": False}), {}, "checklist is incomplete"),
        (take(side="sell", stop=3010.0, target=2960.0), {"side": "buy"}, "turn it around"),
        (take(stop=2994.0), {"sweep": 2993.0}, "beyond the liquidity sweep"),
        ({**take(), "stop": None}, {}, "needs side, stop and target"),
    ],
    ids=["stop-side", "target-side", "rr", "wide-stop", "checklist", "turned", "inside-sweep", "no-stop"],
)
def test_the_hard_rules_refuse(answer: dict[str, Any], kw: dict[str, Any], why: str) -> None:
    res = check(AiDecision(**answer), 2999.8, 3000.0, min_rr=3.0, max_stop=15.0, **kw)
    assert isinstance(res, str) and why in res


def test_a_short_uses_the_bid_and_mirrors_every_rule() -> None:
    res = check(
        AiDecision(**take("sell", 3010.0, 2960.0)), 3000.0, 3000.3, min_rr=3.0, max_stop=15.0, sweep=3008.0
    )
    assert not isinstance(res, str) and res.entry == 3000.0 and res.rr == pytest.approx(4.0)
    bad = check(
        AiDecision(**take("sell", 3007.0, 2960.0)), 3000.0, 3000.3, min_rr=3.0, max_stop=15.0, sweep=3008.0
    )
    assert isinstance(bad, str) and "sweep" in bad


# ---------------------------------------------------------------- the service


class FakeClaude:
    def __init__(self, *answers: dict[str, Any] | Exception) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[str, str]] = []

    async def __call__(self, system: str, context: str) -> dict[str, Any]:
        self.asked.append((system, context))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


class Wall:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def service(
    claude: FakeClaude | None, *, on: tuple[str, ...] = (JUDGE_ID, FREE_ID), env: str = "paper", **cfg: Any
) -> tuple[AiTraderService, InMemoryBus, Wall]:
    bus, wall = InMemoryBus(), Wall()
    rules = load_strategy(ROOT / "strategies" / "library" / "smc_sniper").cls
    s = AiTraderService(
        bus,
        SimClock(T0),
        decider=claude,
        versions={JUDGE_ID: "1.0.0", FREE_ID: "1.0.0"},
        rules=rules,
        config=AiConfig(model="claude-sonnet-5", **cfg),
        env=env,
        wall=wall,
    )
    s.stages = {(t, "1.0.0"): Stage.DEMO_ONLY if t in on else Stage.SHADOW for t in (JUDGE_ID, FREE_ID)}
    s.quotes["XAUUSD"] = Quote(symbol="XAUUSD", bid=Decimal("2999.8"), ask=Decimal("3000.0"), time=T0)
    s.limits = lambda _sym: (3.0, 15.0)  # type: ignore[method-assign,assignment]
    return s, bus, wall


def candidate(side: str = "buy") -> Signal:
    return Signal(
        signal_id=uuid.uuid5(uuid.NAMESPACE_URL, "candidate"),
        strategy_id="smc_sniper",
        strategy_version="1.0.1",
        symbol="XAUUSD",
        side=side,
        entry_type="market",
        entry_price=None,
        stop_price=2992.0,
        target_price=3035.0,
        created_at=T0,
        reason="SMC bullish bias ...",
        tags={"setup": "smc_sniper", "sweep": "2993.00000"},
    )


async def settle(s: AiTraderService) -> None:
    await s.settle()


async def signals(bus: InMemoryBus) -> list[Signal]:
    return [m.signal for _, m in await bus.read(SIGNALS, "t", "t-1") if isinstance(m, SignalEmitted)]


async def test_the_judge_takes_a_setup_under_its_own_id() -> None:
    claude = FakeClaude(take())
    s, bus, _ = service(claude)
    s.ask(JUDGE_ID, "XAUUSD", candidate())
    await settle(s)
    [sig] = await signals(bus)
    assert (sig.strategy_id, sig.side, sig.stop_price, sig.target_price) == (JUDGE_ID, "buy", 2990.0, 3040.0)
    assert sig.reason.startswith("Claude (4.0R") and sig.tags["ai"] == "claude-sonnet-5"
    assert s.decisions[-1]["action"] == "take"
    system, context = claude.asked[0]
    assert "JUDGE" in system and "SNIPER ENTRY" in system and "sweep -> rejection" in system
    assert json.loads(context)["candidate"]["tags"]["sweep"] == "2993.00000"


@pytest.mark.parametrize(
    ("answer", "action"),
    [
        ({**take(), "action": "skip"}, "skip"),
        (take(stop=2994.0), "refused"),  # inside the sweep at 2993
        (take(side="sell", stop=3010.0, target=2960.0), "refused"),  # turned around
        (take(target=3020.0), "refused"),  # 2R
        (ValueError("overloaded"), "error"),
    ],
    ids=["skip", "inside-sweep", "turned", "low-rr", "api-error"],
)
async def test_no_signal_unless_claude_takes_and_the_rules_agree(answer: Any, action: str) -> None:
    s, bus, _ = service(FakeClaude(answer))
    s.ask(JUDGE_ID, "XAUUSD", candidate())
    await settle(s)
    assert await signals(bus) == []
    assert s.decisions[-1]["action"] == action
    assert not s.busy  # the next setup can be asked about


async def test_a_track_that_is_off_never_calls_claude() -> None:
    claude = FakeClaude(take())
    s, _, _ = service(claude, on=(FREE_ID,))
    s.ask(JUDGE_ID, "XAUUSD", candidate())
    assert claude.asked == [] and not s.tasks
    s.ask(FREE_ID, "XAUUSD", None)  # control: the track that is on
    await settle(s)
    assert len(claude.asked) == 1


async def test_switched_off_while_thinking_means_no_trade() -> None:
    s, bus, _ = service(FakeClaude(take()))
    s.ask(FREE_ID, "XAUUSD", None)
    s.stages[(FREE_ID, "1.0.0")] = Stage.SHADOW
    await settle(s)
    assert await signals(bus) == [] and "switched off" in s.decisions[-1]["detail"]


async def test_the_free_track_reads_at_most_every_interval_and_all_within_the_daily_budget() -> None:
    claude = FakeClaude(*[{**take(), "action": "skip"}] * 4)
    s, _, wall = service(claude, free_every_s=900.0, max_calls_per_day=3)
    s.ask(FREE_ID, "XAUUSD", None)
    await settle(s)
    wall.t = 100.0
    assert s.blocked(FREE_ID, "XAUUSD") == "waiting for the next read"
    wall.t = 1000.0
    assert s.blocked(FREE_ID, "XAUUSD") is None
    s.ask(FREE_ID, "XAUUSD", None)
    s.ask(JUDGE_ID, "XAUUSD", candidate())
    await settle(s)
    assert len(claude.asked) == 3
    wall.t = 5000.0
    assert "daily budget" in (s.blocked(JUDGE_ID, "XAUUSD") or "")
    wall.t = 1000.0 + 86_401  # a day after the first call
    assert s.blocked(JUDGE_ID, "XAUUSD") is None


async def test_one_position_at_a_time_per_track() -> None:
    s, _, _ = service(FakeClaude())
    s.account = AccountUpdate(
        at=T0,
        account_id="a",
        currency="USD",
        balance=Decimal(50000),
        equity=Decimal(50000),
        free_margin=Decimal(50000),
        exposures=(
            Exposure(
                symbol="XAUUSD",
                side="buy",
                lots=Decimal("0.1"),
                entry=Decimal(3000),
                stop=Decimal(2990),
                strategy_id=JUDGE_ID,
                strategy_version="1.0.0",
            ),
        ),
        quotes=(),
        margin_per_lot={},
    )
    assert s.blocked(JUDGE_ID, "XAUUSD") == "a position is open"
    assert s.blocked(FREE_ID, "XAUUSD") is None  # control: the other track has none


async def test_never_with_real_money_and_never_without_a_key() -> None:
    live, _, _ = service(FakeClaude(), env="live")
    assert "real money" in (live.blocked(FREE_ID, "XAUUSD") or "")
    keyless, _, _ = service(None)
    assert "API key" in (keyless.blocked(FREE_ID, "XAUUSD") or "")
    paper, _, _ = service(FakeClaude())  # control
    assert paper.blocked(FREE_ID, "XAUUSD") is None


async def test_a_late_answer_is_not_traded() -> None:
    s, bus, _ = service(FakeClaude())
    await s.act(FREE_ID, "XAUUSD", None, AiDecision(**take()), T0 - timedelta(minutes=16))
    assert await signals(bus) == [] and "too late" in s.decisions[-1]["detail"]
    await s.act(FREE_ID, "XAUUSD", None, AiDecision(**take()), T0 - timedelta(minutes=5))  # control
    assert len(await signals(bus)) == 1


def test_the_hub_view_says_what_is_on_and_why_claude_is_idle() -> None:
    s, _, _ = service(None, on=(JUDGE_ID,))
    v = s.view()
    assert v["tracks"] == {JUDGE_ID: True, FREE_ID: False}
    assert not v["enabled"] and "API key" in v["disabled_reason"]


async def test_settle_waits_for_claude_so_a_held_clock_sees_the_answer() -> None:
    gate = asyncio.Event()

    class Slow(FakeClaude):
        async def __call__(self, system: str, context: str) -> dict[str, Any]:
            await gate.wait()
            return await super().__call__(system, context)

    s, bus, _ = service(Slow(take()))
    s.ask(FREE_ID, "XAUUSD", None)
    assert await signals(bus) == [] and s.busy  # still thinking
    asyncio.get_running_loop().call_later(0.01, gate.set)
    await s.settle()
    assert len(await signals(bus)) == 1 and not s.busy
