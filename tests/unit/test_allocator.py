"""Allocator (spec section 13): weights, clusters, caps, weekly rebalance, proposals; property tests."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from autotrader.allocator.allocator import Allocator
from autotrader.allocator.config import load_allocator_config
from autotrader.allocator.weights import VersionPerformance, capped_shares, clusters, shrunk_weight
from autotrader.core.models import Signal, Stage
from autotrader.core.timeutil import utc
from autotrader.data.instruments import load_instruments

ROOT = Path(__file__).resolve().parents[2]
CFG = load_allocator_config(ROOT / "config" / "allocator.yaml", ROOT / "config" / "promotion.yaml")
INSTR, _ = load_instruments(ROOT / "config" / "instruments.yaml")
T0 = utc(2026, 1, 5, 12)
DAYS = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]


def series(seed: int) -> dict[date, float]:
    rng = np.random.default_rng(seed)
    return {d: float(x) for d, x in zip(DAYS, rng.normal(0.1, 1.0, len(DAYS)), strict=True)}


def perf(
    sid: str, sl: float, sb: float = 1.0, n: int = 150, stage: Stage = Stage.LIVE, seed: int | None = None
) -> VersionPerformance:
    return VersionPerformance(
        strategy_id=sid,
        version="1",
        stage=stage,
        live_trades=n,
        sharpe_live=sl,
        sharpe_backtest=sb,
        daily_r=series(seed if seed is not None else hash(sid) % 1000),
    )


def test_shrinkage_formula() -> None:
    assert shrunk_weight(0, 3.0, 1.0, 150) == 1.0  # no live trades: all backtest
    assert shrunk_weight(150, 3.0, 1.0, 150) == pytest.approx(2.0)
    assert shrunk_weight(1350, 3.0, 1.0, 150) == pytest.approx(2.8)


def test_correlated_versions_share_one_slot() -> None:
    a, b, c = perf("a", 1.0, seed=1), perf("b", 1.0, seed=1), perf("c", 1.0, seed=2)  # a and b identical
    assert clusters([a, b, c], 0.6, 20) == [[("a", "1"), ("b", "1")], [("c", "1")]]
    alloc = Allocator(CFG).rebalance([a, b, c], T0)
    # two slots of equal weight, both capped at 30%; a and b split theirs (40% stays unallocated)
    assert alloc.shares["a@1"] == pytest.approx(0.15) and alloc.shares["c@1"] == pytest.approx(0.30)


def test_cap_redistributes_and_leaves_rest_unallocated() -> None:
    s = capped_shares({("a", "1"): 10.0, ("b", "1"): 1.0, ("c", "1"): 1.0}, 0.3)
    assert s[("a", "1")] == pytest.approx(0.3) and s[("b", "1")] == pytest.approx(0.3)
    assert sum(s.values()) == pytest.approx(0.9)  # three versions x 30%: 10% stays unallocated


def test_negative_weight_gets_nothing_and_micro_gets_micro_fraction() -> None:
    al = Allocator(CFG)
    al.rebalance([perf("bad", -2.0, sb=-1.0), perf("good", 1.5), perf("new", 9.0, stage=Stage.MICRO)], T0)
    assert al.risk_fraction("bad", "1", Stage.LIVE) == 0.0
    assert al.risk_fraction("new", "1", Stage.MICRO) == CFG.stage_limit(Stage.MICRO)
    assert "new@1" not in al.current.shares  # type: ignore[union-attr]
    assert al.risk_fraction("good", "1", Stage.LIVE) == CFG.stage_limit(Stage.LIVE)  # 0.3 x 3% capped at 0.5%


def test_weekly_rebalance_never_intraday_but_demotion_lowers_at_once(tmp_path: Path) -> None:
    al = Allocator(CFG, tmp_path / "alloc.json")
    first = al.rebalance(
        [perf("a", 1.0, stage=Stage.SCALED), perf("b", 1.0), perf("c", 1.0), perf("d", 1.0)], T0
    )
    assert al.rebalance([perf("a", 5.0)], T0 + timedelta(days=3)) == first  # not due: unchanged
    assert al.risk_fraction("a", "1", Stage.SCALED) == pytest.approx(0.25 * 0.03)
    assert al.risk_fraction("a", "1", Stage.LIVE) == CFG.stage_limit(Stage.LIVE)  # demoted: capped now
    assert al.risk_fraction("a", "1", Stage.SHADOW) == 0.0
    assert Allocator(CFG, tmp_path / "alloc.json").current == first  # survives restart
    later = al.rebalance([perf("a", 5.0)], T0 + timedelta(days=7))
    assert later.at == T0 + timedelta(days=7)


def sig(stop: float = 4332.0) -> Signal:
    return Signal(
        signal_id=uuid4(),
        strategy_id="a",
        strategy_version="1",
        symbol="XAUUSD",
        side="buy",
        entry_type="market",
        entry_price=None,
        stop_price=stop,
        target_price=None,
        created_at=T0,
        reason="t",
    )


def test_proposal_sizes_from_the_fraction_and_is_one_intent_per_signal() -> None:
    al = Allocator(CFG)
    al.rebalance([perf("a", 1.0), perf("b", 1.0), perf("c", 1.0), perf("d", 1.0)], T0)
    s = sig()
    kw = {
        "equity": Decimal(20000),
        "entry": Decimal("4342.00"),
        "instrument": INSTR["XAUUSD"],
        "to_account": Decimal(1),
        "account_id": "a",
    }
    it = al.propose(s, Stage.LIVE, **kw)  # type: ignore[arg-type]
    assert it is not None and it.proposed_lots == Decimal("0.10")  # 0.5% of 20k / (10 x 100)
    assert al.propose(s, Stage.LIVE, **kw) == it  # type: ignore[arg-type]
    assert al.propose(s, Stage.SHADOW, **kw) is None  # type: ignore[arg-type]
    tiny = {**kw, "equity": Decimal(100)}
    assert al.propose(s, Stage.LIVE, **tiny) is None  # type: ignore[arg-type]  # below min lot: never rounded up


# ---------------------------------------------------------------- property tests

STAGES = st.sampled_from([Stage.MICRO, Stage.LIVE, Stage.SCALED])


@st.composite
def portfolios(draw: st.DrawFn) -> list[VersionPerformance]:
    n = draw(st.integers(1, 9))
    out = []
    for i in range(n):
        seed = draw(st.integers(0, 3))  # few seeds: correlated clusters happen often
        out.append(
            VersionPerformance(
                strategy_id=f"s{i}",
                version="1",
                stage=draw(STAGES),
                live_trades=draw(st.integers(0, 2000)),
                sharpe_live=draw(st.floats(-3, 6)),
                sharpe_backtest=draw(st.floats(-1, 4)),
                daily_r=series(seed),
            )
        )
    return out


@settings(max_examples=300, deadline=None)
@given(portfolios())
def test_allocator_never_exceeds_caps(p: list[VersionPerformance]) -> None:
    al = Allocator(CFG)
    a = al.rebalance(p, T0)
    shares = list(a.shares.values())
    event(f"versions weighted: {min(len(shares), 5)}")
    event(f"clusters with 2+: {sum(len(c) > 1 for c in a.clusters) > 0}")
    assert all(0.0 <= s <= CFG.max_share_per_version + 1e-12 for s in shares)
    assert sum(shares) <= 1.0 + 1e-9
    for v in p:
        rf = al.risk_fraction(v.strategy_id, v.version, v.stage)
        assert 0.0 <= rf <= CFG.stage_limit(v.stage) + 1e-15
        assert rf <= CFG.total_risk_budget * CFG.max_share_per_version + 1e-15 or v.stage == Stage.MICRO
        if v.stage == Stage.MICRO:
            assert rf == CFG.stage_limit(Stage.MICRO)
        elif shrunk_weight(v.live_trades, v.sharpe_live, v.sharpe_backtest, CFG.shrinkage_k) <= 0:
            assert rf == 0.0
    # a cluster together is one slot: never more than one version's cap
    for c in a.clusters:
        assert sum(a.shares[m] for m in c) <= CFG.max_share_per_version + 1e-12
