"""Risk gate (spec section 11): mandatory property tests, sizing example, signatures, halts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from autotrader.core.hashing import canonical_json
from autotrader.core.indicators.sessions import EventIndex
from autotrader.core.models import CalendarEvent, HaltState, OrderIntent, Signal, Stage
from autotrader.core.signing import (
    DecisionRejectedError,
    DecisionVerifier,
    generate_keypair,
    load_private,
    load_public,
    sign_bytes,
    sign_decision,
)
from autotrader.core.timeutil import utc
from autotrader.data.instruments import load_instruments
from autotrader.risk.config import ConfigSignatureError, RiskLimits, load_signed
from autotrader.risk.gate import Exposure, RiskGate, Snapshot
from autotrader.risk.state import StateStore

ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS, _ = load_instruments(ROOT / "config" / "instruments.yaml")
WED_NOON = utc(2026, 1, 7, 12)  # a Wednesday, no cutoff
OWNER_PRIV, OWNER_PUB = generate_keypair()
GATE_PRIV, GATE_PUB = generate_keypair()


def signed_config(tmp: Path, text: str | None = None) -> tuple[Path, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = tmp / "risk.yaml"
    cfg.write_text(text if text is not None else (ROOT / "config" / "risk.yaml").read_text())
    sig = tmp / "risk.yaml.sig"
    sig.write_text(sign_bytes(load_private(OWNER_PRIV), cfg.read_bytes()))
    return cfg, sig


@pytest.fixture
def limits(tmp_path: Path) -> tuple[RiskLimits, str]:
    cfg, sig = signed_config(tmp_path)
    return load_signed(cfg, sig, load_public(OWNER_PUB))


def halt_of(g: RiskGate) -> HaltState:
    """Read the halt state through a call so mypy does not narrow it across method calls."""
    return g.state.halt


def make_gate(tmp: Path, limits: tuple[RiskLimits, str]) -> RiskGate:
    return RiskGate(
        limits[0], limits[1], load_private(GATE_PRIV), load_public(OWNER_PUB), StateStore(tmp / "st.json")
    )


def signal(symbol: str = "XAUUSD", side: str = "buy", stop: float = 4332.0, sid: str = "s1") -> Signal:
    return Signal(
        signal_id=uuid4(),
        strategy_id=sid,
        strategy_version="1.0.0",
        symbol=symbol,
        side=side,
        entry_type="market",
        entry_price=None,
        stop_price=stop,
        target_price=None,
        created_at=WED_NOON,
        reason="t",
    )


def snap(
    *,
    equity: str = "20000",
    bid: str = "4341.80",
    ask: str = "4342.00",
    exposures: tuple[Exposure, ...] = (),
    stage: Stage = Stage.LIVE,
    now: datetime = WED_NOON,
    margin: dict[str, Decimal] | None = None,
    events: EventIndex | None = None,
    sid: str = "s1",
) -> Snapshot:
    return Snapshot(
        now=now,
        equity=Decimal(equity),
        free_margin=Decimal(equity),
        bid=Decimal(bid),
        ask=Decimal(ask),
        exposures=exposures,
        instruments=INSTRUMENTS,
        to_account=lambda ccy: Decimal(1) if ccy == "USD" else Decimal("0.0067"),
        margin_per_lot=margin if margin is not None else {s: Decimal(100) for s in INSTRUMENTS},
        stages={(sid, "1.0.0"): stage},
        events=events,
    )


def intent(sig: Signal, lots: str, rf: float = 0.005) -> OrderIntent:
    return OrderIntent(
        intent_id=uuid4(), signal=sig, proposed_lots=Decimal(lots), risk_fraction=rf, account_id="a"
    )


# ---------------------------------------------------------------- spec examples


def test_xauusd_sizing_example(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    g = make_gate(tmp_path, limits)
    d = g.decide(intent(signal(), "0.10"), snap())
    assert d.verdict == "approve"
    assert d.approved_lots == Decimal("0.10")
    # 9 lots would risk 9,000 USD (45% of the account): never approved
    d9 = g.decide(intent(signal(), "9"), snap())
    assert d9.verdict == "resize"
    assert d9.approved_lots == Decimal("0.10")


def test_tampered_config_prevents_startup(tmp_path: Path) -> None:
    cfg, sig = signed_config(tmp_path)
    cfg.write_text(cfg.read_text().replace("daily_loss_halt: 0.02", "daily_loss_halt: 0.20"))
    with pytest.raises(ConfigSignatureError, match="invalid signature"):
        load_signed(cfg, sig, load_public(OWNER_PUB))
    sig.unlink()
    with pytest.raises(ConfigSignatureError, match="missing"):
        load_signed(cfg, sig, load_public(OWNER_PUB))
    # signed by someone else
    other_priv, _ = generate_keypair()
    sig.write_text(sign_bytes(load_private(other_priv), cfg.read_bytes()))
    with pytest.raises(ConfigSignatureError):
        load_signed(cfg, sig, load_public(OWNER_PUB))


def test_require_stop_cannot_be_disabled(tmp_path: Path) -> None:
    text = (ROOT / "config" / "risk.yaml").read_text().replace("require_stop: true", "require_stop: false")
    cfg, sig = signed_config(tmp_path, text)
    with pytest.raises(ValueError, match="require_stop"):
        load_signed(cfg, sig, load_public(OWNER_PUB))


# ---------------------------------------------------------------- mandatory property tests

exposure_st = st.builds(
    Exposure,
    symbol=st.sampled_from(["EURUSD", "GBPUSD", "XAUUSD", "AUDUSD"]),
    side=st.sampled_from(["buy", "sell"]),
    lots=st.decimals(min_value="0.01", max_value="3", places=2),
    entry=st.just(Decimal("1.1")),
    stop=st.one_of(st.none(), st.just(Decimal("1.09")), st.just(Decimal("1.11"))),
    strategy_id=st.sampled_from(["s1", "s2", None]),
    pending=st.booleans(),
    external=st.booleans(),
)

PROPS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


@PROPS
@given(
    lots=st.decimals(min_value="0.01", max_value="50", places=2),
    stop_dist=st.decimals(min_value="-20", max_value="40", places=2),
    equity=st.decimals(min_value="100", max_value="1000000", places=0),
    rf=st.floats(min_value=0.0001, max_value=0.05),
    exposures=st.lists(exposure_st, max_size=6),
    side=st.sampled_from(["buy", "sell"]),
)
def test_properties(
    tmp_path: Path,
    limits: tuple[RiskLimits, str],
    lots: Decimal,
    stop_dist: Decimal,
    equity: Decimal,
    rf: float,
    exposures: list[Exposure],
    side: str,
) -> None:
    g = make_gate(tmp_path, limits)
    ask, bid = Decimal("4342.00"), Decimal("4341.80")
    entry = ask if side == "buy" else bid
    stop = entry - stop_dist if side == "buy" else entry + stop_dist
    if stop <= 0:
        return
    sig = signal(side=side, stop=float(stop))
    s = snap(equity=str(equity), exposures=tuple(exposures))
    d = g.decide(intent(sig, str(lots), rf), s)
    event(d.verdict)
    # approved never exceeds proposed
    assert d.approved_lots <= lots
    # no approval without a valid stop (wrong side or inside 1.5 x spread)
    if stop_dist < Decimal("1.5") * (ask - bid):
        assert d.verdict == "reject"
    if d.verdict != "reject":
        # total open risk after approval stays within the limit
        total = sum((g._risk_money(e, s) for e in exposures), Decimal(0))
        new = abs(entry - stop) * INSTRUMENTS["XAUUSD"].contract_size * d.approved_lots
        assert total + new <= limits[0].open_risk_total_max * equity + Decimal("1e-9")
        assert d.approved_lots >= INSTRUMENTS["XAUUSD"].min_lot


@PROPS
@given(
    halt=st.sampled_from([h for h in HaltState if h != HaltState.NORMAL]),
    lots=st.decimals(min_value="0.01", max_value="5", places=2),
)
def test_no_approval_in_any_halt(
    tmp_path: Path, limits: tuple[RiskLimits, str], halt: HaltState, lots: Decimal
) -> None:
    g = make_gate(tmp_path, limits)
    g.state.halt = halt
    d = g.decide(intent(signal(), str(lots)), snap())
    assert d.verdict == "reject"
    assert d.approved_lots == 0


# ---------------------------------------------------------------- individual checks


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"stage": Stage.SHADOW}, "does not trade"),
        ({"stage": Stage.DEMO_ONLY}, "does not trade"),
        ({"margin": {}}, "no margin data"),
        ({"now": utc(2026, 1, 9, 17, 30)}, "weekly cutoff"),  # Friday 18:30 Belgrade
        ({"now": utc(2026, 1, 10, 12)}, "weekly cutoff"),  # Saturday
    ],
)
def test_rejections(
    tmp_path: Path, limits: tuple[RiskLimits, str], kw: dict[str, object], reason: str
) -> None:
    g = make_gate(tmp_path, limits)
    d = g.decide(intent(signal(), "0.05"), snap(**kw))  # type: ignore[arg-type]
    assert d.verdict == "reject"
    assert any(reason in r for r in d.reasons), d.reasons


def test_news_blackout(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    ev = EventIndex(
        [CalendarEvent(time=WED_NOON + timedelta(minutes=10), currency="USD", impact="high", name="x")]
    )
    d = make_gate(tmp_path, limits).decide(intent(signal(), "0.05"), snap(events=ev))
    assert d.verdict == "reject"
    assert "news blackout" in d.reasons


def test_same_currency_same_direction(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    # two positions already short USD (long EURUSD, long GBPUSD); buying XAUUSD is short USD too
    ex = tuple(
        Exposure(s, "buy", Decimal("0.01"), Decimal("1.1"), Decimal("1.09"), "s2")
        for s in ("EURUSD", "GBPUSD")
    )
    d = make_gate(tmp_path, limits).decide(intent(signal(), "0.05"), snap(exposures=ex))
    assert d.verdict == "reject"
    assert any("short USD" in r for r in d.reasons)


def test_margin_resizes(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    # free margin 20000, ratio 3 -> required margin <= 5000 -> 0.05 lot at 100000 per lot
    d = make_gate(tmp_path, limits).decide(
        intent(signal(), "0.10"), snap(margin={"XAUUSD": Decimal(100_000)})
    )
    assert d.verdict == "resize"
    assert d.approved_lots == Decimal("0.05")


def test_symbol_lot_cap(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    ex = (Exposure("XAUUSD", "sell", Decimal("1.98"), Decimal("4342"), Decimal("4400"), "s9"),)
    d = make_gate(tmp_path, limits).decide(intent(signal(), "0.10"), snap(equity="1000000", exposures=ex))
    assert d.approved_lots <= Decimal("0.02")


# ---------------------------------------------------------------- signatures


def test_decision_signature_expiry_and_replay(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    g = make_gate(tmp_path, limits)
    it = intent(signal(), "0.10")
    d = g.decide(it, snap())
    v = DecisionVerifier(load_public(GATE_PUB))
    v.verify(d, str(it.intent_id), WED_NOON)
    with pytest.raises(DecisionRejectedError, match="replayed"):
        v.verify(d, str(it.intent_id), WED_NOON)
    d2 = g.decide(it, snap())
    tampered = d2.model_copy(update={"approved_lots": Decimal(9)})
    with pytest.raises(DecisionRejectedError, match="signature"):
        DecisionVerifier(load_public(GATE_PUB)).verify(tampered, str(it.intent_id), WED_NOON)
    with pytest.raises(DecisionRejectedError, match="expired"):
        DecisionVerifier(load_public(GATE_PUB)).verify(d2, str(it.intent_id), WED_NOON + timedelta(seconds=6))
    with pytest.raises(DecisionRejectedError, match="another intent"):
        DecisionVerifier(load_public(GATE_PUB)).verify(d2, str(uuid4()), WED_NOON)
    forged_priv, _ = generate_keypair()
    forged = sign_decision(load_private(forged_priv), d2.model_copy(update={"sequence": 999}))
    with pytest.raises(DecisionRejectedError, match="signature"):
        DecisionVerifier(load_public(GATE_PUB)).verify(forged, str(it.intent_id), WED_NOON)


# ---------------------------------------------------------------- halts


def test_daily_halt_persists_and_clears_next_day(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    g = make_gate(tmp_path, limits)
    g.roll_day(Decimal(10_000), WED_NOON, new_week=True)
    assert g.on_account(Decimal(9_850), WED_NOON) is None  # -1.5%
    act = g.on_account(Decimal(9_790), WED_NOON)  # -2.1%
    assert act is not None
    assert act.state == HaltState.DAILY_HALT
    assert act.close_positions
    assert act.cancel_pending_entries
    restarted = make_gate(tmp_path, limits)  # restart never clears a halt
    assert halt_of(restarted) == HaltState.DAILY_HALT
    restarted.roll_day(Decimal(9_790), WED_NOON + timedelta(days=1), new_week=False)
    assert halt_of(restarted) == HaltState.NORMAL


def test_weekly_halt_on_a_slow_slide_that_never_trips_the_daily(
    tmp_path: Path, limits: tuple[RiskLimits, str]
) -> None:
    """Spec 11: weekly loss against the week's opening equity; each day stays under the 2% daily limit."""
    g = make_gate(tmp_path, limits)
    g.roll_day(Decimal(10_000), WED_NOON, new_week=True)
    for day, eq in enumerate((9_850, 9_700, 9_560), start=1):
        assert g.on_account(Decimal(eq), WED_NOON + timedelta(days=day - 1, hours=4)) is None  # < 2% a day
        g.roll_day(Decimal(eq), WED_NOON + timedelta(days=day), new_week=False)
    view = {h["name"]: h for h in g.usage(Decimal(9_490))["halts"]}
    assert view["Weekly loss"]["loss"] == pytest.approx(0.051) and view["Daily loss"]["loss"] < 0.02
    act = g.on_account(Decimal(9_490), WED_NOON + timedelta(days=3, hours=4))  # -5.1% on the week
    assert act is not None and act.state == HaltState.WEEKLY_HALT and act.close_positions


def test_usage_view_is_the_gate_formula(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    """The hub's "how close to a halt" bars use the same numbers the gate halts on."""
    g = make_gate(tmp_path, limits)
    assert all(h["loss"] is None for h in g.usage(Decimal(10_000))["halts"])  # no references yet
    g.roll_day(Decimal(10_000), WED_NOON, new_week=True)
    daily = next(h for h in g.usage(Decimal(9_800))["halts"] if h["halt"] == "DAILY_HALT")
    assert daily["loss"] == pytest.approx(0.02) and daily["limit"] == pytest.approx(0.02)
    assert g.on_account(Decimal(9_800), WED_NOON) is not None  # at the limit: halted, as the bar says


def test_full_halt_needs_owner_token(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    g = make_gate(tmp_path, limits)
    g.roll_day(Decimal(10_000), WED_NOON, new_week=True)
    act = g.on_account(Decimal(8_400), WED_NOON)  # -16% from peak
    assert act is not None
    assert act.state == HaltState.FULL_HALT
    g.roll_day(Decimal(8_400), WED_NOON + timedelta(days=7), new_week=True)
    assert halt_of(g) == HaltState.FULL_HALT  # does not clear on its own
    token = {
        "action": "resume_full_halt",
        "nonce": "n1",
        "expires_at": (WED_NOON + timedelta(hours=1)).isoformat(),
    }
    wrong_priv, _ = generate_keypair()
    assert not g.resume(token, sign_bytes(load_private(wrong_priv), canonical_json(token).encode()), WED_NOON)
    good = sign_bytes(load_private(OWNER_PRIV), canonical_json(token).encode())
    assert not g.resume(token, good, WED_NOON + timedelta(hours=2))  # expired
    assert g.resume(token, good, WED_NOON)
    assert halt_of(g) == HaltState.NORMAL
    g.enter_full_halt("test", WED_NOON)
    assert not g.resume(token, good, WED_NOON)  # nonce already used


def test_recon_halt_does_not_close_positions(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    g = make_gate(tmp_path, limits)
    act = g.enter_recon_halt("mismatch", WED_NOON)
    assert act is not None
    assert not act.close_positions
    assert act.cancel_pending_entries
    assert g.clear_recon_halt()


def test_corrupt_state_loads_as_full_halt(tmp_path: Path, limits: tuple[RiskLimits, str]) -> None:
    g = make_gate(tmp_path, limits)
    g.roll_day(Decimal(10_000), WED_NOON, new_week=True)
    p = tmp_path / "st.json"
    doc = json.loads(p.read_text())
    doc["state"]["halt"] = "NORMAL"
    doc["state"]["peak_eod_equity"] = "1"
    p.write_text(json.dumps(doc))
    assert make_gate(tmp_path, limits).state.halt == HaltState.FULL_HALT


approvable_exposure = st.builds(
    Exposure,
    symbol=st.just("XAUUSD"),
    side=st.just("sell"),  # opposite direction: does not trip the same-currency rule
    lots=st.decimals(min_value="0.01", max_value="0.5", places=2),
    entry=st.just(Decimal("4342")),
    stop=st.decimals(min_value="4343", max_value="4400", places=2),
    strategy_id=st.sampled_from(["s1", "s2"]),
    pending=st.booleans(),
    external=st.just(False),
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    lots=st.decimals(min_value="0.01", max_value="5", places=2),
    stop_dist=st.decimals(min_value="1", max_value="60", places=2),
    equity=st.decimals(min_value="20000", max_value="2000000", places=0),
    rf=st.floats(min_value=0.001, max_value=0.02),
    exposures=st.lists(approvable_exposure, max_size=3),
)
def test_properties_when_approving(
    tmp_path: Path,
    limits: tuple[RiskLimits, str],
    lots: Decimal,
    stop_dist: Decimal,
    equity: Decimal,
    rf: float,
    exposures: list[Exposure],
) -> None:
    """Same invariants, with inputs biased toward approvals so the post-approval checks are exercised."""
    g = make_gate(tmp_path, limits)
    entry = Decimal("4342.00")
    stop = entry - stop_dist
    s = snap(equity=str(equity), exposures=tuple(exposures), margin={"XAUUSD": Decimal(1000)})
    d = g.decide(intent(signal(stop=float(stop)), str(lots), rf), s)
    event(d.verdict)
    assert d.approved_lots <= lots
    if d.verdict != "reject":
        lim = limits[0]
        contract = INSTRUMENTS["XAUUSD"].contract_size
        new = stop_dist * contract * d.approved_lots
        total = sum((g._risk_money(e, s) for e in exposures), Decimal(0))
        strat = sum((g._risk_money(e, s) for e in exposures if e.strategy_id == "s1"), Decimal(0))
        assert new <= lim.risk_per_trade_max * equity + Decimal("1e-9")
        assert new <= Decimal(str(rf)) * equity + Decimal("1e-9")
        assert strat + new <= lim.open_risk_per_strategy_max * equity + Decimal("1e-9")
        assert total + new <= lim.open_risk_total_max * equity + Decimal("1e-9")
        held = sum((e.lots for e in exposures), Decimal(0))
        assert held + d.approved_lots <= lim.max_lots("XAUUSD")
        assert d.approved_lots % INSTRUMENTS["XAUUSD"].lot_step == 0


def test_demo_only_needs_a_signed_stage_limit_and_trades_minimum_size(tmp_path: Path) -> None:
    prod = make_gate(
        tmp_path / "prod", load_signed(*signed_config(tmp_path / "prod_cfg"), load_public(OWNER_PUB))
    )
    d = prod.decide(intent(signal(), "0.10"), snap(stage=Stage.DEMO_ONLY))
    assert d.verdict == "reject" and "stage demo_only does not trade" in d.reasons
    paper_text = (ROOT / "config" / "risk.paper.yaml").read_text()
    paper = make_gate(
        tmp_path / "paper",
        load_signed(*signed_config(tmp_path / "paper_cfg", paper_text), load_public(OWNER_PUB)),
    )
    d = paper.decide(intent(signal(), "0.10"), snap(stage=Stage.DEMO_ONLY))
    assert d.verdict == "resize" and d.approved_lots == Decimal("0.01")  # minimum size only
