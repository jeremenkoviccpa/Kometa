"""`at` command line tool. Subcommands are added per phase."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from datetime import date as date_t
from decimal import Decimal
from pathlib import Path

import polars as pl
import yaml

from autotrader.core.alerts import MemoryAlertSink
from autotrader.core.broker import is_system_comment
from autotrader.core.clock import LiveClock
from autotrader.core.hashing import canonical_json
from autotrader.core.ledger import JsonlLedger
from autotrader.core.models import Instrument
from autotrader.core.settings import Settings
from autotrader.core.signing import generate_keypair, load_private, load_public, sign_bytes
from autotrader.core.timeutil import utc
from autotrader.data.instruments import load_instruments
from autotrader.data.market_hours import FX_HOURS
from autotrader.data.quality import check_bars
from autotrader.data.sources import FileSource
from autotrader.data.synthetic import SyntheticSpec, generate, synthetic_instrument
from autotrader.data.versioning import data_version
from autotrader.engine.backtest import BacktestConfig, run_backtest
from autotrader.execution.adapter import BrokerAdapter, BrokerUnavailableError
from autotrader.execution.mt5 import MT5Adapter
from autotrader.execution.service import StartupRefusedError, startup_checks
from autotrader.lifecycle.registry import IllegalTransitionError, Registry, VersionInfo
from autotrader.risk.config import ConfigSignatureError, RiskLimits, load_signed
from autotrader.strategies_api.loader import StrategyLoadError, load_strategy
from autotrader.validation.config import ValidationConfig
from autotrader.validation.inputs import prepare
from autotrader.validation.poisoning import future_poisoning_test
from autotrader.validation.report import write_html, write_json
from autotrader.validation.runner import Validator, default_epoch
from autotrader.validation.store import HoldoutLock, TrialRegistry


def _synth(args: argparse.Namespace) -> int:
    df = generate(SyntheticSpec(symbol=args.symbol, days=args.days, seed=args.seed))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.symbol}_M1.parquet"
    df.write_parquet(path)
    print(f"{path}  rows={df.height}  data_version={data_version(df, args.symbol, 'M1', synthetic=True)}")
    return 0


def _check(args: argparse.Namespace) -> int:
    df = FileSource(Path(args.root), assume_tz=args.assume_tz).m1_bars(
        args.symbol, utc(1970, 1, 1), utc(2100, 1, 1)
    )
    issues = check_bars(df, args.symbol, FX_HOURS)
    for i in issues:
        print(f"{i.severity:6} {i.issue_type:14} {i.start.isoformat()} .. {i.end.isoformat()} {i.detail}")
    print(f"{len(issues)} issues, {sum(i.severity == 'high' for i in issues)} high")
    return 1 if any(i.severity == "high" for i in issues) else 0


def _load_frames(args: argparse.Namespace, symbols: tuple[str, ...]) -> tuple[dict[str, pl.DataFrame], bool]:
    if args.data is None:
        spec = SyntheticSpec(days=args.days, seed=args.seed)
        return {
            s: generate(SyntheticSpec(symbol=s, days=spec.days, seed=spec.seed + i))
            for i, s in enumerate(symbols)
        }, True
    src = FileSource(Path(args.data), assume_tz=args.assume_tz)
    return {s: src.m1_bars(s, utc(1970, 1, 1), utc(2100, 1, 1)) for s in symbols}, False


def _instruments(
    args: argparse.Namespace, symbols: tuple[str, ...], synthetic: bool
) -> dict[str, Instrument]:
    if synthetic:
        return {s: synthetic_instrument(s) for s in symbols}
    inst, _ = load_instruments(Path(args.instruments))
    return {s: inst[s] for s in symbols}


def _backtest(args: argparse.Namespace) -> int:
    ls = load_strategy(Path(args.strategy))
    frames, synthetic = _load_frames(args, ls.manifest.symbols)
    inp = prepare(
        frames, ls.manifest, _instruments(args, ls.manifest.symbols, synthetic), synthetic=synthetic
    )
    cfg = BacktestConfig(initial_equity=args.equity, account_ccy=args.account_ccy, risk_fraction=args.risk)
    r = run_backtest(ls.cls, inp.m1, inp.series, inp.instruments, inp.cost_model, config=cfg)
    m = r.metrics
    print(f"{ls.manifest.id} {ls.manifest.version}  code={ls.code_hash[:12]}  data={inp.data_versions}")
    print(
        f"trades={m.trades} win={m.win_rate:.1%} avgR={m.avg_r:.3f} totalR={m.total_r:.1f} "
        f"PF={m.profit_factor:.2f} maxDD={m.max_drawdown:.1%} sharpe={m.sharpe_daily_ann:.2f}"
    )
    print(
        f"net={m.net_pnl:,.2f} commission={m.commission:,.2f} swap={m.swap:,.2f} "
        f"spread={m.spread_cost:,.2f} slippage={m.slippage_cost:,.2f} rejections={len(r.rejections)}"
    )
    if synthetic:
        print("NOTE: synthetic data; results are plumbing checks, never evidence.")
    return 0


def _strategy_check(args: argparse.Namespace) -> int:
    try:
        ls = load_strategy(Path(args.strategy))
    except StrategyLoadError as e:
        print(f"FAIL load/static checks: {e}")
        return 1
    print(f"ok   static checks ({ls.manifest.id} {ls.manifest.version})")
    frames, synthetic = _load_frames(args, ls.manifest.symbols)
    instruments = _instruments(args, ls.manifest.symbols, synthetic)
    first = next(iter(frames.values()))
    rc = 0
    for frac in (0.3, 0.6, 0.9):
        cut = first["open_time"][int(first.height * frac)]
        rep = future_poisoning_test(ls.cls, frames, instruments, cut)
        status = "ok  " if rep.passed else "FAIL"
        print(f"{status} future poisoning at {cut.isoformat()} ({rep.signals_checked} signals)")
        if not rep.passed:
            print(rep.first_difference)
            rc = 1
    return rc


def _validate(args: argparse.Namespace) -> int:
    settings = Settings()
    ls = load_strategy(Path(args.strategy))
    cfg, cfg_hash = ValidationConfig.load(Path(args.config))
    universe = tuple(dict.fromkeys([*ls.manifest.symbols, *(args.universe or [])]))
    if args.data is None and len(universe) < cfg.cross_market.of_pairs:
        universe = universe + tuple(
            f"SYN{i}" for i in range(2, 2 + cfg.cross_market.of_pairs - len(universe))
        )
    frames, synthetic = _load_frames(args, universe)
    ledger = JsonlLedger(settings.ledger_path)
    epoch_end = min(f["open_time"][-1] for f in frames.values())
    epoch = default_epoch(epoch_end, cfg.holdout.months, args.epoch)
    v = Validator(
        cfg,
        TrialRegistry(ledger),
        HoldoutLock(ledger, epoch.epoch, cfg.holdout.max_attempts_per_family_per_epoch),
        config_hash=cfg_hash,
    )
    rep = v.validate(
        ls,
        frames,
        _instruments(args, universe, synthetic),
        epoch,
        universe=list(universe),
        synthetic=synthetic,
    )
    out = Path(args.out or settings.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{ls.manifest.id}_{ls.manifest.version}_{ls.code_hash[:8]}"
    write_json(rep, stem.with_suffix(".json"))
    write_html(rep, stem.with_suffix(".html"))
    for c in rep.checks:
        print(f"{'ok  ' if c.passed else 'FAIL'} {c.name:28} {c.value:>12.4f} {c.op} {c.threshold}")
    for w in rep.warnings:
        print(f"warn {w}")
    print(f"{'PASSED' if rep.passed else 'FAILED'}  report: {stem.with_suffix('.html')}")
    return 0 if rep.passed else 1


def _risk_keygen(args: argparse.Namespace) -> int:
    key_path, pub_path = Path(args.key).expanduser(), Path(args.pub)
    if key_path.exists() and not args.force:
        print(f"refusing to overwrite {key_path} (use --force)")
        return 1
    priv, pub = generate_keypair()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.touch(mode=0o600, exist_ok=True)
    key_path.chmod(0o600)
    key_path.write_text(priv + "\n")
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_text(pub + "\n")
    print(f"private key: {key_path} (keep it OFF every server)\npublic key:  {pub_path}")
    return 0


def _risk_sign(args: argparse.Namespace) -> int:
    cfg = Path(args.config)
    RiskLimits.model_validate(yaml.safe_load(cfg.read_text()))  # never sign an invalid config
    sig = sign_bytes(load_private(Path(args.key).expanduser()), cfg.read_bytes())
    out = cfg.with_name(cfg.name + ".sig")
    out.write_text(sig + "\n")
    print(f"signed {cfg} -> {out}")
    return 0


def _risk_verify(args: argparse.Namespace) -> int:
    cfg = Path(args.config)
    try:
        _, digest = load_signed(cfg, cfg.with_name(cfg.name + ".sig"), load_public(Path(args.pub)))
    except (ConfigSignatureError, ValueError) as e:
        print(f"INVALID: {e}")
        return 1
    print(f"valid  sha256={digest}")
    return 0


def _risk_resume_token(args: argparse.Namespace) -> int:
    now = datetime.now(UTC)
    token = {
        "action": "resume_full_halt",
        "nonce": secrets.token_hex(16),
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=args.ttl_minutes)).isoformat(),
    }
    sig = sign_bytes(load_private(Path(args.key).expanduser()), canonical_json(token).encode())
    print(json.dumps({"token": token, "signature": sig}))
    return 0


def _make_adapter(settings: Settings) -> BrokerAdapter:
    if not settings.bridge_url or settings.bridge_token is None:
        raise SystemExit("AT_BRIDGE_URL and AT_BRIDGE_TOKEN must be set")
    return MT5Adapter(settings.bridge_url, settings.bridge_token.get_secret_value())


async def _execution_check_async(settings: Settings) -> int:
    adapter = _make_adapter(settings)
    try:
        acct = await startup_checks(
            adapter,
            LiveClock(),
            env=settings.env,
            expected_account_id=settings.account_id if settings.account_id != "default" else None,
            max_skew_seconds=settings.max_clock_skew_seconds,
        )
        symbols = await adapter.symbols()
        positions = await adapter.open_positions()
        orders = await adapter.pending_orders()
    except (StartupRefusedError, BrokerUnavailableError, PermissionError) as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"account {acct.account_id} {acct.trade_mode} {acct.margin_mode} {acct.currency}")
    print(f"balance {acct.balance} equity {acct.equity} free margin {acct.free_margin}")
    print(f"server time {acct.server_time.isoformat()}")
    for si in symbols:
        print(
            f"  {si.symbol}: digits={si.digits} contract={si.contract_size} lots {si.min_lot}..{si.max_lot}"
        )
    external = [p for p in positions if not is_system_comment(p.comment)]
    print(f"positions {len(positions)} (external {len(external)}), pending orders {len(orders)}")
    return 0


def _execution_check(args: argparse.Namespace) -> int:
    """Read-only: can execution start against this broker account? Places nothing."""
    return asyncio.run(_execution_check_async(Settings()))


def _registry() -> tuple[Registry, MemoryAlertSink]:
    alerts = MemoryAlertSink()
    return Registry(JsonlLedger(Settings().registry_path), LiveClock(), alerts), alerts


def _lifecycle_submit(args: argparse.Namespace) -> int:
    """Register a strategy version as a candidate. demo_only versions go straight to (capped) shadow."""
    try:
        loaded = load_strategy(Path(args.strategy))
    except StrategyLoadError as e:
        print(f"REFUSED: {e}")
        return 1
    m = loaded.manifest
    reg, _ = _registry()
    reg.submit_candidate(
        VersionInfo(
            strategy_id=m.id,
            version=m.version,
            family=m.family,
            origin=m.origin,
            demo_only=m.demo_only,
            code_hash=loaded.code_hash,
            params=dict(m.param_values({})),
            created_by="owner",
        )
    )
    print(f"{m.id} {m.version}: candidate")
    if m.demo_only:
        reg.promote_candidate(m.id, m.version, validation_passed=False, synthetic=True)
        print(f"{m.id} {m.version}: shadow (demo_only, capped)")
    return 0


def _lifecycle_status(args: argparse.Namespace) -> int:
    reg, _ = _registry()
    for v in sorted(reg.versions(), key=lambda v: v.key):
        flag = " demo_only" if v.info.demo_only else ""
        since = f"{v.stage_since:%Y-%m-%d}"
        print(f"{v.info.strategy_id:24} {v.info.version:10} {v.stage.value:9} since {since}{flag}")
    return 0


def _lifecycle_retire(args: argparse.Namespace) -> int:
    reg, _ = _registry()
    try:
        reg.retire(args.strategy_id, args.version, f"owner: {args.reason}")
    except (IllegalTransitionError, KeyError) as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"{args.strategy_id} {args.version}: retired")
    return 0


def _audit_verify(args: argparse.Namespace) -> int:
    from autotrader.core.ledger import LedgerCorruptError  # noqa: PLC0415

    if args.db:
        from autotrader.monitor.pg_ledger import PgLedger  # noqa: PLC0415

        ledger: JsonlLedger | PgLedger = PgLedger(Settings().database_url.get_secret_value())
        where = "audit_log (postgres)"
    else:
        path = Path(args.path) if args.path else Settings().audit_path
        ledger, where = JsonlLedger(path), str(path)
    try:
        n = ledger.verify()
    except LedgerCorruptError as e:
        print(f"BROKEN: {e}")
        return 1
    print(f"{where}: {n} records, chain intact")
    return 0


def _demo_run(args: argparse.Namespace) -> int:
    from autotrader.cli.demo import DemoConfig, run_demo  # noqa: PLC0415

    cfg = DemoConfig(
        root=Path(args.root).resolve(),
        var=Path(args.var),
        broker=args.broker,
        catchup_days=args.catchup_days,
        speed=args.speed,
        port=args.port,
        equity=Decimal(str(args.equity)),
        symbol=args.symbol,
        start_price=args.start_price,
        trade=tuple(args.trade) if args.trade is not None else None,
        host=args.host,
        run_days=args.run_days,
    )
    if args.fresh:
        if args.broker != "sim":
            print("--fresh only applies to the simulated market (broker state must never be wiped)")
            return 1
        import shutil  # noqa: PLC0415

        shutil.rmtree(cfg.var, ignore_errors=True)
    try:
        asyncio.run(run_demo(cfg))
    except KeyboardInterrupt:
        print("stopped")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    from datetime import UTC, date, datetime, timedelta  # noqa: PLC0415

    from autotrader.data.dukascopy import DukascopyFetcher  # noqa: PLC0415

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else datetime.now(UTC).date() - timedelta(days=1)
    if args.source == "oanda":
        return asyncio.run(_fetch_oanda(args, start, end))
    fetcher = DukascopyFetcher(Path(args.out), concurrency=args.concurrency, pause=args.pause, progress=print)
    rep = asyncio.run(fetcher.fetch(args.symbol, start, end))
    print(
        f"{args.symbol}: {rep.days} days, {rep.downloaded} files downloaded, {rep.cached} cached, "
        f"{len(rep.failed)} failed; {rep.bars:,} M1 bars in {Path(args.out)}/{args.symbol}_M1.parquet"
    )
    if rep.failed:
        print(
            "failed (run again to retry):", ", ".join(rep.failed[:10]), "..." if len(rep.failed) > 10 else ""
        )
    return 1 if rep.failed else 0


async def _fetch_oanda(args: argparse.Namespace, start: date_t, end: date_t) -> int:
    from autotrader.execution.history import fetch_history  # noqa: PLC0415
    from autotrader.execution.oanda import OandaAdapter  # noqa: PLC0415

    settings = Settings()
    if not settings.oanda_account or settings.oanda_token is None:
        print("put AT_OANDA_TOKEN and AT_OANDA_ACCOUNT in .env first (see config/settings.example.env)")
        return 1
    instruments, _ = load_instruments(Path("config/instruments.yaml"))
    adapter = OandaAdapter(
        settings.oanda_account,
        settings.oanda_token.get_secret_value(),
        {args.symbol: instruments[args.symbol].contract_size},
        environment=settings.oanda_environment,
    )
    out = Path(args.out if args.out != "data/dukascopy" else "data/oanda")
    try:
        n = await fetch_history(adapter, args.symbol, start, end, out, progress=print)
    finally:
        await adapter.aclose()
    print(f"{args.symbol}: {n:,} M1 bars in {out}/{args.symbol}_M1.parquet")
    return 0 if n else 1


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data", default=None, help="directory with <SYMBOL>_M1.{csv,parquet}; synthetic if omitted"
    )
    p.add_argument("--assume-tz", default=None)
    p.add_argument("--instruments", default="config/instruments.yaml")
    p.add_argument("--days", type=int, default=365, help="synthetic days")
    p.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="at", description="AutoTrader control tool")
    sub = p.add_subparsers(dest="cmd", required=True)
    data = sub.add_parser("data", help="market data tools").add_subparsers(dest="data_cmd", required=True)

    s = data.add_parser("synth", help="generate synthetic M1 bars")
    s.add_argument("--symbol", default="SYNTH")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="data/synthetic")
    s.set_defaults(func=_synth)

    f = data.add_parser("fetch", help="download real M1 history into <out>/<SYMBOL>_M1.parquet")
    f.add_argument(
        "--source",
        choices=["dukascopy", "oanda"],
        default="dukascopy",
        help="oanda: your OANDA account's history (AT_OANDA_TOKEN/ACCOUNT in .env), fast",
    )
    f.add_argument("--symbol", default="XAUUSD")
    f.add_argument("--from", dest="start", required=True, help="first day, YYYY-MM-DD (UTC)")
    f.add_argument("--to", dest="end", default=None, help="last day, YYYY-MM-DD (default: yesterday)")
    f.add_argument("--out", default="data/dukascopy")
    f.add_argument("--concurrency", type=int, default=2)
    f.add_argument("--pause", type=float, default=0.5, help="seconds between requests per worker")
    f.set_defaults(func=_fetch)

    c = data.add_parser("check", help="run quality checks on a symbol's M1 file")
    c.add_argument("symbol")
    c.add_argument("--root", default="data/synthetic")
    c.add_argument("--assume-tz", default=None)
    c.set_defaults(func=_check)

    b = sub.add_parser("backtest", help="run one backtest")
    b.add_argument("strategy", help="strategy directory")
    _add_data_args(b)
    b.add_argument("--equity", type=float, default=100_000.0)
    b.add_argument("--account-ccy", default="USD")
    b.add_argument("--risk", type=float, default=0.005)
    b.set_defaults(func=_backtest)

    st = sub.add_parser("strategy", help="strategy tools").add_subparsers(dest="strategy_cmd", required=True)
    sc = st.add_parser("check", help="static checks and future poisoning test")
    sc.add_argument("strategy")
    _add_data_args(sc)
    sc.set_defaults(func=_strategy_check)

    va = sub.add_parser("validate", help="full validation suite with HTML and JSON report")
    va.add_argument("strategy")
    _add_data_args(va)
    va.add_argument("--universe", nargs="*", help="extra symbols for cross-market")
    va.add_argument("--config", default="config/validation.yaml")
    va.add_argument("--epoch", default=None, help="holdout epoch id")
    va.add_argument("--out", default=None)
    va.set_defaults(func=_validate)

    rk = sub.add_parser("risk", help="owner-side risk tools").add_subparsers(dest="risk_cmd", required=True)
    kg = rk.add_parser("keygen", help="create the owner Ed25519 key pair (run on the owner's machine)")
    kg.add_argument("--key", default="~/.autotrader/owner_ed25519.key")
    kg.add_argument("--pub", default="config/owner_ed25519.pub")
    kg.add_argument("--force", action="store_true")
    kg.set_defaults(func=_risk_keygen)
    sg = rk.add_parser("sign", help="sign config/risk.yaml with the owner key")
    sg.add_argument("config", nargs="?", default="config/risk.yaml")
    sg.add_argument("--key", default="~/.autotrader/owner_ed25519.key")
    sg.set_defaults(func=_risk_sign)
    vf = rk.add_parser("verify", help="check risk.yaml against its signature")
    vf.add_argument("config", nargs="?", default="config/risk.yaml")
    vf.add_argument("--pub", default="config/owner_ed25519.pub")
    vf.set_defaults(func=_risk_verify)
    rt = rk.add_parser("resume-token", help="owner-signed token that clears a FULL_HALT")
    rt.add_argument("--key", default="~/.autotrader/owner_ed25519.key")
    rt.add_argument("--ttl-minutes", type=int, default=15)
    rt.set_defaults(func=_risk_resume_token)

    ex = sub.add_parser("execution", help="execution tools").add_subparsers(
        dest="execution_cmd", required=True
    )
    ec = ex.add_parser("check", help="read-only startup checks against the broker bridge (places nothing)")
    ec.set_defaults(func=_execution_check)

    lc = sub.add_parser("lifecycle", help="strategy registry and stages").add_subparsers(
        dest="lifecycle_cmd", required=True
    )
    ls = lc.add_parser("submit", help="register a strategy version as a candidate")
    ls.add_argument("strategy", help="strategy directory")
    ls.set_defaults(func=_lifecycle_submit)
    lc.add_parser("status", help="every version and its stage").set_defaults(func=_lifecycle_status)
    lr = lc.add_parser("retire", help="owner command: retire a version from any stage")
    lr.add_argument("strategy_id")
    lr.add_argument("version")
    lr.add_argument("--reason", required=True)
    lr.set_defaults(func=_lifecycle_retire)

    au = sub.add_parser("audit", help="audit log").add_subparsers(dest="audit_cmd", required=True)
    av = au.add_parser("verify", help="walk the hash chain and report any break")
    src = av.add_mutually_exclusive_group()
    src.add_argument("--path", default=None, help="JSON Lines audit ledger (default AT_AUDIT_PATH)")
    src.add_argument("--db", action="store_true", help="the audit_log table in AT_DATABASE_URL")
    av.set_defaults(func=_audit_verify)

    dm = sub.add_parser("demo", help="run the whole system with the dashboard").add_subparsers(
        dest="demo_cmd", required=True
    )
    dr = dm.add_parser("run", help="simulated market (default) or a broker demo account via the MT5 bridge")
    dr.add_argument(
        "--broker",
        choices=["sim", "oanda", "mt5"],
        default="sim",
        help="sim: simulated market; oanda: OANDA practice account; mt5: MT5 demo via the bridge",
    )
    dr.add_argument("--speed", type=float, default=3600.0, help="simulated seconds per real second (sim)")
    dr.add_argument(
        "--catchup-days", type=int, default=15, help="market days played at full speed first (sim)"
    )
    dr.add_argument(
        "--equity",
        type=float,
        default=50_000.0,
        help="starting balance (sim); gold at minimum size needs about 25k at the 0.1%% paper-stage risk",
    )
    dr.add_argument("--port", type=int, default=8000)
    dr.add_argument(
        "--host",
        default="127.0.0.1",
        help="0.0.0.0 for a server: then every API call needs AT_API_OWNER_TOKEN",
    )
    dr.add_argument("--run-days", type=float, default=60.0, help="simulated market days after the catch-up")
    dr.add_argument("--fresh", action="store_true", help="sim only: start from an empty state directory")
    dr.add_argument("--symbol", default="XAUUSD", help="market to trade (sim: synthetic prices, real spec)")
    dr.add_argument(
        "--trade", nargs="*", default=None, help="strategy ids paper trading at start (default: the library)"
    )
    dr.add_argument("--start-price", type=float, default=None, help="first simulated price (sim)")
    dr.add_argument("--var", default="var/demo", help="state directory (journal, audit, registry)")
    dr.add_argument("--root", default=".", help="repository root (config/, strategies/)")
    dr.set_defaults(func=_demo_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
