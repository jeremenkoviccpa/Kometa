"""Validation reports (spec section 9): a JSON result and a self-contained HTML page.

HTML sections: header (hashes, pass/fail per threshold), equity and drawdown,
monthly returns, R distribution, results per symbol / session / weekday,
walk-forward windows and parameter drift, Monte Carlo drawdown distribution,
stability, cross-market, holdout, cost breakdown. Charts are inline SVG; no
external assets.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from autotrader.core.indicators.sessions import sessions_of
from autotrader.core.series import from_ns
from autotrader.engine.simbroker import TradeRecord
from autotrader.validation.runner import ValidationReport, profit_factor

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _jsonable(x: Any) -> Any:
    if is_dataclass(x) and not isinstance(x, type):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else ("inf" if f > 0 else "-inf" if f < 0 else "nan")
    if isinstance(x, np.integer):
        return int(x)
    return x


def to_json(rep: ValidationReport) -> dict[str, Any]:
    out = _jsonable(rep)
    assert isinstance(out, dict)  # noqa: S101 - dataclass always maps to dict
    out["passed"] = rep.passed
    out["eligible_for_promotion"] = rep.eligible_for_promotion
    if rep.monte_carlo is not None:
        out["monte_carlo"].pop("drawdowns", None)
    return out


def write_json(rep: ValidationReport, path: Path) -> None:
    path.write_text(json.dumps(to_json(rep), indent=2, sort_keys=True))


# ------------------------------------------------------------------ breakdowns


def group(
    trades: Sequence[TradeRecord], key: Callable[[TradeRecord], str]
) -> list[tuple[str, int, float, float, float]]:
    """(group, trades, win rate, avg R, profit factor)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        buckets[key(t)].append(t.r_multiple)
    rows = []
    for k in sorted(buckets):
        r = np.asarray(buckets[k])
        rows.append((k, int(r.size), float((r > 0).mean()), float(r.mean()), profit_factor(r)))
    return rows


def _session(t: TradeRecord) -> str:
    s = sessions_of(from_ns(t.entry_time_ns))
    return "+".join(s) if s else "off-session"


# ------------------------------------------------------------------ svg


def _fmt(v: float, pct: bool = False, digits: int = 2) -> str:
    if isinstance(v, float) and not math.isfinite(v):
        return "inf" if v > 0 else "n/a"
    return f"{v:.{digits}%}" if pct else f"{v:,.{digits}f}"


def svg_line(
    series: Sequence[float] | np.ndarray, *, height: int = 180, fill: bool = False, label: str = ""
) -> str:
    w, h, pad = 720, height, 28
    y = np.asarray(series, dtype=np.float64)
    if y.size < 2:
        return "<p class='muted'>not enough data</p>"
    lo, hi = float(y.min()), float(y.max())
    if hi == lo:
        hi = lo + 1.0
    xs = np.linspace(pad, w - 8, y.size)
    ys = h - pad - (y - lo) / (hi - lo) * (h - 2 * pad)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys, strict=True))
    area = f"<polygon class='area' points='{pad},{h - pad} {pts} {w - 8},{h - pad}'/>" if fill else ""
    return (
        f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{html.escape(label)}'>"
        f"<line class='axis' x1='{pad}' y1='{h - pad}' x2='{w - 8}' y2='{h - pad}'/>"
        f"<text class='tick' x='2' y='{pad}'>{_fmt(hi, digits=3)}</text>"
        f"<text class='tick' x='2' y='{h - pad}'>{_fmt(lo, digits=3)}</text>"
        f"{area}<polyline class='line' points='{pts}'/></svg>"
    )


def svg_hist(values: Iterable[float], *, bins: int = 30, label: str = "", mark: float | None = None) -> str:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "<p class='muted'>no data</p>"
    counts, edges = np.histogram(v, bins=bins)
    w, h, pad = 720, 180, 24
    bw = (w - 2 * pad) / bins
    top = max(int(counts.max()), 1)
    bars = "".join(
        f"<rect class='bar' x='{pad + i * bw:.1f}' y='{h - pad - c / top * (h - 2 * pad):.1f}' "
        f"width='{bw - 1:.1f}' height='{c / top * (h - 2 * pad):.1f}'><title>{edges[i]:.3f} to "
        f"{edges[i + 1]:.3f}: {c}</title></rect>"
        for i, c in enumerate(counts)
    )
    marker = ""
    if mark is not None and edges[0] <= mark <= edges[-1]:
        x = pad + (mark - edges[0]) / (edges[-1] - edges[0]) * (w - 2 * pad)
        marker = f"<line class='mark' x1='{x:.1f}' y1='{pad}' x2='{x:.1f}' y2='{h - pad}'/>"
    return (
        f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{html.escape(label)}'>{bars}{marker}"
        f"<text class='tick' x='{pad}' y='{h - 4}'>{edges[0]:.3f}</text>"
        f"<text class='tick' x='{w - pad - 40}' y='{h - 4}'>{edges[-1]:.3f}</text></svg>"
    )


# ------------------------------------------------------------------ html


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def _badge(ok: bool) -> str:
    return "<span class='pass'>PASS</span>" if ok else "<span class='fail'>FAIL</span>"


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#2f5d8a;--area:#2f5d8a22;--grid:#d9d9d4;
--pass:#1f7a3a;--fail:#b3261e;--card:#ffffff;--mark:#b3261e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#ececea;--muted:#9a9a94;
--line:#7fb0e0;--area:#7fb0e022;--grid:#3a3a37;--pass:#6fcf8a;--fail:#ff8a80;--card:#1e1e1c;--mark:#ff8a80}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 8px;border-bottom:1px solid var(--grid);padding-bottom:4px}
.muted{color:var(--muted)}.pass{color:var(--pass);font-weight:600}.fail{color:var(--fail);font-weight:600}
.banner{padding:10px 12px;border:1px solid var(--fail);color:var(--fail);border-radius:6px;margin:12px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}svg{width:100%;height:auto;background:var(--card)}
.line{fill:none;stroke:var(--line);stroke-width:1.5}.area{fill:var(--area);stroke:none}
.axis{stroke:var(--grid)}.tick{fill:var(--muted);font-size:10px}.bar{fill:var(--line)}
.mark{stroke:var(--mark);stroke-width:2;stroke-dasharray:4 3}
code{font-size:12px;word-break:break-all}
"""


def to_html(rep: ValidationReport) -> str:
    esc = html.escape
    trades = rep.oos_trades
    r = np.asarray([t.r_multiple for t in trades])
    equity = np.cumsum(rep.oos_daily_returns) if rep.oos_daily_returns.size else np.zeros(0)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:] if equity.size else equity
    dd = peak - equity
    verdict = "PASSED" if rep.passed else "FAILED"
    parts = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Validation {esc(rep.strategy_id)} {esc(rep.version)}</title>"
        f"<style>{CSS}</style></head><body><main>",
        f"<h1>{esc(rep.strategy_id)} {esc(rep.version)}: "
        f"<span class='{'pass' if rep.passed else 'fail'}'>"
        f"{verdict}</span></h1>",
        f"<p class='muted'>family {esc(rep.family)} · research {rep.research_window[0].date()} to "
        f"{rep.research_window[1].date()} · holdout epoch {esc(rep.holdout_epoch)} "
        f"({rep.holdout_window[0].date()} to {rep.holdout_window[1].date()})</p>",
    ]
    if rep.synthetic:
        parts.append(
            "<div class='banner'>Synthetic data: plumbing check only, never eligible for promotion.</div>"
        )
    parts.append(
        _table(
            ["", "value"],
            [
                ("code hash", f"<code>{esc(rep.code_hash)}</code>"),
                ("config hash", f"<code>{esc(rep.config_hash)}</code>"),
                *[(f"data {esc(k)}", f"<code>{esc(v)}</code>") for k, v in rep.data_versions.items()],
                ("final params", f"<code>{esc(json.dumps(rep.final_params))}</code>"),
            ],
        )
    )
    parts.append("<h2>Thresholds</h2>")
    parts.append(
        _table(
            ["check", "value", "rule", "result"],
            [
                (
                    esc(c.name),
                    _fmt(c.value, digits=4),
                    f"{c.op} {_fmt(c.threshold, digits=4)}",
                    _badge(c.passed),
                )
                for c in rep.checks
            ],
        )
    )
    if rep.warnings:
        parts.append("<ul>" + "".join(f"<li>{esc(w)}</li>" for w in rep.warnings) + "</ul>")

    parts.append("<h2>Out-of-sample equity (cumulative daily R return at fixed risk)</h2>")
    parts.append(svg_line(equity, fill=True, label="equity"))
    parts.append("<h2>Drawdown</h2>")
    parts.append(svg_line(-dd, label="drawdown"))

    parts.append("<h2>Monthly returns</h2>")
    years = sorted({k[:4] for k in rep.monthly_returns})
    rows = []
    for y in years:
        vals = [rep.monthly_returns.get(f"{y}-{m:02d}") for m in range(1, 13)]
        rows.append(
            [
                y,
                *["" if v is None else _fmt(v, pct=True) for v in vals],
                _fmt(sum(v for v in vals if v is not None), pct=True),
            ]
        )
    parts.append(_table(["year", *[f"{m:02d}" for m in range(1, 13)], "total"], rows))

    parts.append("<h2>R distribution</h2>")
    win = float((r > 0).mean()) if r.size else 0.0
    avg = float(r.mean()) if r.size else 0.0
    excl = f" · {rep.excluded_trades} trades excluded (data quality)" if rep.excluded_trades else ""
    parts.append(
        f"<p class='muted'>{r.size} trades · win rate {_fmt(win, pct=True)} · "
        f"avg R {_fmt(avg, digits=3)} · PF {_fmt(profit_factor(r))}{excl}</p>"
    )
    parts.append(svg_hist(r, label="R distribution", mark=0.0))

    hdr = ["", "trades", "win rate", "avg R", "PF"]

    def rows_of(g: list[tuple[str, int, float, float, float]]) -> list[list[str]]:
        return [[esc(k), str(n), _fmt(w, pct=True), _fmt(a, digits=3), _fmt(pf)] for k, n, w, a, pf in g]

    parts.append("<h2>By symbol</h2>" + _table(hdr, rows_of(group(trades, lambda t: t.symbol))))
    parts.append("<h2>By session</h2>" + _table(hdr, rows_of(group(trades, _session))))
    parts.append(
        "<h2>By weekday</h2>"
        + _table(
            hdr,
            rows_of(
                group(
                    trades,
                    lambda t: (
                        f"{from_ns(t.entry_time_ns).weekday()} {WEEKDAYS[from_ns(t.entry_time_ns).weekday()]}"
                    ),
                )
            ),
        )
    )

    parts.append("<h2>Walk-forward windows</h2>")
    parts.append(
        _table(
            ["test window", "params", "train score", "test trades", "test PF", "test Sharpe (daily)"],
            [
                (
                    f"{w.test_start.date()} to {w.test_end.date()}",
                    f"<code>{esc(json.dumps(w.params))}</code>",
                    _fmt(w.train_score),
                    str(w.test_trades),
                    _fmt(w.test_pf),
                    _fmt(w.test_sharpe, digits=3),
                )
                for w in rep.windows
            ],
        )
    )

    if rep.dsr:
        d = rep.dsr
        parts.append("<h2>Deflated Sharpe</h2>")
        parts.append(
            _table(
                [
                    "Sharpe (daily)",
                    "SR* benchmark",
                    "trials in family",
                    "days",
                    "skew",
                    "kurtosis",
                    "probability",
                ],
                [
                    (
                        _fmt(d.sharpe, digits=4),
                        _fmt(d.sr_star, digits=4),
                        str(d.n_trials),
                        str(d.t),
                        _fmt(d.skew),
                        _fmt(d.kurt),
                        _fmt(d.probability, pct=True),
                    )
                ],
            )
        )
    if rep.monte_carlo:
        mc = rep.monte_carlo
        parts.append(
            f"<h2>Monte Carlo drawdown ({mc.runs} runs, block {mc.block_length}, "
            f"{_fmt(mc.risk_fraction, pct=True)} risk)</h2>"
        )
        parts.append(
            _table(
                ["p50", "p95", "p99", "final return p5", "final return p50"],
                [
                    (
                        _fmt(mc.dd_p50, pct=True),
                        _fmt(mc.dd_p95, pct=True),
                        _fmt(mc.dd_p99, pct=True),
                        _fmt(mc.final_return_p5, pct=True),
                        _fmt(mc.final_return_p50, pct=True),
                    )
                ],
            )
        )
        parts.append(svg_hist(mc.drawdowns, label="Monte Carlo drawdowns", mark=mc.dd_p95))

    def variants(rows: Sequence[Any]) -> str:
        return _table(
            ["variant", "trades", "PF", "result"],
            [(esc(v.label), str(v.trades), _fmt(v.profit_factor), _badge(v.passed)) for v in rows],
        )

    parts.append("<h2>Parameter stability</h2>" + variants(rep.stability))
    parts.append("<h2>Cross-market</h2>" + variants(rep.cross_market))
    parts.append(
        "<h2>Locked holdout</h2>"
        + (variants([rep.holdout]) if rep.holdout else "<p class='muted'>not opened</p>")
    )

    parts.append("<h2>Costs (out of sample, account currency)</h2>")
    parts.append(
        _table(
            ["gross P&L", "commission", "swap", "spread (est.)", "slippage (est.)", "net P&L"],
            [
                (
                    _fmt(sum(t.pnl_gross for t in trades)),
                    _fmt(sum(t.commission for t in trades)),
                    _fmt(sum(t.swap for t in trades)),
                    _fmt(sum(t.spread_cost for t in trades)),
                    _fmt(sum(t.slippage_cost for t in trades)),
                    _fmt(sum(t.pnl_net for t in trades)),
                )
            ],
        )
    )
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_html(rep: ValidationReport, path: Path) -> None:
    path.write_text(to_html(rep))
