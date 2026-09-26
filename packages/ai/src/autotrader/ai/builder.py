"""The strategy assistant (owner request 2026-09-26): the owner describes any trading idea in their own words,
Claude asks what is unclear and writes it as a Kometa strategy, every draft is checked (static checks, a
synthetic smoke backtest, the no-lookahead test) in a child process, and the owner saves it; the owner is
responsible for what they save. Saved strategies are owner strategies: they enter as candidates, trade only
where the owner switches them on (paper in the demo), and every order still goes through the risk gate.
Modifying a strategy saves a new version of it; the old one stays in history.

SPEC-QUESTION: spec 14.3 runs agent-generated code only in a network-less Docker sandbox. These are the
owner's strategies, written with the assistant's help and saved by the owner: they are checked in a child
process with a time limit and the AST static checks, then run in the demo process like any owner strategy
(open question 37).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from autotrader.ai.checks import check_draft
from autotrader.strategies_api.static_checks import ALLOWED_IMPORT_PREFIXES, FORBIDDEN_NAMES

# (system prompt, API messages, tools) -> the reply's content blocks as dicts (type text / tool_use)
ChatFn = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]
Checker = Callable[[Path], Awaitable[dict[str, Any]]]
ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
MAX_ROUNDS = 4  # drafts Claude may try within one owner message after failed checks

PROPOSE_TOOL: dict[str, Any] = {
    "name": "propose_strategy",
    "description": "Submit a complete strategy draft (manifest + code); Kometa checks it and replies.",
    "input_schema": {
        "type": "object",
        "properties": {
            "strategy_id": {"type": "string", "description": "lower_snake_case, 3-41 chars"},
            "summary": {
                "type": "string",
                "description": "What it does, in 2-4 plain sentences for the owner.",
            },
            "manifest_yaml": {"type": "string", "description": "The full strategy.yaml."},
            "code": {"type": "string", "description": "The full strategy.py."},
        },
        "required": ["strategy_id", "summary", "manifest_yaml", "code"],
    },
}


def _indicator_api() -> str:
    import autotrader.core.indicators as ind  # noqa: PLC0415 - read at prompt build time

    lines = []
    for name in ind.__all__:
        obj = getattr(ind, name)
        if inspect.isfunction(obj):
            doc = (inspect.getdoc(obj) or "").split("\n")[0]
            lines.append(f"- {name}{inspect.signature(obj)}  {doc}")
    return "\n".join(lines)


def guide(root: Path) -> str:
    """The assistant's instructions: Kometa's strategy API as it really is, and one working example."""
    ex = root / "strategies" / "library" / "scalp_session_breakout"
    return f"""\
You are Kometa's strategy assistant. The owner describes a trading idea in their own words (any language:
answer in theirs). Your job: understand it, ask about anything that is unclear or missing (market, timeframes,
entry trigger, stop, target, exits, filters), explain briefly how you will turn it into rules, and then write
it as a Kometa strategy with the propose_strategy tool. Keep the owner's idea; do not swap in your own. Be
honest: never promise profit; a backtest on synthetic data only shows that it runs. The owner is responsible
for what they save.

A strategy is two files.

strategy.yaml (manifest):
  id: lower_snake_case
  version: 1.0.0            # Kometa sets the real version on save
  origin: owner
  family: a_short_family_name
  symbols: [XAUUSD]         # from config/instruments.yaml: XAUUSD, EURUSD, GBPUSD, USDJPY, AUDUSD, ...
  timeframes: [M5, H1]      # M1 M5 M15 H1 H4 D1; on_bar is called for each
  params:                   # at most 6 with tunable: true; every tunable needs min and max
    name: {{value: 20, min: 10, max: 50, tunable: true}}
    flag: {{value: true}}
  expected: {{trades_per_month: 4, win_rate: 0.45, avg_r: 0.1}}
  description: >-
    Plain words.

strategy.py: one class deriving from Strategy.
  from autotrader.core.events import BarClosed
  from autotrader.core.models import Timeframe
  from autotrader.strategies_api import Request, Strategy, StrategyContext
  class MyStrategy(Strategy):
      def warmup(self) -> dict[tuple[str, Timeframe], int]: closed bars needed per (symbol, timeframe)
      def on_bar(self, ctx: StrategyContext, event: BarClosed) -> list[Request]: called on every closed bar
  event.symbol, event.timeframe, event.bar (open_time, close_time, bid_o/h/l/c, ask_o/h/l/c, volume)
  ctx.market.bars(symbol, tf, n) -> the last n CLOSED bars, arrays: open_time, close_time (int ns),
      bid_o, bid_h, bid_l, bid_c, ask_o, ask_h, ask_l, ask_c, volume (numpy float64); len(bars)
  ctx.market.spread(symbol), ctx.market.now (datetime), ctx.market.minutes_to_news(symbol) ->
      (minutes to the next, minutes since the last high-impact event) or None
  ctx.signal(symbol, "buy"|"sell", stop_price, entry_type="market"|"limit"|"stop", entry_price=None,
      target_price=None, expiry_bars=None, reason="...", tags={{...}})  -> append it to the returned list
      (every signal needs a stop on the right side; a limit/stop entry needs entry_price)
  ctx.close(position_id, reason), ctx.modify_stop(position_id, new_stop, reason), ctx.cancel(signal_id)
  ctx.my_positions(symbol) -> [PositionView(position_id, symbol, side, lots, entry_price, stop_price, ...)]
  ctx.my_pending(symbol), ctx.params["name"], ctx.state (a dict that must stay JSON-serializable)
  Buys fill at the ask, sells at the bid; bars are bid and ask.

Rules the checks enforce:
- Imports only from: {", ".join(ALLOWED_IMPORT_PREFIXES)}.
- Never use: {", ".join(sorted(FORBIDDEN_NAMES))}; no file, network, clock or randomness except ctx.rng.
- Closed bars only (no lookahead): the future-poisoning test replaces future prices and your signals before
  that point must not change.
- Size is never the strategy's: the risk gate sizes every trade and can only make it smaller.

Indicators (from autotrader.core.indicators, vectorized over arrays; output at i uses inputs 0..i only):
{_indicator_api()}

A working example, strategy.yaml:
{(ex / "strategy.yaml").read_text()}
strategy.py:
{(ex / "strategy.py").read_text()}

When the idea is clear enough, call propose_strategy. You will get the check results back; if a check
fails, fix it and propose again. When the checks pass, tell the owner in plain words what the strategy
does, what the smoke test showed (it only shows it runs), and that they can press Save.
"""


class Assistant:
    def __init__(
        self,
        root: Path,
        state: Path,
        chat: ChatFn | None,
        *,
        sources: Callable[[], Mapping[str, Path]],
        checker: Checker | None = None,
        disabled_reason: str = "",
        max_calls_per_day: int = 100,
    ) -> None:
        """`state`: the persistent directory (sessions, drafts, saved strategies). `sources`: every loadable
        strategy id -> its current directory (library and owner strategies), for viewing and modifying."""
        self.root, self.state, self.chat = root, state, chat
        self.sources = sources
        self.checker = checker or (lambda d: asyncio.to_thread(check_draft, d, root))
        self.disabled_reason = disabled_reason if chat is None else ""
        self.max_calls = max_calls_per_day
        self.calls: list[datetime] = []
        self.sessions_dir = state / "builder"
        self.saved_dir = state / "strategies"
        self._guide: str | None = None
        self.busy: set[str] = set()

    # ------------------------------------------------------------ sessions

    def _path(self, sid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", sid):
            raise KeyError(sid)
        return self.sessions_dir / f"{sid}.json"

    def session(self, sid: str) -> dict[str, Any]:
        p = self._path(sid)
        if not p.exists():
            raise KeyError(sid)
        s: dict[str, Any] = json.loads(p.read_text())
        return s

    def _store(self, s: dict[str, Any]) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        p = self._path(s["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(s))
        tmp.replace(p)

    def sessions(self) -> list[dict[str, Any]]:
        if not self.sessions_dir.exists():
            return []
        out = []
        for p in self.sessions_dir.glob("*.json"):
            s = json.loads(p.read_text())
            out.append(
                {
                    "id": s["id"],
                    "title": s["title"],
                    "base": s.get("base"),
                    "updated": s["updated"],
                    "draft": (s.get("draft") or {}).get("strategy_id"),
                    "draft_ok": (s.get("draft") or {}).get("checks", {}).get("ok"),
                }
            )
        return sorted(out, key=lambda x: x["updated"], reverse=True)

    def view(self, s: dict[str, Any]) -> dict[str, Any]:
        """What the hub shows: the conversation and the current draft (not the raw API messages)."""
        return {**{k: v for k, v in s.items() if k != "api"}, "thinking": s["id"] in self.busy}

    # ------------------------------------------------------------ reading strategies

    def code(self, strategy_id: str) -> dict[str, Any]:
        src = self.sources().get(strategy_id)
        if src is None:
            raise KeyError(strategy_id)
        return {
            "strategy_id": strategy_id,
            "manifest": (src / "strategy.yaml").read_text(),
            "code": (src / "strategy.py").read_text(),
            "owner_made": self.saved_dir in src.parents,
        }

    # ------------------------------------------------------------ the conversation

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.chat is not None,
            "disabled_reason": self.disabled_reason or ("" if self.chat else "no Anthropic API key"),
            "calls_today": self._calls_today(),
            "max_calls_per_day": self.max_calls,
        }

    def _calls_today(self) -> int:
        now = datetime.now(UTC)
        self.calls = [t for t in self.calls if (now - t).total_seconds() < 86_400]
        return len(self.calls)

    def post(self, sid: str | None, text: str, base: str | None = None) -> str:
        """Record the owner's message (a new session when `sid` is None) and return the session id; the reply
        comes from `reply()`, which the API runs in the background (it can take minutes)."""
        if self.chat is None:
            raise RuntimeError(self.disabled_reason or "the assistant needs an Anthropic API key")
        now = datetime.now(UTC).isoformat()
        api_text = text
        if sid is None:
            s: dict[str, Any] = {
                "id": uuid.uuid4().hex,
                "title": (f"change {base}: " if base else "") + text[:60],
                "base": base,
                "created": now,
                "updated": now,
                "display": [],
                "api": [],
                "draft": None,
                "saved": [],
            }
            if base is not None:  # Claude sees the current files; the owner's words are what is shown
                cur = self.code(base)
                api_text = (
                    f"I want to change my strategy `{base}`. Current strategy.yaml:\n{cur['manifest']}\n"
                    f"Current strategy.py:\n{cur['code']}\n\nWhat I want: {text}"
                )
        else:
            s = self.session(sid)
            if s["id"] in self.busy:
                raise RuntimeError("the assistant is still answering the last message")
        s["display"].append({"role": "owner", "text": text, "at": now})
        if s["api"] and s["api"][-1]["role"] == "user":  # after check results: one user turn, roles alternate
            last = s["api"][-1]
            blocks = (
                last["content"]
                if isinstance(last["content"], list)
                else [{"type": "text", "text": last["content"]}]
            )
            last["content"] = [*blocks, {"type": "text", "text": api_text}]
        else:
            s["api"].append({"role": "user", "content": api_text})
        s["updated"] = now
        self._store(s)
        self.busy.add(s["id"])
        return str(s["id"])

    async def reply(self, sid: str) -> dict[str, Any]:
        """Claude answers the last owner message, drafting and fixing until the checks pass or rounds end."""
        try:
            await self._reply(self.session(sid))
        finally:
            self.busy.discard(sid)
        return self.view(self.session(sid))  # after the flag is cleared: the finished view says so

    async def message(self, sid: str | None, text: str, base: str | None = None) -> dict[str, Any]:
        """post() and reply() in one call (tests, the CLI)."""
        return await self.reply(self.post(sid, text, base))

    async def _reply(self, s: dict[str, Any]) -> dict[str, Any]:
        assert self.chat is not None  # noqa: S101 - post() refused without it
        now = datetime.now(UTC).isoformat()
        if self._guide is None:
            self._guide = guide(self.root)
        for _ in range(MAX_ROUNDS):
            if self._calls_today() >= self.max_calls:
                s["display"].append(
                    {
                        "role": "assistant",
                        "text": f"The daily limit of {self.max_calls} assistant calls is reached.",
                        "at": now,
                    }
                )
                break
            self.calls.append(datetime.now(UTC))
            blocks = await self.chat(self._guide, s["api"], [PROPOSE_TOOL])
            s["api"].append({"role": "assistant", "content": blocks})
            texts = [b["text"] for b in blocks if b.get("type") == "text" and b.get("text", "").strip()]
            if texts:
                s["display"].append(
                    {"role": "assistant", "text": "\n\n".join(texts), "at": datetime.now(UTC).isoformat()}
                )
            uses = [b for b in blocks if b.get("type") == "tool_use" and b.get("name") == "propose_strategy"]
            if not uses:
                break
            results = []
            done = False
            for u in uses:
                draft = await self._check(s, u.get("input") or {})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": u["id"],
                        "content": json.dumps(draft["checks"])[:6000],
                    }
                )
                done = done or bool(draft["checks"].get("ok"))
            s["api"].append({"role": "user", "content": results})
            if done:  # let Claude explain the passing draft to the owner, without another draft
                continue
        s["updated"] = datetime.now(UTC).isoformat()
        self._store(s)
        return self.view(s)

    async def _check(self, s: dict[str, Any], inp: Mapping[str, Any]) -> dict[str, Any]:
        sid = str(inp.get("strategy_id", ""))
        draft: dict[str, Any] = {
            "strategy_id": sid,
            "summary": str(inp.get("summary", "")),
            "manifest_yaml": str(inp.get("manifest_yaml", "")),
            "code": str(inp.get("code", "")),
            "checks": {"ok": False},
            "at": datetime.now(UTC).isoformat(),
        }
        if not ID_RE.fullmatch(sid):
            draft["checks"] = {
                "ok": False,
                "stage": "id",
                "error": "strategy_id must be lower_snake_case, 3-41 chars",
            }
        else:
            d = self.sessions_dir / "drafts" / s["id"] / sid
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            try:
                manifest = normalize_manifest(draft["manifest_yaml"], sid, "1.0.0")
            except ValueError as e:
                draft["checks"] = {"ok": False, "stage": "manifest", "error": str(e)}
            else:
                (d / "strategy.yaml").write_text(manifest)
                (d / "strategy.py").write_text(draft["code"])
                draft["checks"] = await self.checker(d)
        s["draft"] = draft
        s["display"].append(
            {
                "role": "checks",
                "text": (
                    "passed" if draft["checks"].get("ok") else f"failed at {draft['checks'].get('stage')}"
                ),
                "checks": draft["checks"],
                "at": draft["at"],
            }
        )
        return draft

    # ------------------------------------------------------------ saving

    def save(self, sid: str, responsible: bool, versions: Mapping[str, list[str]]) -> dict[str, Any]:
        """Save the session's passing draft as a new version; `versions`: every known version per id."""
        if not responsible:
            raise PermissionError("saving needs the owner's confirmation that they are responsible for it")
        s = self.session(sid)
        draft = s.get("draft")
        if not draft or not draft["checks"].get("ok"):
            raise ValueError("there is no draft that passed the checks")
        strategy_id = draft["strategy_id"]
        version = next_version(versions.get(strategy_id, []))
        dest = self.saved_dir / strategy_id
        if dest.exists():  # keep the previous owner version for history
            old = yaml.safe_load((dest / "strategy.yaml").read_text()).get("version", "old")
            archive = self.saved_dir / "_history" / strategy_id / str(old)
            if not archive.exists():
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(dest, archive)
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        at = datetime.now(UTC).date().isoformat()
        header = (
            f"# Written with Kometa's strategy assistant, saved by the owner on {at}; the owner is\n"
            "# responsible for it. A candidate: it trades only where the owner switches it on,\n"
            "# and every order goes through the risk gate.\n"
        )
        (dest / "strategy.yaml").write_text(
            header + normalize_manifest(draft["manifest_yaml"], strategy_id, version)
        )
        (dest / "strategy.py").write_text(draft["code"])
        s["saved"].append(
            {"strategy_id": strategy_id, "version": version, "at": datetime.now(UTC).isoformat()}
        )
        s["updated"] = datetime.now(UTC).isoformat()
        self._store(s)
        return {"strategy_id": strategy_id, "version": version, "path": str(dest), "restart_needed": True}


def normalize_manifest(text: str, strategy_id: str, version: str) -> str:
    """The draft's manifest with Kometa's fields fixed: its id and version, origin owner, never demo_only."""
    try:
        m = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"strategy.yaml is not valid YAML: {e}") from e
    if not isinstance(m, dict):
        raise ValueError("strategy.yaml must be a mapping")
    m.update(id=strategy_id, version=version, origin="owner")
    m.pop("demo_only", None)
    return yaml.safe_dump(m, sort_keys=False, allow_unicode=True)


def next_version(known: list[str]) -> str:
    """1.0.0 for a new strategy; otherwise the next minor above every known version."""
    if not known:
        return "1.0.0"
    top = max(tuple(int(x) for x in v.split(".")) for v in known)
    return f"{top[0]}.{top[1] + 1}.0"
