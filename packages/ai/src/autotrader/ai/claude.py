"""The Anthropic client. Kept behind a plain async callable so the trader is testable without the network."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam

from autotrader.ai.decision import DECIDE_TOOL

Decider = Callable[[str, str], Awaitable[dict[str, Any]]]  # (system prompt, context) -> the tool's input


class ClaudeDecider:
    def __init__(self, api_key: str, model: str, *, timeout: float = 90.0) -> None:
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=1)

    async def __call__(self, system: str, context: str) -> dict[str, Any]:
        message: MessageParam = {"role": "user", "content": context}
        r = await self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[message],
            tools=[cast(ToolParam, DECIDE_TOOL)],
            tool_choice={"type": "tool", "name": "decide"},
        )
        for block in r.content:
            if block.type == "tool_use":
                return dict(cast(dict[str, Any], block.input))
        raise ValueError("Claude answered without a decision")
