"""Logging with secret redaction (spec section 17)."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

REDACTED = "***REDACTED***"

# Shapes that look like secrets even when we were not told about them.
_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),  # Anthropic keys
    re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{30,}\b"),  # Telegram bot tokens
    re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key)(\s*[=:]\s*)([^\s,;&]+)"),
    re.compile(r"(postgres(?:ql)?(?:\+\w+)?://[^:/\s]+:)([^@\s]+)(@)"),
]


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = sorted({s for s in secrets if len(s) >= 4}, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, REDACTED)
        text = _PATTERNS[0].sub(REDACTED, text)
        text = _PATTERNS[1].sub(REDACTED, text)
        text = _PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
        return _PATTERNS[3].sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(3)}", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.getMessage())
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        return True


def configure_logging(secrets: Iterable[str] = (), level: int = logging.INFO) -> None:
    """Install the redaction filter on every handler of the root logger."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    flt = RedactingFilter(secrets)
    for h in root.handlers:
        h.addFilter(flt)
