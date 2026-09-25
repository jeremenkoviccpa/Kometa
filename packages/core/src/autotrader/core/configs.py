"""Config files are hashed when they are read (spec section 17).

Every loader of a file in `config/` reads it through `read_config`, which remembers the file's sha256.
The process that composes the services publishes `config_events()` on start, and the audit service
writes them as ConfigChanged records, so the audit log shows which exact config every run used.
tests/unit/test_process_rules.py fails if a loader reads a config file any other way.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from autotrader.core.events import ConfigChanged
from autotrader.core.hashing import sha256_hex

_lock = threading.Lock()
_loaded: dict[str, tuple[str, str]] = {}  # name -> (path, sha256), last read wins


def read_config(path: Path) -> bytes:
    data = path.read_bytes()
    with _lock:
        _loaded[path.name] = (str(path), sha256_hex(data))
    return data


def loaded() -> dict[str, tuple[str, str]]:
    with _lock:
        return dict(_loaded)


def config_events(service: str, at: datetime) -> list[ConfigChanged]:
    return [
        ConfigChanged(at=at, service=service, name=name, path=path, config_hash=h)
        for name, (path, h) in sorted(loaded().items())
    ]
