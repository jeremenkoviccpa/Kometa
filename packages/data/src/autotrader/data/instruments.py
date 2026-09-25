"""Load config/instruments.yaml into validated Instrument models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from autotrader.core.configs import read_config
from autotrader.core.hashing import hash_file
from autotrader.core.models import Instrument


def load_instruments(path: Path) -> tuple[dict[str, Instrument], str]:
    """Returns (instruments by symbol, file hash). The hash goes to the audit log."""
    raw: dict[str, Any] = yaml.safe_load(read_config(path))
    defaults: dict[str, Any] = raw.get("defaults", {})
    out: dict[str, Instrument] = {}
    for symbol, spec in raw["instruments"].items():
        merged = {**defaults.get(spec.get("asset_class", ""), {}), **spec, "symbol": symbol}
        # Decimals from YAML floats would inherit binary error; go via str
        merged = {k: str(v) if isinstance(v, float) else v for k, v in merged.items()}
        out[symbol] = Instrument.model_validate(merged)
    return out, hash_file(path)
