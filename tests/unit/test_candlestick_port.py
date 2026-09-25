"""The candlestick pattern set is a faithful port of github.com/cm45t3r/candlestick (MIT), shifted to the
completing bar: the original library's own output on 4,000 varied candles is the oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autotrader.core import indicators as ind

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads((ROOT / "tests" / "golden" / "candlestick_oracle.json").read_text())
JS_NAME = {n: "".join(w.capitalize() if i else w for i, w in enumerate(n.split("_"))) for n in ind.PATTERNS}


def arrays() -> tuple[np.ndarray, ...]:
    return tuple(np.array(col, dtype=np.float64) for col in zip(*ORACLE["candles"], strict=True))


def test_every_pattern_is_in_the_oracle_and_fires() -> None:
    assert set(JS_NAME.values()) == set(ORACLE["patterns"])
    assert all(len(v) >= 2 for v in ORACLE["patterns"].values())  # no vacuous comparison


@pytest.mark.parametrize("name", list(ind.PATTERNS))
def test_matches_the_original_library_at_the_completing_bar(name: str) -> None:
    o, h, lo, c = arrays()
    got = np.flatnonzero(ind.candlestick_patterns(o, h, lo, c)[name]).tolist()
    k = ORACLE["lengths"][JS_NAME[name]]
    expected = [i + k - 1 for i in ORACLE["patterns"][JS_NAME[name]]]  # first candle -> last candle
    assert got == expected


def test_incremental_equals_vectorized_and_never_looks_ahead() -> None:
    o, h, lo, c = arrays()
    vec = ind.candlestick_patterns(o, h, lo, c)
    t = ind.CandlestickTracker()
    rows = [t.update(*map(float, bar)) for bar in zip(o, h, lo, c, strict=True)]
    for name in ind.PATTERNS:
        assert [r[name] for r in rows] == vec[name].tolist(), name
    cut = 2500  # garbage after the cut must not change anything up to it
    rng = np.random.default_rng(3)
    po, ph, pl, pc = (a.copy() for a in (o, h, lo, c))
    for a in (po, ph, pl, pc):
        a[cut + 1 :] = rng.uniform(-1e6, 1e6, a.size - cut - 1)
    pois = ind.candlestick_patterns(po, ph, pl, pc)
    for name in ind.PATTERNS:
        assert vec[name][: cut + 1].tolist() == pois[name][: cut + 1].tolist(), name


def test_last_bar_patterns_names_what_the_newest_bar_completed() -> None:
    bars = [(50.0, 51.0, 40.0, 41.0), (38.75, 40.0, 38.0, 38.25), (39.0, 48.0, 38.5, 47.0)]  # a morning star
    o, h, lo, c = (np.array(x) for x in zip(*bars, strict=True))
    assert "morning_star" in ind.last_bar_patterns(o, h, lo, c)
