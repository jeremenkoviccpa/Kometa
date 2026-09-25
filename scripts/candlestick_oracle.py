"""Build tests/golden/candlestick_oracle.json: the cm45t3r/candlestick JavaScript library (MIT) run on
varied candles, used to prove core.indicators.candlestick is a faithful port.

Usage: `uv run python scripts/candlestick_oracle.py <path to a clone of github.com/cm45t3r/candlestick>`.
Needs Node. The committed JSON makes the test independent of Node and of network access.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "golden" / "candlestick_oracle.json"
# name -> candles in the pattern (the library reports the index of the FIRST candle)
LENGTHS = {
    "hammer": 1, "bullishHammer": 1, "bearishHammer": 1, "invertedHammer": 1, "bullishInvertedHammer": 1,
    "bearishInvertedHammer": 1, "doji": 1, "marubozu": 1, "bullishMarubozu": 1, "bearishMarubozu": 1,
    "spinningTop": 1, "bullishSpinningTop": 1, "bearishSpinningTop": 1,
    "bullishEngulfing": 2, "bearishEngulfing": 2, "bullishHarami": 2, "bearishHarami": 2, "bullishKicker": 2,
    "bearishKicker": 2, "hangingMan": 2, "shootingStar": 2, "piercingLine": 2, "darkCloudCover": 2,
    "tweezersTop": 2, "tweezersBottom": 2,
    "morningStar": 3, "eveningStar": 3, "threeWhiteSoldiers": 3, "threeBlackCrows": 3,
}  # fmt: skip


def candles(n: int, seed: int) -> list[list[float]]:
    """Random shapes plus gaps, equal extremes and repeated bars, on a 2-decimal grid."""
    rng = random.Random(seed)  # noqa: S311 - test data, not security
    out: list[list[float]] = []
    close = 100.0
    for _ in range(n):
        u = rng.random()
        o = close + (rng.choice([-1, 1]) * rng.uniform(0.5, 3) if u < 0.15 else rng.uniform(-0.2, 0.2))
        shape = rng.random()
        body = rng.choice([0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0]) * rng.choice([-1, 1])
        c = o + body
        up = rng.choice([0.0, 0.02, 0.1, 0.5, 1.0, 2.5]) if shape < 0.9 else 0.0
        dn = rng.choice([0.0, 0.02, 0.1, 0.5, 1.0, 2.5]) if shape > 0.1 else 0.0
        h, lo = max(o, c) + up, min(o, c) - dn
        if out and rng.random() < 0.08:  # equal highs or lows with the previous bar (tweezers)
            if rng.random() < 0.5:
                h = max(out[-1][1], o, c)
            else:
                lo = min(out[-1][2], o, c)
        bar = [round(o, 2), round(h, 2), round(lo, 2), round(c, 2)]
        bar[1], bar[2] = max(bar), min(bar)
        if out and rng.random() < 0.02:
            bar = list(out[-1])  # exact repeats
        out.append(bar)
        close = bar[3]
    return out


def main() -> int:
    lib = Path(sys.argv[1])
    data = candles(4000, 20260925)
    js = f"""
const c = require({json.dumps(str(lib / "index.js"))});
const data = {json.dumps(data)}.map(([open, high, low, close]) => ({{open, high, low, close}}));
const names = {json.dumps(list(LENGTHS))};
const out = {{}};
for (const n of names) out[n] = c[n](data);
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)  # noqa: S603, S607
    patterns = json.loads(res.stdout)
    head = subprocess.run(  # noqa: S603
        ["git", "-C", str(lib), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    doc = {
        "source": "github.com/cm45t3r/candlestick (MIT)",
        "commit": head.stdout.strip(),
        "note": "indices are the FIRST candle of each pattern, as the library reports them",
        "lengths": LENGTHS,
        "candles": data,
        "patterns": patterns,
    }
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print({k: len(v) for k, v in patterns.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
