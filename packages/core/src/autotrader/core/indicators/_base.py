from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

NAN = float("nan")


def as_f64(x: Any) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("indicator inputs must be 1-D")
    return arr


def check_period(n: int, name: str = "n") -> None:
    if n < 1:
        raise ValueError(f"{name} must be >= 1, got {n}")


def nan_array(size: int) -> FloatArray:
    return np.full(size, np.nan, dtype=np.float64)


def windows(x: FloatArray, n: int) -> FloatArray:
    """Rolling windows of length n; row i covers x[i : i + n]."""
    return np.lib.stride_tricks.sliding_window_view(x, n)
