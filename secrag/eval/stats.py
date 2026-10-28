"""Uncertainty: percentile bootstrap confidence intervals over questions (decision D4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(values: list[float] | np.ndarray, n_samples: int = 2000, confidence: float = 0.95,
                 seed: int = 13) -> Estimate:
    """Mean of `values` with a percentile bootstrap CI, resampling questions with replacement."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_samples, x.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Estimate(float(x.mean()), float(low), float(high), int(x.size))
