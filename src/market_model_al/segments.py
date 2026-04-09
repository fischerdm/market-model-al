"""
Actuarially meaningful holdout segments for convergence tracking.

Each segment is a slice of the holdout set defined by a single feature
threshold.  Per-segment RMSE is tracked alongside global RMSE to reveal
whether a strategy is converging uniformly or concentrating on specific
parts of the feature space.

Segments are intentionally coarse and commercially motivated:
- Young drivers    — high-risk, commercially sensitive, often discussed
- High-value cars  — prestige segment, competitor may reprice differently
- High power       — performance vehicles, correlated with claims frequency
- Senior drivers   — low-risk, large share of portfolio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Segment:
    key: str                          # short identifier used as column suffix
    label: str                        # human-readable label for display
    description: str                  # one-line description for tooltips
    filter_fn: Callable[[pd.DataFrame], pd.Series]   # returns boolean mask


SEGMENTS: list[Segment] = [
    Segment(
        key="young_driver",
        label="Young drivers",
        description="driver_age < 30",
        filter_fn=lambda X: X["driver_age"] < 30,
    ),
    Segment(
        key="high_value",
        label="High-value cars",
        description="Value_vehicle > 25 000",
        filter_fn=lambda X: X["Value_vehicle"] > 25_000,
    ),
    Segment(
        key="high_power",
        label="High-power cars",
        description="Power > 150 hp",
        filter_fn=lambda X: X["Power"] > 150,
    ),
    Segment(
        key="senior_driver",
        label="Senior drivers",
        description="driver_age >= 60",
        filter_fn=lambda X: X["driver_age"] >= 60,
    ),
]


def segment_rmse(
    holdout_X: pd.DataFrame,
    holdout_y: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE for each segment on the holdout set.

    Parameters
    ----------
    holdout_X : pd.DataFrame
        Feature matrix for the fixed holdout set.
    holdout_y : np.ndarray
        True labels (oracle premiums) for the holdout.
    preds : np.ndarray
        Competitor model predictions for the holdout.

    Returns
    -------
    dict mapping segment key -> RMSE (NaN if the segment is empty).
    """
    results = {}
    residuals_sq = (holdout_y - preds) ** 2
    for seg in SEGMENTS:
        mask = seg.filter_fn(holdout_X).values
        if mask.sum() == 0:
            results[seg.key] = float("nan")
        else:
            results[seg.key] = float(np.sqrt(residuals_sq[mask].mean()))
    return results
