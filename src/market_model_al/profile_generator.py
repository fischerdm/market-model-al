"""
Ceteris-paribus profile generator.

Builds candidate profiles by taking anchor rows and sweeping one continuous
feature at a time across its range while holding all other features fixed.
This mirrors how a real aggregator scraper probes a competitor's tariff.

Usage
-----
    from market_model_al.profile_generator import generate_ceteris_paribus
    from market_model_al.oracle_engine import OraclePricingEngine

    engine  = OraclePricingEngine("outputs/models/oracle.pkl")
    anchors = df[FEATURES].sample(100, random_state=42)

    profiles = generate_ceteris_paribus(anchors)   # invalid rows dropped automatically
    prices   = engine.query(profiles)
"""

import numpy as np
import pandas as pd

from market_model_al.constraints import validate as _validate

# Continuous features eligible for sweeping.
CONTINUOUS_FEATURES = [
    "driver_age",
    "licence_age",
    "vehicle_age",
    "Power",
    "Cylinder_capacity",
    "Value_vehicle",
    "Seniority",
]

# Default sweep grids — chosen to cover the actuarially relevant range with
# enough resolution to reveal tariff structure.  Callers may override any
# subset via the feature_ranges argument.
DEFAULT_RANGES: dict[str, np.ndarray] = {
    "driver_age":        np.arange(18, 81, 1),               # 63 values
    "licence_age":       np.arange(0, 51, 1),                # 51 values
    "vehicle_age":       np.arange(0, 31, 1),                # 31 values
    "Power":             np.arange(40, 401, 20),             # 19 values
    "Cylinder_capacity": np.arange(500, 5001, 250),          # 19 values
    "Value_vehicle":     np.geomspace(1_000, 80_000, 30),    # 30 log-spaced values
    "Seniority":         np.arange(0, 41, 1),                # 41 values
}

# Pre-validation upper bound on profiles generated per anchor row.
# Actual count is lower due to the licence_age <= driver_age - 18 constraint.
# Use this to convert a weekly scraping budget (profiles) to an anchor count.
PROFILES_PER_ANCHOR: int = sum(len(v) for v in DEFAULT_RANGES.values())  # 254


def generate_ceteris_paribus(
    anchors: pd.DataFrame,
    feature_ranges: dict[str, np.ndarray] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Generate ceteris-paribus profiles from anchor rows.

    For each anchor row and each continuous feature present in both the anchor
    DataFrame and the sweep grid, a copy of the row is produced for every
    value in the feature's grid, with all other columns held fixed.

    Parameters
    ----------
    anchors : pd.DataFrame
        Rows in engineered-feature space (same columns as the oracle expects).
        Typically a random sample of real policy rows.
    feature_ranges : dict, optional
        Override the default sweep grid for any subset of features.
        Keys are feature names; values are 1-D arrays of values to try.
    validate : bool, default True
        Whether to drop physically invalid profiles (e.g. licence_age >
        driver_age - 18) before returning.

    Returns
    -------
    pd.DataFrame
        Generated profiles (no labels), index reset.
    """
    ranges = {**DEFAULT_RANGES, **(feature_ranges or {})}

    sweep_features = [f for f in CONTINUOUS_FEATURES if f in anchors.columns and f in ranges]

    chunks = []
    for _, anchor in anchors.iterrows():
        for feature in sweep_features:
            block = pd.DataFrame(
                [anchor.to_dict()] * len(ranges[feature])
            )
            block[feature] = ranges[feature]
            chunks.append(block)

    if not chunks:
        return pd.DataFrame(columns=anchors.columns)

    profiles = pd.concat(chunks, ignore_index=True)

    # Restore categorical dtypes (lost during dict round-trip)
    for col in anchors.select_dtypes("category").columns:
        profiles[col] = profiles[col].astype("category")

    if validate:
        mask = _validate(profiles)
        profiles = profiles[mask].reset_index(drop=True)

    return profiles
