"""
Profile generators for the AL simulation.

Builds candidate profiles from anchor rows using one of two methods:

* Ceteris-paribus (CP) — sweeps one continuous feature at a time across its
  full range while holding all other features fixed.  Mirrors how a systematic
  aggregator scraper probes a competitor's tariff.

* Gaussian perturbations — perturbs all continuous features simultaneously with
  independent Gaussian noise.  Keeps profiles near the anchor while exposing
  the model to joint feature variation.

The shared helper ``create_market_supplement`` wraps both generators and is the
single entry point for building the synthetic portion of the market (the
10 % top-up in the warm start and the ``random_market`` strategy each week).

Usage
-----
    from market_model_al.profile_generator import (
        generate_ceteris_paribus,
        generate_gaussian_profiles,
        create_market_supplement,
    )
    from market_model_al.oracle_engine import OraclePricingEngine

    engine  = OraclePricingEngine("outputs/models/oracle.pkl")
    anchors = df[FEATURES].sample(100, random_state=42)
    rng     = np.random.default_rng(42)

    profiles = generate_ceteris_paribus(anchors)
    supplement = create_market_supplement(anchors, n=500, method="gaussian", rng=rng)
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

# Upper bounds for clipping Gaussian-perturbed features (derived from sweep grids).
_FEATURE_UPPER: dict[str, float] = {f: float(v.max()) for f, v in DEFAULT_RANGES.items()}
_FEATURE_LOWER: dict[str, float] = {f: float(v.min()) for f, v in DEFAULT_RANGES.items()}

# Gaussian profiles per anchor — set equal to PROFILES_PER_ANCHOR so that
# Gaussian and CP strategies run at identical weekly budgets (same n_anchors).
GAUSSIAN_PROFILES_PER_ANCHOR: int = PROFILES_PER_ANCHOR  # 254


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


def generate_gaussian_profiles(
    anchors: pd.DataFrame,
    n_per_anchor: int = GAUSSIAN_PROFILES_PER_ANCHOR,
    sigma_frac: float = 0.3,
    rng: np.random.Generator | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Generate joint Gaussian-perturbation profiles from anchor rows.

    For each anchor row, draw ``n_per_anchor`` profiles by perturbing *all*
    continuous features simultaneously with independent Gaussian noise.  Unlike
    ceteris-paribus profiles — which vary one feature at a time — these profiles
    expose the model to joint feature variation and therefore allow LightGBM to
    learn interaction effects from a single anchor's batch.

    Sigma for each feature is ``sigma_frac × (feature_max − feature_min)``
    where the bounds are derived from ``DEFAULT_RANGES``.  Values are clipped to
    the valid range and then constraint-validated (same rules as CP profiles).

    Parameters
    ----------
    anchors : pd.DataFrame
        Rows in engineered-feature space, same columns as the oracle expects.
    n_per_anchor : int
        Number of profiles to draw per anchor (pre-validation).
    sigma_frac : float
        Noise width as a fraction of each feature's range.
        0.3 → σ ≈ 30 % of range (moderate spread, stays near anchor).
        1.0 → σ ≈ full range (approaches uniform over the feature space).
    rng : np.random.Generator, optional
        RNG for reproducibility.  If None a fresh generator is created.
    validate : bool, default True
        Drop physically invalid profiles before returning.

    Returns
    -------
    pd.DataFrame
        Generated profiles (no labels), index reset.
    """
    if rng is None:
        rng = np.random.default_rng()

    sweep_features = [
        f for f in CONTINUOUS_FEATURES
        if f in anchors.columns and f in _FEATURE_UPPER
    ]

    # Pre-compute per-feature sigma and clip bounds.
    sigmas = {
        f: sigma_frac * (_FEATURE_UPPER[f] - _FEATURE_LOWER[f])
        for f in sweep_features
    }

    chunks = []
    for _, anchor in anchors.iterrows():
        block = pd.DataFrame([anchor.to_dict()] * n_per_anchor)
        for feature in sweep_features:
            noise = rng.normal(loc=0.0, scale=sigmas[feature], size=n_per_anchor)
            perturbed = anchor[feature] + noise
            perturbed = np.clip(perturbed, _FEATURE_LOWER[feature], _FEATURE_UPPER[feature])
            block[feature] = perturbed
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


def create_market_supplement(
    anchors: pd.DataFrame,
    n: int,
    method: str,
    rng: np.random.Generator,
    gaussian_sigma_frac: float = 0.3,
) -> pd.DataFrame:
    """Generate and sample n supplement profiles from anchor rows.

    Builds a pool using the requested method, then draws n rows from it
    (with replacement if the pool is smaller than n).  Used consistently in
    the warm start and the ``random_market`` strategy to represent the fraction
    of aggregator traffic in segments where the insurer is uncompetitive and
    therefore has no portfolio rows.

    Parameters
    ----------
    anchors : pd.DataFrame
        Anchor rows in engineered-feature space.
    n : int
        Number of profiles to return.
    method : str
        Profile generation method: ``"cp"`` for ceteris-paribus sweeps or
        ``"gaussian"`` for joint Gaussian perturbations.
    rng : np.random.Generator
        RNG for reproducibility (used by the Gaussian generator and sampling).
    gaussian_sigma_frac : float
        Noise width as a fraction of each feature's range (Gaussian method
        only; ignored for ``"cp"``).

    Returns
    -------
    pd.DataFrame
        Sampled profiles, index reset.  Empty DataFrame if pool generation
        produces no valid profiles.
    """
    if method == "cp":
        pool = generate_ceteris_paribus(anchors, validate=True)
    elif method == "gaussian":
        pool = generate_gaussian_profiles(
            anchors, sigma_frac=gaussian_sigma_frac, rng=rng, validate=True
        )
    else:
        raise ValueError(
            f"Unknown market_profile_method '{method}'. Valid: 'cp', 'gaussian'."
        )

    if len(pool) == 0:
        return pd.DataFrame(columns=anchors.columns)

    sample_idx = rng.choice(len(pool), size=n, replace=n > len(pool))
    return pool.iloc[sample_idx].reset_index(drop=True)
