"""
Smoke test — profile generator
================================
Verifies that:
  1. Output shape is as expected: N anchors × F features × S steps (before
     validation filtering).
  2. Ceteris-paribus property: for each generated block, only the swept
     feature varies; all other columns are identical across the block.
  3. validate=True (default) drops physically invalid rows.
  4. validate=False keeps all rows including invalid ones.
  5. feature_ranges override replaces the default grid for that feature.
  6. Categorical dtypes are preserved in the output.
  7. End-to-end: generated profiles can be queried through the oracle engine.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from market_model_al.features import load_raw, engineer_features
from market_model_al.oracle_engine import OraclePricingEngine
from market_model_al.profile_generator import (
    CONTINUOUS_FEATURES,
    DEFAULT_RANGES,
    generate_ceteris_paribus,
)

# ── Setup ──────────────────────────────────────────────────────────────────────

raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df  = engineer_features(raw)

TARGET   = "Premium"
FEATURES = [c for c in df.columns if c != TARGET]

N_ANCHORS = 5
anchors = df[FEATURES].sample(n=N_ANCHORS, random_state=42)
sweep_features = [f for f in CONTINUOUS_FEATURES if f in anchors.columns]

print(f"Anchors: {N_ANCHORS} rows, {len(sweep_features)} continuous features to sweep.\n")

# ── Test 1: output shape ───────────────────────────────────────────────────────
# Expected row count before validation = N_anchors × sum of grid sizes.

expected_before_validation = N_ANCHORS * sum(len(DEFAULT_RANGES[f]) for f in sweep_features)
profiles_unvalidated = generate_ceteris_paribus(anchors, validate=False)

assert len(profiles_unvalidated) == expected_before_validation, (
    f"Expected {expected_before_validation} rows before validation, "
    f"got {len(profiles_unvalidated)}."
)
print(f"Test 1 PASSED — {len(profiles_unvalidated):,} rows generated before validation "
      f"({N_ANCHORS} anchors × {len(sweep_features)} features).\n")

# ── Test 2: ceteris-paribus property ──────────────────────────────────────────
# For a single anchor, pick the driver_age sweep block and verify that every
# column except driver_age is constant across all rows in the block.

anchor_row = anchors.iloc[[0]]
single = generate_ceteris_paribus(anchor_row, validate=False)

driver_age_steps = len(DEFAULT_RANGES["driver_age"])
block = single.iloc[:driver_age_steps]  # first block = driver_age sweep

other_cols = [c for c in block.columns if c != "driver_age"]
for col in other_cols:
    unique_vals = block[col].nunique()
    assert unique_vals == 1, (
        f"Column '{col}' should be constant in the driver_age sweep block, "
        f"but has {unique_vals} distinct values."
    )

# The swept feature itself must have the expected grid values
np.testing.assert_array_almost_equal(
    block["driver_age"].values,
    DEFAULT_RANGES["driver_age"],
    err_msg="driver_age sweep values don't match DEFAULT_RANGES.",
)
print("Test 2 PASSED — ceteris-paribus property holds (only swept feature varies).\n")

# ── Test 3: validate=True drops invalid rows ───────────────────────────────────
# Sweeping licence_age across 0–50 for a young anchor (driver_age ~25) will
# produce rows where licence_age > driver_age - 18, which must be filtered.

young_anchor = df[FEATURES][df["driver_age"] < 30].iloc[[0]]
profiles_validated   = generate_ceteris_paribus(young_anchor, validate=True)
profiles_unvalidated = generate_ceteris_paribus(young_anchor, validate=False)

n_dropped = len(profiles_unvalidated) - len(profiles_validated)
assert n_dropped > 0, (
    "Expected some rows to be dropped for a young driver sweeping licence_age, "
    "but none were."
)
print(f"Test 3 PASSED — {n_dropped} invalid rows dropped by validate=True "
      f"(young anchor, driver_age={young_anchor['driver_age'].iloc[0]:.1f}).\n")

# ── Test 4: validate=False keeps everything ────────────────────────────────────

assert len(profiles_unvalidated) > len(profiles_validated), (
    "validate=False should return more rows than validate=True for a young anchor."
)
print("Test 4 PASSED — validate=False retains all rows including invalid ones.\n")

# ── Test 5: feature_ranges override ───────────────────────────────────────────

custom_grid = np.array([18.0, 30.0, 50.0, 70.0])
custom = generate_ceteris_paribus(
    anchor_row,
    feature_ranges={"driver_age": custom_grid},
    validate=False,
)
# The driver_age block should now have exactly 4 steps
driver_age_block = custom.iloc[:len(custom_grid)]
np.testing.assert_array_almost_equal(
    driver_age_block["driver_age"].values,
    custom_grid,
    err_msg="Custom driver_age grid not applied correctly.",
)
print("Test 5 PASSED — feature_ranges override applied correctly.\n")

# ── Test 6: categorical dtypes preserved ──────────────────────────────────────

cat_cols_in  = list(anchors.select_dtypes("category").columns)
cat_cols_out = list(profiles_validated.select_dtypes("category").columns)

assert set(cat_cols_in) == set(cat_cols_out), (
    f"Categorical columns changed.\n  Input:  {cat_cols_in}\n  Output: {cat_cols_out}"
)
print(f"Test 6 PASSED — categorical dtypes preserved ({len(cat_cols_in)} columns).\n")

# ── Test 7: end-to-end with oracle engine ─────────────────────────────────────

engine   = OraclePricingEngine(ROOT / "outputs" / "models" / "oracle.pkl")
profiles = generate_ceteris_paribus(anchors, validate=True)
prices   = engine.query(profiles)

assert prices.shape == (len(profiles),), (
    f"Expected {len(profiles)} prices, got {prices.shape}."
)
assert (prices > 0).all(), "All predicted premiums should be positive."
print(f"Test 7 PASSED — oracle engine queried on {len(profiles):,} generated profiles, "
      f"all prices positive (mean: {prices.mean():.2f}).\n")

print("All smoke tests passed.")
