"""
Smoke test — OraclePricingEngine
=================================
Verifies that:
  1. Engine loads and returns predictions for a valid profile.
  2. validate() returns True for valid profiles, False for each type of
     physical violation.
  3. query() raises ValueError on invalid profiles.
  4. Batch query on a sample of real rows (rows the oracle was trained on) returns
     predictions — verifying the engine loads correctly and wires up to the oracle.
     This is an in-sample sanity check only, not a test of generalisation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from market_model_al.features import CAT_FEATURES, CAT_FEATURES_OBJ, load_raw, engineer_features
from market_model_al.oracle_engine import OraclePricingEngine

# ── Setup ──────────────────────────────────────────────────────────────────────

engine = OraclePricingEngine(ROOT / "outputs" / "models" / "oracle.pkl")
print("Oracle loaded.\n")

# Use one real row as a valid baseline profile
raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df  = engineer_features(raw)

TARGET   = "Premium"
FEATURES = [c for c in df.columns if c != TARGET]

baseline = df[FEATURES].iloc[[0]].copy()
print("Baseline profile:")
print(baseline.T, "\n")

# ── Test 1: valid profile returns a positive prediction ────────────────────────

prices = engine.query(baseline)
assert prices.shape == (1,), f"Expected shape (1,), got {prices.shape}"
assert prices[0] > 0, f"Expected positive premium, got {prices[0]}"
print(f"Test 1 PASSED — predicted premium: {prices[0]:.2f}\n")

# ── Test 2: validate() correctly identifies each type of violation ─────────────

def make_invalid(col, value):
    """Return a copy of baseline with one field set to an invalid value."""
    p = baseline.copy()
    p[col] = value
    return p

violations = {
    "driver_age < 18":         make_invalid("driver_age", 15.0),
    "licence_age < 0":         make_invalid("licence_age", -1.0),
    "vehicle_age < 0":         make_invalid("vehicle_age", -1),
    "Power < 1":               make_invalid("Power", 0),
    "Cylinder_capacity < 1":   make_invalid("Cylinder_capacity", 0),
    "Value_vehicle < 1":       make_invalid("Value_vehicle", 0.0),
    "Seniority < 0":           make_invalid("Seniority", -1),
    "licence_age > driver_age - 18":
        make_invalid("licence_age", baseline["driver_age"].iloc[0] - 10.0),
}

all_passed = True
for label, profile in violations.items():
    mask = engine.validate(profile)
    ok = not mask.iloc[0]
    status = "PASSED" if ok else "FAILED"
    if not ok:
        all_passed = False
    print(f"  Test 2 [{status}] violation '{label}' detected: {not mask.iloc[0]}")

assert all_passed, "One or more violation checks failed."
print("\nTest 2 PASSED — all violations correctly detected.\n")

# ── Test 3: query() raises on invalid input ────────────────────────────────────

bad_profile = make_invalid("driver_age", 10.0)
try:
    engine.query(bad_profile)
    assert False, "Expected ValueError was not raised."
except ValueError as e:
    print(f"Test 3 PASSED — ValueError raised as expected: {e}\n")

# ── Test 4: batch query on real rows ──────────────────────────────────────────
# Filter out the few real rows with data-quality violations (e.g. negative
# licence_age) before querying; this is expected and intentional.

SAMPLE_N = 1_000
sample = df[FEATURES].sample(n=SAMPLE_N, random_state=42)
valid_mask = engine.validate(sample)
n_invalid = (~valid_mask).sum()
sample_valid = sample[valid_mask]

print(f"Batch query: {SAMPLE_N} sampled rows, {n_invalid} rejected by validate() (data-quality violations in raw data), {len(sample_valid)} queried.")

predicted  = engine.query(sample_valid)
actual     = df.loc[sample_valid.index, TARGET].values
rmse       = np.sqrt(np.mean((predicted - actual) ** 2))
rel_rmse   = rmse / actual.mean()

print(f"  Predicted vs actual — RMSE: {rmse:.2f}  |  Relative RMSE: {rel_rmse:.4f}")
# The oracle is a "reasonably good" tariff, not a perfect memoriser — insurance
# premiums have high natural variance. A threshold of 0.40 guards only against
# a clearly broken load (wrong model file, feature mismatch, etc.).
assert rel_rmse < 0.40, f"Relative RMSE too high ({rel_rmse:.4f}); oracle may not have loaded correctly."
print(f"Test 4 PASSED — batch predictions functional (relative RMSE {rel_rmse:.4f}).\n")

print("All smoke tests passed.")
