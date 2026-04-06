"""
Phase 2 — Build and label the candidate profile pool
======================================================
Generates ceteris-paribus profiles from a random sample of anchor rows,
queries the oracle for every valid profile, and saves the result.

Outputs:
  outputs/pool/pool_X.parquet   — engineered feature profiles (no target)
  outputs/pool/pool_y.npy       — oracle-predicted premiums (aligned 1-D array)

This is a one-time step.  Re-run only if you want to regenerate the pool
with different anchors or sweep grids.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from market_model_al.features import load_raw, engineer_features
from market_model_al.oracle_engine import OraclePricingEngine
from market_model_al.profile_generator import generate_ceteris_paribus

# ── Config ─────────────────────────────────────────────────────────────────────

N_ANCHORS = 1_500   # ~1500 × 254 steps ≈ 380k rows before validation filter
SEED      = 42

POOL_DIR = ROOT / "outputs" / "pool"
POOL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data & oracle ─────────────────────────────────────────────────────────

print("Loading data and oracle…")
raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df  = engineer_features(raw)

TARGET   = "Premium"
FEATURES = [c for c in df.columns if c != TARGET]

engine = OraclePricingEngine(ROOT / "outputs" / "models" / "oracle.pkl")
print("Oracle loaded.\n")

# ── Sample anchors ─────────────────────────────────────────────────────────────

rng     = np.random.default_rng(SEED)
anchors = df[FEATURES].sample(n=N_ANCHORS, random_state=SEED)
print(f"Anchor set: {len(anchors):,} rows")

# ── Generate ceteris-paribus profiles ─────────────────────────────────────────

print("Generating ceteris-paribus profiles (this may take a minute)…")
pool_X = generate_ceteris_paribus(anchors, validate=True)
print(f"  Profiles after validation filter: {len(pool_X):,} rows\n")

# ── Oracle-label every profile ─────────────────────────────────────────────────

print("Querying oracle for all profiles…")
pool_y = engine.query(pool_X)
print(f"  Done.  Premium range: [{pool_y.min():.2f}, {pool_y.max():.2f}]  "
      f"mean={pool_y.mean():.2f}\n")

# ── Save ───────────────────────────────────────────────────────────────────────

pool_X_path = POOL_DIR / "pool_X.parquet"
pool_y_path = POOL_DIR / "pool_y.npy"

pool_X.to_parquet(pool_X_path, index=False)
np.save(pool_y_path, pool_y)

print(f"Saved pool_X → {pool_X_path}  ({pool_X_path.stat().st_size / 1e6:.1f} MB)")
print(f"Saved pool_y → {pool_y_path}  ({pool_y_path.stat().st_size / 1e6:.1f} MB)")
print(f"\nPool summary:")
print(f"  Rows   : {len(pool_X):,}")
print(f"  Columns: {pool_X.shape[1]}")
print(f"  Dtypes :")
for dtype, cols in pool_X.dtypes.groupby(pool_X.dtypes):
    print(f"    {dtype}: {list(cols.index)}")
