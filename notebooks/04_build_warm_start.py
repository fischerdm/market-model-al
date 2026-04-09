"""
Phase 2 — Build warm start dataset
====================================
Generates and oracle-labels the warm start used to seed the AL simulation.

The warm start has two components — mirroring how a real competitor model
would be initialised:

  (A) Real rows  — a random sample from the Spanish dataset, simulating
                   organic quote requests that have arrived via comparis.
                   These are already "real" policy profiles with natural
                   feature correlations.

  (B) CP profiles — ceteris-paribus sweeps from a small number of anchor rows,
                    simulating a structured initial scraping run where the
                    insurer probes the competitor tariff systematically before
                    setting up the continuous loop.

Both components are oracle-labeled and saved together.

Outputs
-------
  outputs/warm_start/warm_start_X.parquet  — engineered feature profiles
  outputs/warm_start/warm_start_y.npy      — oracle-predicted premiums (aligned)
  outputs/warm_start/metadata.json         — mix config for reproducibility

Run this script once before running 05_al_simulation.py.
Re-run with different CONFIG values to experiment with warm-start mix ratios.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from market_model_al.features import load_raw, engineer_features
from market_model_al.oracle_engine import OraclePricingEngine
from market_model_al.profile_generator import generate_ceteris_paribus

# ── Config (edit to experiment with different warm-start mixes) ───────────────

CONFIG = dict(
    real_rows_n   = 10_000,   # (A) real policy rows  (~10 % of the dataset)
    cp_anchors_n  = 0,       # (B) CP anchors  →  50 × ~254 steps ≈ 12 k profiles
    seed          = 42,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

WARM_DIR = ROOT / "outputs" / "warm_start"
WARM_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data & oracle ─────────────────────────────────────────────────────────

print("Loading data and oracle…")
raw    = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df     = engineer_features(raw)
TARGET = "Premium"
FEATURES = [c for c in df.columns if c != TARGET]

engine = OraclePricingEngine(ROOT / "outputs" / "models" / "oracle.pkl")
print(f"  Dataset: {len(df):,} rows × {len(FEATURES)} features")
print("  Oracle loaded.\n")

rng = np.random.default_rng(CONFIG["seed"])

# ── Component A: real rows ─────────────────────────────────────────────────────

print(f"(A) Sampling {CONFIG['real_rows_n']:,} real policy rows…")
real_sample = df[FEATURES].sample(n=CONFIG["real_rows_n"], random_state=CONFIG["seed"])

valid_mask = engine.validate(real_sample)
n_invalid  = (~valid_mask).sum()
real_sample = real_sample[valid_mask].reset_index(drop=True)
print(f"    {n_invalid} rows dropped (data-quality violations) → {len(real_sample):,} kept")

real_y = engine.query(real_sample)
print(f"    Oracle labels: mean={real_y.mean():.2f}  range=[{real_y.min():.2f}, {real_y.max():.2f}]\n")

# ── Component B: ceteris-paribus profiles ──────────────────────────────────────

if CONFIG["cp_anchors_n"] > 0:
    print(f"(B) Generating CP profiles from {CONFIG['cp_anchors_n']} anchors…")
    anchors = df[FEATURES].sample(n=CONFIG["cp_anchors_n"], random_state=CONFIG["seed"] + 1)
    cp_profiles = generate_ceteris_paribus(anchors, validate=True)
    print(f"    {len(cp_profiles):,} valid CP profiles generated")
    cp_y = engine.query(cp_profiles)
    print(f"    Oracle labels: mean={cp_y.mean():.2f}  range=[{cp_y.min():.2f}, {cp_y.max():.2f}]\n")
else:
    print("(B) cp_anchors_n = 0 — skipping CP profiles (real rows only warm start)\n")
    cp_profiles = pd.DataFrame(columns=real_sample.columns)
    cp_y        = np.array([], dtype=float)

# ── Combine ────────────────────────────────────────────────────────────────────

warm_start_X = pd.concat([real_sample, cp_profiles], ignore_index=True)
warm_start_y = np.concatenate([real_y, cp_y])

print(f"Warm start combined: {len(warm_start_X):,} profiles")
print(f"  Real rows : {len(real_sample):,}  ({100*len(real_sample)/len(warm_start_X):.1f} %)")
print(f"  CP profiles: {len(cp_profiles):,}  ({100*len(cp_profiles)/len(warm_start_X):.1f} %)\n")

# ── Save ───────────────────────────────────────────────────────────────────────

X_path    = WARM_DIR / "warm_start_X.parquet"
y_path    = WARM_DIR / "warm_start_y.npy"
meta_path = WARM_DIR / "metadata.json"

warm_start_X.to_parquet(X_path, index=False)
np.save(y_path, warm_start_y)

metadata = {
    **CONFIG,
    "n_total"       : int(len(warm_start_X)),
    "n_real"        : int(len(real_sample)),
    "n_cp"          : int(len(cp_profiles)),
    "premium_mean"  : float(warm_start_y.mean()),
    "premium_min"   : float(warm_start_y.min()),
    "premium_max"   : float(warm_start_y.max()),
}
meta_path.write_text(json.dumps(metadata, indent=2))

print(f"Saved → {X_path}  ({X_path.stat().st_size / 1e6:.1f} MB)")
print(f"Saved → {y_path}  ({y_path.stat().st_size / 1e6:.1f} MB)")
print(f"Saved → {meta_path}")
