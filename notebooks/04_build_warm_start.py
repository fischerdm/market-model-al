"""
Phase 2 — Build warm start dataset
====================================
Generates and oracle-labels the warm start used to seed the AL simulation.

Composition mirrors the market_cp_ratio assumption from simulation.yaml:
  - (1 − market_cp_ratio) × weekly_budget real portfolio rows
  - market_cp_ratio × weekly_budget rows sampled from a CP pool

The CP pool is built from random_market.n_cp_anchors anchor rows using
ceteris-paribus sweeps, then sampled — the same mechanism used by the
random_market strategy each week.  This ensures the warm start reflects the
same market composition assumption as the ongoing AL loop.

Outputs
-------
  outputs/warm_start/warm_start_X.parquet  — engineered feature profiles
  outputs/warm_start/warm_start_y.npy      — oracle-predicted premiums (aligned)
  outputs/warm_start/metadata.json         — mix config for reproducibility

Run this script once before running 05_al_simulation.py.
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
from market_model_al.config import load_simulation_cfg

# ── Load config from simulation.yaml ─────────────────────────────────────────

sim_cfg = load_simulation_cfg(ROOT / "config" / "simulation.yaml")

SEED             = sim_cfg["seed"]
WEEKLY_BUDGET    = sim_cfg["weekly_budget"]
MARKET_CP_RATIO  = sim_cfg["market_cp_ratio"]
N_CP_ANCHORS     = sim_cfg["random_market_n_cp_anchors"]

N_CP_SAMPLE  = max(0, int(WEEKLY_BUDGET * MARKET_CP_RATIO))
N_REAL       = WEEKLY_BUDGET - N_CP_SAMPLE

print("Warm start config (from simulation.yaml):")
print(f"  weekly_budget={WEEKLY_BUDGET}  market_cp_ratio={MARKET_CP_RATIO}")
print(f"  → {N_REAL:,} real rows + {N_CP_SAMPLE:,} CP rows  (n_cp_anchors={N_CP_ANCHORS})\n")

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

rng = np.random.default_rng(SEED)

# ── Component A: real rows ─────────────────────────────────────────────────────

print(f"(A) Sampling {N_REAL:,} real portfolio rows…")
real_sample = df[FEATURES].sample(n=N_REAL, random_state=SEED)

valid_mask = engine.validate(real_sample)
n_invalid  = (~valid_mask).sum()
real_sample = real_sample[valid_mask].reset_index(drop=True)
print(f"    {n_invalid} rows dropped (data-quality violations) → {len(real_sample):,} kept")

real_y = engine.query(real_sample)
print(f"    Oracle labels: mean={real_y.mean():.2f}  range=[{real_y.min():.2f}, {real_y.max():.2f}]\n")

# ── Component B: CP profiles sampled from pool ────────────────────────────────

if N_CP_SAMPLE > 0:
    print(f"(B) Building CP pool from {N_CP_ANCHORS} anchors, sampling {N_CP_SAMPLE:,} rows…")
    anchors    = df[FEATURES].sample(n=N_CP_ANCHORS, random_state=SEED + 1)
    cp_pool    = generate_ceteris_paribus(anchors, validate=True)
    print(f"    CP pool size: {len(cp_pool):,} valid profiles")
    sample_idx = rng.choice(
        len(cp_pool),
        size=min(N_CP_SAMPLE, len(cp_pool)),
        replace=N_CP_SAMPLE > len(cp_pool),
    )
    cp_sample  = cp_pool.iloc[sample_idx].reset_index(drop=True)
    cp_y       = engine.query(cp_sample)
    print(f"    Oracle labels: mean={cp_y.mean():.2f}  range=[{cp_y.min():.2f}, {cp_y.max():.2f}]\n")
else:
    print("(B) market_cp_ratio = 0 — skipping CP profiles (real rows only)\n")
    cp_sample = pd.DataFrame(columns=real_sample.columns)
    cp_y      = np.array([], dtype=float)

# ── Combine ────────────────────────────────────────────────────────────────────

warm_start_X = pd.concat([real_sample, cp_sample], ignore_index=True)
warm_start_y = np.concatenate([real_y, cp_y])

print(f"Warm start combined: {len(warm_start_X):,} profiles")
print(f"  Real rows  : {len(real_sample):,}  ({100*len(real_sample)/len(warm_start_X):.1f} %)")
print(f"  CP profiles: {len(cp_sample):,}  ({100*len(cp_sample)/len(warm_start_X):.1f} %)\n")

# ── Save ───────────────────────────────────────────────────────────────────────

X_path    = WARM_DIR / "warm_start_X.parquet"
y_path    = WARM_DIR / "warm_start_y.npy"
meta_path = WARM_DIR / "metadata.json"

warm_start_X.to_parquet(X_path, index=False)
np.save(y_path, warm_start_y)

metadata = {
    "weekly_budget"   : WEEKLY_BUDGET,
    "market_cp_ratio" : MARKET_CP_RATIO,
    "n_cp_anchors"    : N_CP_ANCHORS,
    "seed"            : SEED,
    "n_total"         : int(len(warm_start_X)),
    "n_real"          : int(len(real_sample)),
    "n_cp"            : int(len(cp_sample)),
    "premium_mean"    : float(warm_start_y.mean()),
    "premium_min"     : float(warm_start_y.min()),
    "premium_max"     : float(warm_start_y.max()),
}
meta_path.write_text(json.dumps(metadata, indent=2))

print(f"Saved → {X_path}  ({X_path.stat().st_size / 1e6:.1f} MB)")
print(f"Saved → {y_path}  ({y_path.stat().st_size / 1e6:.1f} MB)")
print(f"Saved → {meta_path}")
