"""
Phase 2 — Active learning simulation
======================================
Runs the AL loop for multiple strategies and saves results for the dashboard.

Prerequisites
-------------
  notebooks/04_build_warm_start.py must have been run first.

Scenarios
---------
  1. All four strategies, no tariff change  — baseline strategy comparison
  2. Young-driver surcharge at week 26      — tariff change recovery
     Per strategy: one "continue" run (no restart) so practitioners can see
     how each strategy handles drift.

Outputs
-------
  outputs/al_results/results.parquet          — all per-week metrics
  outputs/figures/al_convergence_rmse.png
  outputs/figures/al_convergence_shap.png
  outputs/figures/al_tariff_change_rmse.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from market_model_al.features import load_raw, engineer_features
from market_model_al.oracle_engine import OraclePricingEngine
from market_model_al.perturbed_oracle import PerturbedOracleEngine, young_driver_surcharge
from market_model_al.al_loop import ALSimulation
from market_model_al.strategies import STRATEGIES

# ── Paths & config ─────────────────────────────────────────────────────────────

WARM_DIR    = ROOT / "outputs" / "warm_start"
RESULTS_DIR = ROOT / "outputs" / "al_results"
FIGURES_DIR = ROOT / "outputs" / "figures"
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PROFILES_PER_WEEK  = 100
N_ANCHORS_PER_WEEK = 8     # 8 anchors × ~254 steps ≈ 2 000 candidates/week
N_WEEKS            = 52
TARIFF_CHANGE_WEEK = 26
SEED               = 42

PALETTE = {
    "random":          "#888888",
    "uncertainty":     "#1f77b4",
    "error_based":     "#ff7f0e",
    "shap_divergence": "#2ca02c",
}

# ── Load warm start ────────────────────────────────────────────────────────────

print("Loading warm start...")
warm_start_X = pd.read_parquet(WARM_DIR / "warm_start_X.parquet")
warm_start_y = np.load(WARM_DIR / "warm_start_y.npy")
print(f"  {len(warm_start_X):,} profiles\n")

# ── Load real data (anchor pool + holdout source) ──────────────────────────────

print("Loading real data...")
raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df  = engineer_features(raw)
TARGET   = "Premium"
FEATURES = [c for c in df.columns if c != TARGET]
real_X   = df[FEATURES]
print(f"  {len(real_X):,} rows\n")

# ── Oracles ────────────────────────────────────────────────────────────────────

oracle    = OraclePricingEngine(ROOT / "outputs" / "models" / "oracle.pkl")
perturbed = PerturbedOracleEngine(oracle, young_driver_surcharge(factor=0.20))

# ── Build simulation object ────────────────────────────────────────────────────

sim = ALSimulation(oracle, real_X, seed=SEED)

# ── Scenario 1: all strategies, no tariff change ──────────────────────────────

print("\n" + "=" * 60)
print("SCENARIO 1 -- Strategy comparison (no tariff change)")
print("=" * 60)

results_s1 = []
for strategy in STRATEGIES:
    print(f"\n--- {strategy} ---")
    df_run = sim.run(
        strategy=strategy,
        warm_start_X=warm_start_X,
        warm_start_y=warm_start_y,
        profiles_per_week=PROFILES_PER_WEEK,
        n_anchors_per_week=N_ANCHORS_PER_WEEK,
        n_weeks=N_WEEKS,
    )
    df_run["scenario"] = "no_tariff_change"
    results_s1.append(df_run)

# ── Scenario 2: tariff change at week 26 (all strategies, no restart) ─────────

print("\n" + "=" * 60)
print(f"SCENARIO 2 -- Tariff change at week {TARIFF_CHANGE_WEEK} (young-driver +20 %)")
print("=" * 60)

results_s2 = []
for strategy in STRATEGIES:
    print(f"\n--- {strategy} + tariff change ---")
    df_run = sim.run(
        strategy=strategy,
        warm_start_X=warm_start_X,
        warm_start_y=warm_start_y,
        profiles_per_week=PROFILES_PER_WEEK,
        n_anchors_per_week=N_ANCHORS_PER_WEEK,
        n_weeks=N_WEEKS,
        tariff_change_week=TARIFF_CHANGE_WEEK,
        perturbed_oracle=perturbed,
    )
    df_run["scenario"] = "tariff_change"
    results_s2.append(df_run)

# ── Save all results ───────────────────────────────────────────────────────────

results = pd.concat(results_s1 + results_s2, ignore_index=True)
results_path = RESULTS_DIR / "results.parquet"
results.to_parquet(results_path, index=False)
print(f"\nAll results saved -> {results_path}")

# ── Plot: scenario 1 -- RMSE & SHAP convergence ───────────────────────────────

s1 = results[results["scenario"] == "no_tariff_change"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Strategy comparison -- no tariff change", fontsize=13)

for strategy, grp in s1.groupby("strategy"):
    c = PALETTE.get(strategy)
    axes[0].plot(grp["week"], grp["rmse"],                   label=strategy, color=c, lw=2)
    axes[1].plot(grp["week"], grp["shap_cosine_similarity"],  label=strategy, color=c, lw=2)

for ax, ylabel, title in [
    (axes[0], "RMSE on holdout",        "Premium prediction error"),
    (axes[1], "Mean cosine similarity",  "SHAP structure recovery"),
]:
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
p = FIGURES_DIR / "al_convergence_rmse.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Figure saved -> {p}")
plt.close()

# ── Plot: scenario 2 -- tariff change RMSE ────────────────────────────────────

s2 = results[results["scenario"] == "tariff_change"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title(f"Tariff change at week {TARIFF_CHANGE_WEEK} -- RMSE recovery by strategy")

for strategy, grp in s2.groupby("strategy"):
    ax.plot(grp["week"], grp["rmse"], label=strategy, color=PALETTE.get(strategy), lw=2)

ax.axvline(TARIFF_CHANGE_WEEK, color="red", linestyle="--", lw=1.5, label="tariff change")
ax.set_xlabel("Week")
ax.set_ylabel("RMSE on holdout (vs new oracle)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
p = FIGURES_DIR / "al_tariff_change_rmse.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Figure saved -> {p}")
plt.close()

# ── Summary table ──────────────────────────────────────────────────────────────

print(f"\nFinal metrics -- scenario 1 (week {N_WEEKS}):")
final_s1 = (
    s1[s1["week"] == N_WEEKS]
    .set_index("strategy")[["n_labeled", "rmse", "rel_rmse", "shap_cosine_similarity"]]
)
print(final_s1.to_string(float_format="{:.4f}".format))

print(f"\nFinal metrics -- scenario 2 (week {N_WEEKS}):")
final_s2 = (
    s2[s2["week"] == N_WEEKS]
    .set_index("strategy")[["n_labeled", "rmse", "rel_rmse", "shap_cosine_similarity"]]
)
print(final_s2.to_string(float_format="{:.4f}".format))
