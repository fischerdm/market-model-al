"""
Phase 2 — Active learning simulation
======================================
Runs the AL loop for all four query strategies and saves results.

Prerequisites:
  - notebooks/04_build_pool.py must have been run (outputs/pool/ must exist)

Outputs:
  outputs/al_results/results.parquet   — per-iteration metrics for all strategies
  outputs/figures/al_convergence_*.png — convergence plots (RMSE + SHAP similarity)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from market_model_al.features import load_raw, engineer_features
from market_model_al.oracle_engine import OraclePricingEngine
from market_model_al.al_loop import ALSimulation
from market_model_al.strategies import STRATEGIES

# ── Paths & config ─────────────────────────────────────────────────────────────

POOL_DIR    = ROOT / "outputs" / "pool"
RESULTS_DIR = ROOT / "outputs" / "al_results"
FIGURES_DIR = ROOT / "outputs" / "figures"
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WARM_START_N  = 10_000
BATCH_SIZE    = 500
N_ITERATIONS  = 40
SEED          = 42

# ── Load pool ──────────────────────────────────────────────────────────────────

print("Loading pool…")
pool_X = pd.read_parquet(POOL_DIR / "pool_X.parquet")
pool_y = np.load(POOL_DIR / "pool_y.npy")
print(f"  Pool: {len(pool_X):,} profiles × {pool_X.shape[1]} features\n")

# ── Load oracle ────────────────────────────────────────────────────────────────

oracle = OraclePricingEngine(ROOT / "outputs" / "models" / "oracle.pkl")

# ── Run simulation ─────────────────────────────────────────────────────────────

sim = ALSimulation(oracle, pool_X, pool_y, seed=SEED)

all_results = []
for strategy in STRATEGIES:
    print(f"\n{'='*60}")
    df_strategy = sim.run(
        strategy=strategy,
        warm_start_n=WARM_START_N,
        batch_size=BATCH_SIZE,
        n_iterations=N_ITERATIONS,
    )
    all_results.append(df_strategy)

results = pd.concat(all_results, ignore_index=True)

# ── Save results ───────────────────────────────────────────────────────────────

results_path = RESULTS_DIR / "results.parquet"
results.to_parquet(results_path, index=False)
print(f"\nResults saved → {results_path}")

# ── Plot convergence ───────────────────────────────────────────────────────────

PALETTE = {
    "random":          "#888888",
    "uncertainty":     "#1f77b4",
    "error_based":     "#ff7f0e",
    "shap_divergence": "#2ca02c",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Active learning convergence — all strategies", fontsize=13)

for strategy, grp in results.groupby("strategy"):
    color = PALETTE.get(strategy, None)
    axes[0].plot(grp["n_labeled"], grp["rmse"], label=strategy, color=color, lw=2)
    axes[1].plot(grp["n_labeled"], grp["shap_cosine_similarity"],
                 label=strategy, color=color, lw=2)

axes[0].set_xlabel("Labeled profiles")
axes[0].set_ylabel("RMSE on holdout")
axes[0].set_title("Premium prediction error")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Labeled profiles")
axes[1].set_ylabel("Mean cosine similarity")
axes[1].set_title("SHAP structure recovery")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = FIGURES_DIR / "al_convergence_all.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved → {fig_path}")
plt.close()

# ── Per-strategy RMSE plot ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
for strategy, grp in results.groupby("strategy"):
    color = PALETTE.get(strategy, None)
    ax.plot(grp["n_labeled"], grp["rel_rmse"], label=strategy, color=color, lw=2)

ax.set_xlabel("Labeled profiles")
ax.set_ylabel("Relative RMSE (RMSE / mean premium)")
ax.set_title("Relative RMSE by strategy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = FIGURES_DIR / "al_convergence_rel_rmse.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved → {fig_path}")
plt.close()

# ── Summary table ──────────────────────────────────────────────────────────────

final = results[results["iteration"] == N_ITERATIONS].set_index("strategy")
print("\nFinal metrics (iteration {}):\n".format(N_ITERATIONS))
print(
    final[["n_labeled", "rmse", "rel_rmse", "shap_cosine_similarity"]]
    .to_string(float_format="{:.4f}".format)
)
