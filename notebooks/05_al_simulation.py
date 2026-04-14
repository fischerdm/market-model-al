"""
Phase 2 — Active learning simulation
======================================
Runs the AL loop for all configured simulations and strategies, then saves
results for the dashboard.

Configuration
-------------
  config/simulation.yaml      — n_weeks, budget, strategies, metrics,
                                simulations (each with its own tariff-change
                                timeline)
  config/tariff_changes.yaml  — named perturbation library (type + params)

Prerequisites
-------------
  notebooks/04_build_warm_start.py must have been run first.

Outputs
-------
  outputs/al_results/results.parquet
  outputs/figures/al_<simulation_name>_rmse.png   (one per simulation)
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
from market_model_al.al_loop import ALSimulation
from market_model_al.config import (
    load_simulation_cfg,
    load_tariff_changes_cfg,
    resolve_simulations,
    build_perturbed_oracle,
)

# ── Load config ────────────────────────────────────────────────────────────────

sim_cfg    = load_simulation_cfg(ROOT / "config" / "simulation.yaml")
tc_library = load_tariff_changes_cfg(ROOT / "config" / "tariff_changes.yaml")
simulations = resolve_simulations(sim_cfg, tc_library)

N_WEEKS:           int       = sim_cfg["n_weeks"]
WEEKLY_BUDGET:     int       = sim_cfg["weekly_budget"]
CANDIDATE_MULT:    int       = sim_cfg["candidate_multiplier"]
SEED:              int       = sim_cfg["seed"]
STRATEGIES_RUN:    list[str] = sim_cfg["strategies"]
RESTART_STRATS:    list[str] = sim_cfg["restart_strategies"]
COMPUTE_SHAP:      bool      = sim_cfg["compute_shap_similarity"]
RM_N_CP_ANCHORS:   int       = sim_cfg["random_market_n_cp_anchors"]
MARKET_CP_RATIO:   float     = sim_cfg["market_cp_ratio"]
GAUSSIAN_SIGMA:    float     = sim_cfg["gaussian_sigma_frac"]

print("Simulation config:")
print(f"  n_weeks={N_WEEKS}  weekly_budget={WEEKLY_BUDGET}  "
      f"candidate_multiplier={CANDIDATE_MULT}  seed={SEED}")
print(f"  strategies        : {STRATEGIES_RUN}")
print(f"  restart_strategies: {RESTART_STRATS}")
print(f"  metrics           : {sorted(sim_cfg['metrics'])}")
print(f"  SHAP similarity   : {'enabled' if COMPUTE_SHAP else 'disabled'}")
print(f"  market_cp_ratio   : {MARKET_CP_RATIO}  (warm start + random_market)")
print(f"  random_market     : n_cp_anchors={RM_N_CP_ANCHORS}")
print(f"  gaussian_sigma    : {GAUSSIAN_SIGMA}")
print(f"\nSimulations ({len(simulations)}):")
for s in simulations:
    if s["has_tariff_changes"]:
        changes = ", ".join(f"week {w}" for w, _ in s["tariff_changes"])
        print(f"  [{s['name']}]  changes at: {changes}  —  {s['label']}")
    else:
        print(f"  [{s['name']}]  baseline  —  {s['label']}")
print()

# ── Paths ──────────────────────────────────────────────────────────────────────

WARM_DIR    = ROOT / "outputs" / "warm_start"
RESULTS_DIR = ROOT / "outputs" / "al_results"
FIGURES_DIR = ROOT / "outputs" / "figures"
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PALETTE = {
    # CP strategies
    "random_cp":                    "#888888",
    "random_market":                "#17becf",
    "uncertainty_cp":               "#1f77b4",
    "error_based_cp":               "#ff7f0e",
    "segment_adaptive_cp":          "#9467bd",
    "disruption_cp":                "#2ca02c",
    "random_cp_restart":            "#bbbbbb",
    "segment_adaptive_cp_restart":  "#c5b0d5",
    # Gaussian variants — lighter tints of their CP counterparts
    "random_gauss":                 "#cccccc",
    "uncertainty_gauss":            "#aec7e8",
    "error_based_gauss":            "#ffbb78",
    "segment_adaptive_gauss":       "#c5b0d5",
    "disruption_gauss":             "#98df8a",
}

# ── Load warm start ────────────────────────────────────────────────────────────

print("Loading warm start...")
warm_start_X = pd.read_parquet(WARM_DIR / "warm_start_X.parquet")
warm_start_y = np.load(WARM_DIR / "warm_start_y.npy")
print(f"  {len(warm_start_X):,} profiles\n")

# ── Load real data ─────────────────────────────────────────────────────────────

print("Loading real data...")
raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df  = engineer_features(raw)
TARGET   = "Premium"
FEATURES = [c for c in df.columns if c != TARGET]
real_X   = df[FEATURES]
print(f"  {len(real_X):,} rows\n")

# ── Oracle ─────────────────────────────────────────────────────────────────────

oracle = OraclePricingEngine(ROOT / "outputs" / "models" / "oracle.pkl")

# ── Build simulation object ────────────────────────────────────────────────────

sim = ALSimulation(
    oracle, real_X, seed=SEED,
    compute_shap_similarity=COMPUTE_SHAP,
)


# ── Helper: single strategy run ───────────────────────────────────────────────

def _run(strategy, sim_name, tc_pairs, restart=False, strategy_label=None):
    df_run = sim.run(
        strategy=strategy,
        warm_start_X=warm_start_X,
        warm_start_y=warm_start_y,
        weekly_budget=WEEKLY_BUDGET,
        candidate_multiplier=CANDIDATE_MULT,
        n_weeks=N_WEEKS,
        tariff_changes=tc_pairs or None,
        restart_at_tariff_change=restart,
        random_market_n_cp_anchors=RM_N_CP_ANCHORS,
        market_cp_ratio=MARKET_CP_RATIO,
        gaussian_sigma_frac=GAUSSIAN_SIGMA,
    )
    df_run["simulation"] = sim_name
    if strategy_label is not None:
        df_run["strategy"] = strategy_label
    return df_run


# ── Run all simulations ────────────────────────────────────────────────────────

all_frames = []

for simulation in simulations:
    sim_name  = simulation["name"]
    sim_label = simulation["label"]
    has_tc    = simulation["has_tariff_changes"]

    # Build (week, perturbed_oracle) pairs for this simulation's timeline
    tc_pairs = [
        (week, build_perturbed_oracle(oracle, perturb_cfg))
        for week, perturb_cfg in simulation["tariff_changes"]
    ]

    print("\n" + "=" * 60)
    print(f"SIMULATION: {sim_label}")
    print("=" * 60)

    # Continuous runs — all strategies
    for strategy in STRATEGIES_RUN:
        print(f"\n--- {strategy} ---")
        all_frames.append(_run(strategy, sim_name, tc_pairs))

    # Restart runs — only for simulations that have tariff changes
    if has_tc:
        for strategy in RESTART_STRATS:
            label = f"{strategy}_restart"
            print(f"\n--- {label} ---")
            all_frames.append(_run(strategy, sim_name, tc_pairs,
                                   restart=True, strategy_label=label))

# ── Save all results ───────────────────────────────────────────────────────────

results = pd.concat(all_frames, ignore_index=True)
results_path = RESULTS_DIR / "results.parquet"
results.to_parquet(results_path, index=False)
print(f"\nAll results saved -> {results_path}")

# ── Plotting ───────────────────────────────────────────────────────────────────

def _color(strategy):
    return PALETTE.get(strategy, "#333333")


for simulation in simulations:
    sim_name  = simulation["name"]
    sim_label = simulation["label"]
    tc_weeks  = [w for w, _ in simulation["tariff_changes"]]

    df_sim = results[results["simulation"] == sim_name]

    n_cols = 2 if COMPUTE_SHAP else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(sim_label, fontsize=13)

    for strategy, grp in df_sim.groupby("strategy"):
        axes[0].plot(grp["week"], grp["rmse"],
                     label=strategy, color=_color(strategy), lw=2)
        if COMPUTE_SHAP:
            axes[1].plot(grp["week"], grp["shap_cosine_similarity"],
                         label=strategy, color=_color(strategy), lw=2)

    for ax in axes:
        for tc_week in tc_weeks:
            ax.axvline(tc_week, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Week")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("RMSE on holdout")
    axes[0].set_title("Premium prediction error")
    if COMPUTE_SHAP:
        axes[1].set_ylabel("Mean cosine similarity")
        axes[1].set_title("SHAP structure recovery")

    plt.tight_layout()
    p = FIGURES_DIR / f"al_{sim_name}_rmse.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {p}")
    plt.close()

# ── Summary tables ─────────────────────────────────────────────────────────────

metric_cols = ["n_labeled", "rmse", "rel_rmse"]
if COMPUTE_SHAP:
    metric_cols.append("shap_cosine_similarity")

for simulation in simulations:
    sim_name = simulation["name"]
    df_sim   = results[results["simulation"] == sim_name]
    print(f"\nFinal metrics — {simulation['label']} (week {N_WEEKS}):")
    print(
        df_sim[df_sim["week"] == N_WEEKS]
        .set_index("strategy")[metric_cols]
        .to_string(float_format="{:.4f}".format)
    )
