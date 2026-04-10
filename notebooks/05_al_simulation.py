"""
Phase 2 — Active learning simulation
======================================
Runs the AL loop for all configured strategies and scenarios, then saves
results for the dashboard.

Configuration
-------------
  config/simulation.yaml      — n_weeks, budget, strategies, metrics
  config/tariff_changes.yaml  — tariff-change scenarios (perturbation type,
                                 injection week, restart strategies)

Prerequisites
-------------
  notebooks/04_build_warm_start.py must have been run first.

Outputs
-------
  outputs/al_results/results.parquet
  outputs/figures/al_convergence_rmse.png   (scenario 1 only)
  outputs/figures/al_convergence_shap.png   (scenario 1, if SHAP enabled)
  outputs/figures/al_tariff_<name>_rmse.png (one per tariff-change scenario)
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
from market_model_al.strategies import STRATEGIES
from market_model_al.config import (
    load_simulation_cfg,
    load_tariff_changes_cfg,
    build_perturbed_oracle,
)

# ── Load config ────────────────────────────────────────────────────────────────

sim_cfg = load_simulation_cfg(ROOT / "config" / "simulation.yaml")
tc_list  = load_tariff_changes_cfg(ROOT / "config" / "tariff_changes.yaml")

N_WEEKS        = sim_cfg["n_weeks"]
WEEKLY_BUDGET  = sim_cfg["weekly_budget"]
CANDIDATE_MULT = sim_cfg["candidate_multiplier"]
SEED           = sim_cfg["seed"]
STRATEGIES_RUN = sim_cfg["strategies"]
COMPUTE_SHAP   = sim_cfg["compute_shap_similarity"]

print("Simulation config:")
print(f"  n_weeks={N_WEEKS}  weekly_budget={WEEKLY_BUDGET}  "
      f"candidate_multiplier={CANDIDATE_MULT}  seed={SEED}")
print(f"  strategies : {STRATEGIES_RUN}")
print(f"  metrics    : {sorted(sim_cfg['metrics'])}")
print(f"  SHAP sim   : {'enabled' if COMPUTE_SHAP else 'disabled'}")
print(f"\nTariff-change scenarios ({len(tc_list)}):")
for tc in tc_list:
    print(f"  [{tc['name']}]  week={tc['week']}  "
          f"restarts={tc['restart_strategies'] or 'none'}")
print()

# ── Paths ──────────────────────────────────────────────────────────────────────

WARM_DIR    = ROOT / "outputs" / "warm_start"
RESULTS_DIR = ROOT / "outputs" / "al_results"
FIGURES_DIR = ROOT / "outputs" / "figures"
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "random":                   "#888888",
    "uncertainty":              "#1f77b4",
    "error_based":              "#ff7f0e",
    "segment_adaptive":         "#9467bd",
    "disruption":               "#2ca02c",
    "random_restart":           "#bbbbbb",
    "segment_adaptive_restart": "#c5b0d5",
    "disruption_restart":       "#98df8a",
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


# ── Helper: run one strategy ──────────────────────────────────────────────────

def _run(strategy, scenario_name, tariff_change_week=None, perturbed=None,
         restart=False, strategy_label=None):
    df_run = sim.run(
        strategy=strategy,
        warm_start_X=warm_start_X,
        warm_start_y=warm_start_y,
        weekly_budget=WEEKLY_BUDGET,
        candidate_multiplier=CANDIDATE_MULT,
        n_weeks=N_WEEKS,
        tariff_change_week=tariff_change_week,
        perturbed_oracle=perturbed,
        restart_at_tariff_change=restart,
    )
    df_run["scenario"] = scenario_name
    if strategy_label is not None:
        df_run["strategy"] = strategy_label
    return df_run


# ── Scenario 1: no tariff change ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("SCENARIO 1 -- Strategy comparison (no tariff change)")
print("=" * 60)

results_s1 = []
for strategy in STRATEGIES_RUN:
    print(f"\n--- {strategy} ---")
    results_s1.append(_run(strategy, "no_tariff_change"))

# ── Scenarios 2+: one per tariff-change entry ─────────────────────────────────

all_tc_results: list[tuple[dict, list[pd.DataFrame]]] = []

for tc in tc_list:
    tc_name  = tc["name"]
    tc_label = tc["label"]
    tc_week  = tc["week"]
    perturbed = build_perturbed_oracle(oracle, tc["perturbation"])
    scenario_name = f"tariff_change_{tc_name}"

    print("\n" + "=" * 60)
    print(f"SCENARIO -- {tc_label}  (week {tc_week})")
    print("=" * 60)

    tc_results = []

    for strategy in STRATEGIES_RUN:
        print(f"\n--- {strategy} ---")
        tc_results.append(_run(
            strategy, scenario_name,
            tariff_change_week=tc_week, perturbed=perturbed,
        ))

    for strategy in tc["restart_strategies"]:
        label = f"{strategy}_restart"
        print(f"\n--- {label} ---")
        tc_results.append(_run(
            strategy, scenario_name,
            tariff_change_week=tc_week, perturbed=perturbed,
            restart=True, strategy_label=label,
        ))

    all_tc_results.append((tc, tc_results))

# ── Save all results ───────────────────────────────────────────────────────────

all_frames = results_s1 + [df for _, runs in all_tc_results for df in runs]
results = pd.concat(all_frames, ignore_index=True)
results_path = RESULTS_DIR / "results.parquet"
results.to_parquet(results_path, index=False)
print(f"\nAll results saved -> {results_path}")

# ── Plot helpers ───────────────────────────────────────────────────────────────

def _strategy_color(s):
    return PALETTE.get(s, "#333333")


def _plot_metric(ax, grp_data, metric, label_map=None):
    for strategy, grp in grp_data:
        label = label_map.get(strategy, strategy) if label_map else strategy
        ax.plot(grp["week"], grp[metric], label=label,
                color=_strategy_color(strategy), lw=2)
    ax.set_xlabel("Week")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ── Plot: scenario 1 — RMSE convergence ───────────────────────────────────────

s1 = results[results["scenario"] == "no_tariff_change"]
s1_groups = list(s1.groupby("strategy"))

n_cols = 2 if COMPUTE_SHAP else 1
fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
if n_cols == 1:
    axes = [axes]
fig.suptitle("Strategy comparison — no tariff change", fontsize=13)

_plot_metric(axes[0], s1_groups, "rmse")
axes[0].set_ylabel("RMSE on holdout")
axes[0].set_title("Premium prediction error")

if COMPUTE_SHAP:
    _plot_metric(axes[1], s1_groups, "shap_cosine_similarity")
    axes[1].set_ylabel("Mean cosine similarity")
    axes[1].set_title("SHAP structure recovery")

plt.tight_layout()
p = FIGURES_DIR / "al_convergence_rmse.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
print(f"Figure saved -> {p}")
plt.close()

# ── Plot: one RMSE figure per tariff-change scenario ─────────────────────────

for tc, tc_results in all_tc_results:
    tc_name  = tc["name"]
    tc_label = tc["label"]
    tc_week  = tc["week"]
    scenario_name = f"tariff_change_{tc_name}"

    s2 = results[results["scenario"] == scenario_name]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"{tc_label}  (tariff change at week {tc_week})")

    for strategy, grp in s2.groupby("strategy"):
        ax.plot(grp["week"], grp["rmse"], label=strategy,
                color=_strategy_color(strategy), lw=2)

    ax.axvline(tc_week, color="red", linestyle="--", lw=1.5, label="tariff change")
    ax.set_xlabel("Week")
    ax.set_ylabel("RMSE on holdout (vs new oracle)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = FIGURES_DIR / f"al_tariff_{tc_name}_rmse.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {p}")
    plt.close()

# ── Summary tables ─────────────────────────────────────────────────────────────

metric_cols = ["n_labeled", "rmse", "rel_rmse"]
if COMPUTE_SHAP:
    metric_cols.append("shap_cosine_similarity")

print(f"\nFinal metrics — scenario 1 (week {N_WEEKS}):")
final_s1 = (
    s1[s1["week"] == N_WEEKS]
    .set_index("strategy")[metric_cols]
)
print(final_s1.to_string(float_format="{:.4f}".format))

for tc, _ in all_tc_results:
    scenario_name = f"tariff_change_{tc['name']}"
    s2 = results[results["scenario"] == scenario_name]
    print(f"\nFinal metrics — {tc['label']} (week {N_WEEKS}):")
    final_s2 = (
        s2[s2["week"] == N_WEEKS]
        .set_index("strategy")[metric_cols]
    )
    print(final_s2.to_string(float_format="{:.4f}".format))
