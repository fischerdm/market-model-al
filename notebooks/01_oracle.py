"""
Phase 1 — Oracle: Learn the competitor's pricing engine
========================================================
Fits a LightGBM model on Premium ~ engineered features using all renewal rows.
The fitted model becomes the oracle: given any policy profile, it returns a
simulated competitor quote.

Validates the learned tariff structure with SHAP (actuarial sanity check).

Outputs:
  outputs/models/oracle.pkl                   — saved LightGBM model
  outputs/figures/oracle_shap_summary.png     — SHAP beeswarm
  outputs/figures/oracle_shap_importance.png  — SHAP importance bar
  outputs/figures/oracle_shap_dep_*.png       — SHAP dependence plots (key features)
"""

import joblib
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from market_model_al.features import CAT_FEATURES, load_raw, engineer_features

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent.parent
MODELS  = ROOT / "outputs" / "models"
FIGURES = ROOT / "outputs" / "figures"
MODELS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── 1. Load and engineer features ─────────────────────────────────────────────

raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df  = engineer_features(raw)  # all rows (all renewal years)

TARGET   = "Premium"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"Oracle training set: {len(X):,} rows, {len(FEATURES)} features")

# ── 2. Fit oracle on all rows ──────────────────────────────────────────────────

params = {
    "objective":          "regression",
    "learning_rate":      0.02,
    "num_leaves":         128,
    "min_child_samples":  20,
    "n_estimators":       2000,
    "verbose":            -1,
}

oracle = lgb.LGBMRegressor(**params)
oracle.fit(X, y, categorical_feature=CAT_FEATURES)

# Sanity check — in-sample fit (oracle is meant to memorise the tariff)
pred = oracle.predict(X)
rmse = np.sqrt(mean_squared_error(y, pred))
r2   = r2_score(y, pred)
print(f"In-sample — RMSE: {rmse:.2f}, R²: {r2:.4f}")

joblib.dump(oracle, MODELS / "oracle.pkl")
print(f"Oracle saved: {MODELS / 'oracle.pkl'}")

# ── 3. SHAP validation ─────────────────────────────────────────────────────────

SAMPLE_N = 5_000
X_sample = X.sample(n=SAMPLE_N, random_state=42)

explainer   = shap.TreeExplainer(oracle)
shap_values = explainer.shap_values(X_sample)

# Beeswarm: shows direction and magnitude of each feature's effect
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(FIGURES / "oracle_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP beeswarm saved.")

# Bar: mean |SHAP| importance ranking
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(FIGURES / "oracle_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP importance bar saved.")

# ── 4. SHAP dependence plots — key actuarial factors ──────────────────────────
# Each plot shows the marginal effect of one feature on the premium prediction.
# The colour axis shows the most important interacting feature (auto-selected by
# SHAP). These are the curves we validate for actuarial plausibility.

DEPENDENCE_FEATURES = [
    "driver_age",       # expect U-shape: young and elderly drivers costlier
    "licence_age",      # expect decreasing: more experience → lower risk
    "vehicle_age",      # expect non-linear: older cars may be cheaper to insure
    "Power",            # expect increasing: more powerful → higher premium
    "Cylinder_capacity",
    "Value_vehicle",    # expect increasing: more valuable → higher premium
    "Seniority",        # loyalty effect: longer customer → potential discount
]

for feature in DEPENDENCE_FEATURES:
    if feature not in X_sample.columns:
        print(f"  Skipping {feature} (not in features)")
        continue
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.dependence_plot(
        feature,
        shap_values,
        X_sample,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP dependence — {feature}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES / f"oracle_shap_dep_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dependence plot saved: {feature}")

print("\nDone.")
