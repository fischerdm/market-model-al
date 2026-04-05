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
import pandas as pd
import seaborn as sns
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

print("Loading and engineering features...")
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

print("Fitting oracle (this may take a minute)...")
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

print(f"Computing SHAP values on {SAMPLE_N:,} sample rows...")
explainer   = shap.TreeExplainer(oracle)
shap_values = explainer.shap_values(X_sample)
print("SHAP values done.")

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
        interaction_index=None,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP dependence — {feature}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES / f"oracle_shap_dep_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dependence plot saved: {feature}")

# ── 5. SHAP interaction values — detect strongest pairwise interactions ────────
# Expensive (O(n × p²)) so use a small subsample.
# Step 1: heatmap of mean |interaction| across all pairs.
# Step 2: dependence plots for hand-picked actuarially interesting pairs,
#         coloured by the interaction partner to reveal the joint effect.

INTERACTION_SAMPLE_N = 500
X_inter = X_sample.sample(n=INTERACTION_SAMPLE_N, random_state=42)

# shap_interaction_values requires numeric input — encode categoricals to integer codes
X_inter_encoded = X_inter.copy()
for col in X_inter_encoded.select_dtypes(["category"]).columns:
    X_inter_encoded[col] = X_inter_encoded[col].cat.codes

print(f"Computing SHAP interaction values on {INTERACTION_SAMPLE_N} rows (slow, O(n×p²))...")
shap_inter = explainer.shap_interaction_values(X_inter_encoded)  # (n, p, p)
print("SHAP interaction values done.")

features_list = list(X_inter_encoded.columns)
n_feat = len(features_list)

inter_matrix = np.zeros((n_feat, n_feat))
for i in range(n_feat):
    for j in range(n_feat):
        if i != j:
            inter_matrix[i, j] = np.abs(shap_inter[:, i, j]).mean()

inter_df = pd.DataFrame(inter_matrix, index=features_list, columns=features_list)

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.eye(n_feat, dtype=bool)
sns.heatmap(
    inter_df, mask=mask, annot=True, fmt=".2f",
    cmap="YlOrRd", linewidths=0.4, ax=ax,
)
ax.set_title("SHAP interaction strengths (mean |interaction value|)", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES / "oracle_shap_interaction_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Interaction heatmap saved.")

# Step 2: dependence plots for key actuarial interaction pairs
INTERACTION_PAIRS = [
    ("driver_age",   "Power"),          # young driver + powerful car
    ("driver_age",   "Value_vehicle"),  # young driver + expensive car
    ("driver_age",   "licence_age"),    # age vs experience
    ("Power",        "Value_vehicle"),  # performance vs value
]

for feat, interaction in INTERACTION_PAIRS:
    if feat not in X_inter_encoded.columns or interaction not in X_inter_encoded.columns:
        continue
    print(f"  Plotting interaction: {feat} × {interaction}...")
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.dependence_plot(
        feat,
        shap_inter[:, :, features_list.index(feat)],
        X_inter_encoded,
        interaction_index=interaction,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP interaction — {feat} × {interaction}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    fname = FIGURES / f"oracle_shap_inter_{feat}_x_{interaction}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Interaction plot saved: {feat} × {interaction}")

print("\nDone.")
