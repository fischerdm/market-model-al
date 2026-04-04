"""
freMTPL2 – SHAP Interaction Analysis
=====================================
Preliminary analysis of feature interactions in the French motor TPL dataset.
This forms the basis for the competitor model / active learning pipeline.

Data: https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq
Place freMTPL2freq.csv and freMTPL2sev.csv in data/raw/ before running.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
FIGURES  = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── 1. Load & prepare data ────────────────────────────────────────────────────

freq = pd.read_csv(DATA_RAW / "freMTPL2freq.csv")
sev  = pd.read_csv(DATA_RAW / "freMTPL2sev.csv")

# Aggregate claim amounts per policy
sev_agg = sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()

# Merge and compute pure premium
df = freq.merge(sev_agg, on="IDpol", how="left")
df["ClaimAmount"] = df["ClaimAmount"].fillna(0)
df["PurePremium"] = df["ClaimAmount"] / df["Exposure"].clip(lower=1 / 365)

# Remove known data errors (PolicyID <= 24500, see literature)
df = df[df["IDpol"] > 24500].copy()

# Encode categorical features
cat_cols = ["Area", "VehBrand", "VehGas", "Region"]
for col in cat_cols:
    df[col] = df[col].astype("category")

FEATURES = ["Area", "VehPower", "VehAge", "DrivAge",
            "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
TARGET = "PurePremium"

X = df[FEATURES]
y = df[TARGET]

# ── 2. Train LightGBM ─────────────────────────────────────────────────────────

# Tweedie regression is standard for pure premium modelling
params = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 50,
    "n_estimators": 300,
    "verbose": -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(X, y, categorical_feature=cat_cols)
print("Training complete.")

# ── 3. Compute SHAP interaction values ───────────────────────────────────────

# Subsample to keep computation tractable (interaction values are O(n * p^2))
SAMPLE_N = 2_000
X_sample = X.sample(n=SAMPLE_N, random_state=42)

explainer  = shap.TreeExplainer(model)
shap_inter = explainer.shap_interaction_values(X_sample)  # shape: (n, p, p)

# ── 4. Summarise interaction strength per feature pair ───────────────────────

n_features = len(FEATURES)
inter_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        if i != j:
            inter_matrix[i, j] = np.abs(shap_inter[:, i, j]).mean()

inter_df = pd.DataFrame(inter_matrix, index=FEATURES, columns=FEATURES)

# ── 5. Heatmap of interaction strengths ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.eye(n_features, dtype=bool)  # hide diagonal

sns.heatmap(
    inter_df,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax,
)
ax.set_title("SHAP Interaction Values – freMTPL2\n(mean absolute value)", fontsize=13)
plt.tight_layout()

heatmap_path = FIGURES / "shap_interaction_heatmap.png"
plt.savefig(heatmap_path, dpi=150)
plt.show()
print(f"Heatmap saved: {heatmap_path}")

# ── 6. Print top interaction pairs ───────────────────────────────────────────

rows, cols = np.triu_indices(n_features, k=1)
pairs = [
    {
        "Feature_1": FEATURES[i],
        "Feature_2": FEATURES[j],
        "Interaction_Strength": inter_matrix[i, j],
    }
    for i, j in zip(rows, cols)
]
pairs_df = (
    pd.DataFrame(pairs)
    .sort_values("Interaction_Strength", ascending=False)
    .reset_index(drop=True)
)

print("\nTop-10 interaction pairs:")
print(pairs_df.head(10).to_string(index=False))

# ── 7. Dependence plots for top-3 pairs ──────────────────────────────────────

top3 = pairs_df.head(3)

for _, row in top3.iterrows():
    f1, f2 = row["Feature_1"], row["Feature_2"]
    i, j = FEATURES.index(f1), FEATURES.index(f2)

    shap.dependence_plot(
        ind=f1,
        shap_values=shap_inter[:, i, :],
        features=X_sample,
        interaction_index=f2,
        title=f"Interaction: {f1} × {f2}",
        show=False,
    )
    fname = FIGURES / f"shap_dependence_{f1}_x_{f2}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dependence plot saved: {fname}")

print("\nDone.")
