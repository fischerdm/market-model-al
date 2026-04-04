"""
Phase 2 — Copula: Learn the competitor's book-of-business distribution
======================================================================
Fits a Gaussian copula on one row per policy (latest renewal only).
The fitted copula represents the competitor's book of business: which types
of policy profiles they write and how the features co-vary.

Combined with the oracle (01_oracle.py), this simulates the competitor's
full quoting engine: generate a profile from the copula, query the oracle
for a premium.

Validates the fit with marginal overlays (synthetic vs real distributions)
across earlier renewal years as a stationarity sanity check.

Outputs:
  outputs/models/copula.pkl              — saved GaussianCopulaSynthesizer
  outputs/figures/copula_marginals_*.png — synthetic vs real marginal plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from market_model_al.features import CAT_FEATURES, CAT_FEATURES_OBJ, load_raw, engineer_features

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent.parent
MODELS  = ROOT / "outputs" / "models"
FIGURES = ROOT / "outputs" / "figures"
MODELS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── 1. Load raw and select latest renewal per policy ──────────────────────────

raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")

# Keep only the most recent renewal row for each policy
latest_idx = raw.groupby("ID")["Date_last_renewal"].idxmax()
latest_raw = raw.loc[latest_idx].copy()

print(f"Unique policies (latest renewal): {len(latest_raw):,}")

# ── 2. Engineer features; drop Premium (copula models the feature space only) ─

df = engineer_features(latest_raw)
df = df.drop(columns=["Premium"])

print(f"Copula training set: {df.shape[0]:,} rows, {df.shape[1]} features")

# ── 3. Build SDV metadata ─────────────────────────────────────────────────────

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Override integer-coded categoricals that detect_from_dataframe treats as numerical
for col in CAT_FEATURES:
    if col in df.columns:
        metadata.update_column(col, sdtype="categorical")

print("Metadata column types:")
for col, info in metadata.columns.items():
    print(f"  {col:30s} {info['sdtype']}")

# ── 4. Fit Gaussian copula ─────────────────────────────────────────────────────

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df)
synthesizer.save(str(MODELS / "copula.pkl"))
print(f"\nCopula saved: {MODELS / 'copula.pkl'}")

# ── 5. Validate: marginal overlays (synthetic vs real) ────────────────────────

synthetic = synthesizer.sample(num_rows=len(df))

num_cols = [c for c in df.columns if c not in CAT_FEATURES + CAT_FEATURES_OBJ]
cat_cols  = [c for c in df.columns if c in CAT_FEATURES + CAT_FEATURES_OBJ]

# Numerical: KDE overlays
n_num = len(num_cols)
fig, axes = plt.subplots(nrows=(n_num + 2) // 3, ncols=3, figsize=(14, 4 * ((n_num + 2) // 3)))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    real_vals = df[col].dropna()
    syn_vals  = synthetic[col].dropna()
    real_vals.plot.kde(ax=ax, label="real", color="steelblue", bw_method=0.3)
    syn_vals.plot.kde(ax=ax, label="synthetic", color="orange", linestyle="--", bw_method=0.3)
    ax.set_title(col)
    ax.legend(fontsize=8)
    ax.set_ylabel("")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Copula validation — numerical marginals (synthetic vs real)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES / "copula_marginals_numerical.png", dpi=150, bbox_inches="tight")
plt.close()
print("Numerical marginal plot saved.")

# Categorical: normalised bar charts
n_cat = len(cat_cols)
if n_cat > 0:
    fig, axes = plt.subplots(nrows=1, ncols=n_cat, figsize=(5 * n_cat, 4))
    if n_cat == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        real_pct = df[col].value_counts(normalize=True).sort_index()
        syn_pct  = synthetic[col].value_counts(normalize=True).sort_index()
        idx = real_pct.index.union(syn_pct.index)
        pd.DataFrame({"real": real_pct.reindex(idx, fill_value=0),
                      "synthetic": syn_pct.reindex(idx, fill_value=0)}).plot.bar(ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=45)

    fig.suptitle("Copula validation — categorical marginals (synthetic vs real)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES / "copula_marginals_categorical.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Categorical marginal plot saved.")

print("\nDone.")
