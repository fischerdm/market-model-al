"""
Segment summary — calibrate segment thresholds
===============================================
Explores the feature distributions relevant to the holdout segments defined in
segments.py, so we can pick thresholds that are commercially meaningful and
leave enough rows in the holdout for reliable RMSE estimates.

Covers:
  - Value_vehicle  → "high-value cars" segment
  - Power          → "high-power cars" segment
  - driver_age     → young (<30) and senior (>=60) segments

Also shows how segment size scales with different holdout sizes.

Run from the project root:
  python notebooks/06_segment_summary.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from market_model_al.features import load_raw, engineer_features
from market_model_al.segments import SEGMENTS

# ── Load data ─────────────────────────────────────────────────────────────────

raw = load_raw(ROOT / "data" / "raw" / "Motor vehicle insurance data.csv")
df  = engineer_features(raw)
n_total = len(df)
print(f"Total rows after feature engineering: {n_total:,}\n")

# ── 1. Value_vehicle distribution ─────────────────────────────────────────────

print("=" * 60)
print("Value_vehicle")
print("=" * 60)
v = df["Value_vehicle"]
print(v.describe(percentiles=[.50, .75, .90, .95, .97, .99]))
print()
print("Percentile table:")
for p in [75, 80, 85, 90, 95, 97, 99]:
    print(f"  p{p:>3d}: {np.percentile(v, p):>10,.0f}")
print()
print("Share of portfolio above threshold:")
for thresh in [25_000, 30_000, 40_000, 50_000, 60_000, 75_000, 80_000, 100_000]:
    pct = (v > thresh).mean()
    print(f"  > {thresh:>7,}: {pct:5.1%}  (~{pct * 2000:4.0f} rows in a 2 000-row holdout"
          f" / ~{pct * 5000:4.0f} in a 5 000-row holdout)")

# ── 2. Power distribution ──────────────────────────────────────────────────────

print()
print("=" * 60)
print("Power (hp)")
print("=" * 60)
pw = df["Power"]
print(pw.describe(percentiles=[.50, .75, .90, .95, .97, .99]))
print()
print("Percentile table:")
for p in [75, 80, 85, 90, 95, 97, 99]:
    print(f"  p{p:>3d}: {np.percentile(pw, p):>8,.0f} hp")
print()
print("Share of portfolio above threshold:")
for thresh in [100, 120, 130, 150, 180, 200]:
    pct = (pw > thresh).mean()
    print(f"  > {thresh:>4} hp: {pct:5.1%}  (~{pct * 2000:4.0f} rows in a 2 000-row holdout"
          f" / ~{pct * 5000:4.0f} in a 5 000-row holdout)")

# ── 3. driver_age distribution ────────────────────────────────────────────────

print()
print("=" * 60)
print("driver_age")
print("=" * 60)
age = df["driver_age"]
print(age.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95]))
print()
for thresh, label in [(30, "young  (age < 30)"), (60, "senior (age >= 60)")]:
    if label.startswith("young"):
        pct = (age < thresh).mean()
    else:
        pct = (age >= thresh).mean()
    print(f"  {label}: {pct:5.1%}  (~{pct * 2000:4.0f} rows in 2 000-row holdout"
          f" / ~{pct * 5000:4.0f} in 5 000-row holdout)")

# ── 4. Current segment sizes on holdout sizes ─────────────────────────────────

print()
print("=" * 60)
print("Current segment sizes (segments.py thresholds)")
print("=" * 60)
print(f"  {'Segment':<20} {'% portfolio':>12}  {'n (holdout=2k)':>16}  {'n (holdout=5k)':>16}")
print("  " + "-" * 68)
for seg in SEGMENTS:
    mask = seg.filter_fn(df)
    pct  = mask.mean()
    print(f"  {seg.label:<20} {pct:>11.1%}  {pct * 2000:>16.0f}  {pct * 5000:>16.0f}")

# ── 5. Plots ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Feature distributions — segment calibration", fontsize=13)

# Value_vehicle
ax = axes[0]
ax.hist(v, bins=80, color="steelblue", edgecolor="none")
ax.axvline(25_000, color="tomato", linestyle="--",
           label=f">25k current ({(v > 25_000).mean():.1%})")
ax.axvline(28_000, color="gold", linestyle="-",
           label=f">28k proposed ({(v > 28_000).mean():.1%})")
ax.set_xlabel("Value_vehicle (€)")
ax.set_ylabel("Count")
ax.set_title("Vehicle value")
ax.set_xlim(0, 120_000)
ax.legend(fontsize=8)

# Power
ax = axes[1]
ax.hist(pw, bins=60, color="steelblue", edgecolor="none")
ax.axvline(150, color="tomato", linestyle="--",
           label=f">150 hp current ({(pw > 150).mean():.1%})")
ax.axvline(130, color="gold", linestyle="-",
           label=f">130 hp proposed ({(pw > 130).mean():.1%})")
ax.set_xlabel("Power (hp)")
ax.set_ylabel("Count")
ax.set_title("Engine power")
ax.legend(fontsize=8)

# driver_age
ax = axes[2]
ax.hist(age, bins=60, color="steelblue", edgecolor="none")
ax.axvline(30, color="tomato", linestyle="--",
           label=f"<30 young — unchanged ({(age < 30).mean():.1%})")
ax.axvline(60, color="tomato", linestyle="-.",
           label=f"≥60 senior current ({(age >= 60).mean():.1%})")
ax.axvline(65, color="gold", linestyle="-",
           label=f"≥65 senior proposed ({(age >= 65).mean():.1%})")
ax.set_xlabel("Driver age (years)")
ax.set_ylabel("Count")
ax.set_title("Driver age")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = ROOT / "outputs" / "figures" / "segment_summary.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"\nFigure saved to {out_path.relative_to(ROOT)}")
plt.show()
