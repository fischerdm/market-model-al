# Project context for Claude

## What this project is

A simulation of the *nanny model* use case in non-life insurance pricing: an insurer scrapes competitor quotes from aggregator websites and trains a model on them. This project builds a controlled synthetic environment to test active learning (AL) query strategies, where the ground truth is known.

## Architecture decisions

### Dataset
Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, doi: 10.17632/5cxyb5fp4f.2.

Chosen because it contains an actual `Premium` (net premium) column — a direct proxy for a competitor quote. No frequency/severity modelling needed.

### Three-phase pipeline

The real dataset is treated as the competitor's actual tariff. The goal is to learn it well enough to simulate a quoting engine, then test how efficiently an AL strategy can recover that engine from a limited scraping budget.

1. **Learn from real data (Phases 1 & 2)**
   - Fit a **Gaussian copula** on one row per policy (latest renewal) → the feature distribution represents the competitor's book of business
   - Fit **LightGBM** on `Premium ~ features` on all rows (including all renewal years) → this becomes the **oracle** (the competitor's pricing engine)
   - Validate the oracle with SHAP to confirm actuarially sensible factor effects
   - Validate the copula fit across earlier renewal years (lightweight: marginal overlays) as a sanity check that the learned distribution is reasonably stationary
   - Excluded from oracle features: claim outcome columns (`Cost_claims_year`, `N_claims_year`, `N_claims_history`, `R_Claims_history`) — not observable at quote time; also raw date columns (engineer to ages/durations instead)

2. **Synthetic world (Phase 2 output)**
   - The copula + oracle together simulate the competitor's quoting engine: generate any profile, get a quote
   - Drifts (oracle weight perturbations representing competitor repricings) deferred to a later extension

3. **AL simulation loop (Phase 3)**
   - Warm start: ~50k labeled profiles, mix of:
     - Random samples from the copula
     - Ceteris paribus profiles: select real profiles from the sample, vary one factor at a time (others held constant) — equivalent to one-way analyses
   - Train the nanny model on the warm-start budget
   - Apply an AL query strategy to select next profiles → label via oracle → retrain → repeat
   - Multiple AL strategies implemented and compared (uncertainty sampling, error-based, SHAP divergence); no prior preference
   - Track convergence in MSE and SHAP structure similarity to the oracle

### Generative model choice
Start simple: **Gaussian copula + parametric marginals** (e.g. SDV `GaussianCopulaSynthesizer`). The synthetic world only needs to represent interesting relationships, not perfectly replicate reality.

Alternatives considered but deferred:
- **CTGAN / TVAE** — better at complex non-linear dependencies in mixed-type tabular data, but a black box; revisit if the copula fails to capture important structure

### SHAP
Used in two places:
- On the **oracle** — validate the learned tariff structure looks actuarially sensible
- On the **nanny model** — track recovery of the oracle's SHAP structure across AL iterations (a richer convergence metric than MSE alone)

## Stack
Python 3.12, LightGBM, SHAP, SDV (copula synthesis). All notebooks are `.py` files (numbered), not `.ipynb`.
