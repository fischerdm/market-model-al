# Project context for Claude

## What this project is

A simulation of the *competitor model* use case in non-life insurance pricing: an insurer scrapes competitor quotes from aggregator websites and trains a model on them. This project builds a controlled synthetic environment to test active learning (AL) query strategies, where the ground truth is known.

The end deliverable is a **Streamlit dashboard** that lets users explore and compare AL query strategies interactively.

## Architecture decisions

### Dataset
Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, doi: 10.17632/5cxyb5fp4f.2.

Chosen because it contains an actual `Premium` (net premium) column — a direct proxy for a competitor quote. No frequency/severity modelling needed.

### Two-phase pipeline

The real dataset is treated as the competitor's actual tariff. The oracle learns it. The AL loop tests how efficiently a competitor model can recover it.

1. **Oracle (Phase 1)**
   - Fit **LightGBM** on `Premium ~ features` on all rows (including all renewal years) → the **oracle** (the competitor's pricing engine)
   - Validate oracle curves for actuarial plausibility: `driver_age` shape, key interactions — goal is "reasonably good", not perfect; the oracle only needs to be a credible synthetic tariff for testing the AL loop
   - Excluded features: `Cost_claims_year`, `N_claims_year` — current-year outcomes, not observable at quote time
   - `N_claims_history` and `R_Claims_history` also excluded — in practice scraping is done with claim history set to 0 (standardised input on aggregators like comparis.ch), so this matches real-world scraping behaviour
   - Raw date columns engineered to ages/durations instead

2. **AL simulation loop (Phase 2)**
   - Profile pool: the real dataset rows (no generative model needed — 105k rows is sufficient)
   - Warm start: ~50k labeled profiles drawn from the pool (random sample + ceteris paribus profiles)
   - Train the competitor model on the warm-start budget
   - Apply an AL query strategy to select next profiles → label via oracle → retrain → repeat
   - Multiple AL strategies implemented and compared (uncertainty sampling, error-based, SHAP divergence); no prior preference
   - Track convergence in MSE and SHAP structure similarity to the oracle

### No copula / generative model
The copula was dropped. The real dataset (~105k rows) is large enough to serve as the profile pool directly. Drawing from real data is simpler and more principled — those profiles represent the true feature distribution by definition.

### SHAP
Used in two places:
- On the **oracle** — validate the learned tariff structure looks actuarially sensible (driver age curve, vehicle age, power, interactions)
- On the **competitor model** — track recovery of the oracle's SHAP structure across AL iterations (a richer convergence metric than MSE alone)

## Stack
Python 3.12, LightGBM, SHAP, Streamlit. All notebooks are `.py` files (numbered), not `.ipynb`.
