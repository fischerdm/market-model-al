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
   - `licence_age` = years since obtaining the licence (driving experience), NOT age at licensing; the physical constraint is `licence_age <= driver_age - MIN_AGE_AT_LICENSING` where `MIN_AGE_AT_LICENSING = 18` (Spain)

2. **AL simulation loop (Phase 2)**

   **Warm start** seeds the competitor model with ~5,000 real rows (≈ one week's scraping budget), simulating organic quote requests arriving via the aggregator before the systematic AL loop begins. CP profiles in the warm start were removed — they create distribution mismatch and add little value at this warm-start size.

   **Weekly AL loop**:
   - Each week: sample `n_candidates = candidate_multiplier × n_anchors` anchor rows from the real dataset → apply query strategy to select `n_anchors` → generate ceteris-paribus profiles from selected anchors → label via current oracle → retrain competitor model
   - `n_anchors = weekly_budget // PROFILES_PER_ANCHOR` (254 profiles per anchor)
   - No pre-built pool on disk — profiles are generated lazily on demand
   - No deduplication across weeks — the same anchor can in principle be selected again; rare given pool size (~100k rows, 19 anchors/week)

   **Convergence metrics**:
   - **RMSE on holdout** — 2,000 real rows carved from the dataset at construction time, oracle-labeled, never used for training. Population-representative; comparable across weeks and strategies.
   - **SHAP cosine similarity** *(simulation-only)* — mean cosine similarity between oracle and competitor SHAP vectors on the holdout. Measures tariff *structure* recovery, not just premium accuracy. Requires oracle access → cannot be observed in real-world deployment. Included as a simulation diagnostic only.

   **AL query strategies** (all deployable in practice except where noted):
   - `random` — uniform random baseline; representative by construction
   - `uncertainty` — bootstrap variance across lightweight ensemble members; falls back to random if labeled set is empty
   - `error_based` — expected *relative* error (|residual| / label) predicted by a proxy model trained on labeled data; falls back to random if labeled set is empty
   - `segment_adaptive` — scores each anchor by global + per-segment *relative* RMSE on the labeled set; converges toward random as gaps close; falls back to random if labeled set is empty
   - `disruption` — monitors week-on-week *change* in per-segment RMSE; concentrates budget on random sampling within disrupted segments (≥15% relative RMSE increase); reverts to global random when no disruption; falls back to random on first week or after restart

   **Removed strategy**: `shap_divergence` required oracle SHAP values to score candidates — not available in real-world deployment. Removed from the simulation.

   **Tariff change simulation** (`PerturbedOracleEngine` in `perturbed_oracle.py`):
   - A perturbed oracle applies a systematic premium shift (e.g. young-driver surcharge +20%, uniform reprice, area repricing)
   - Injected mid-run at `tariff_change_week`; holdout labels switch to the new oracle so RMSE measures recovery of the new tariff
   - `restart_at_tariff_change=True` clears the labeled set after the tariff-change week's evaluation; model re-learns from new-oracle labels only
   - Restart variants run for `random` and `segment_adaptive` in scenario 2

   **Core research question**: does the AL strategy rediscover systematic ceteris paribus profiling on its own? A good strategy should converge on varying one factor at a time across its range — this is the structure of a competitor tariff that scraping is trying to reveal.

   **Simulation findings (10-week run, 5 000 profiles/week)**:
   - Random anchor sampling is competitive with or better than all informativeness strategies on global RMSE and SHAP similarity
   - `error_based` recovers the young-driver segment (age < 30) faster — the one segment where a sophisticated strategy wins; commercially important
   - Root cause: greedy informativeness strategies are not representative — they starve mainstream segments of scraping budget
   - Full restart after a targeted tariff change (young-driver surcharge) is not always optimal: it discards valid labels from unchanged segments and can end up with higher global RMSE at week 10 than continuous scraping
   - `disruption` is the principled response: targets only disrupted segments without discarding any labels

### No copula / generative model
The copula was dropped. The real dataset (~105k rows) is large enough to serve as anchor points directly. Drawing from real data is simpler and more principled — those profiles represent the true feature distribution by definition.

### SHAP
Used in two places:
- On the **oracle** — validate the learned tariff structure looks actuarially sensible (driver age curve, vehicle age, power, interactions)
- On the **competitor model** — track recovery of the oracle's SHAP structure across AL weeks (simulation diagnostic only; not observable in practice)

## Stack
Python 3.12, LightGBM, SHAP, Streamlit. All notebooks are `.py` files (numbered), not `.ipynb`. Virtual environment is `.venv` (not conda).
