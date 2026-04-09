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

   **Warm start** seeds the competitor model with two components:
   - **(A) Real rows** — a random sample from the Spanish dataset, simulating organic quote requests arriving via comparis (e.g. 10k rows ≈ 10% of the dataset)
   - **(B) CP profiles** — structured ceteris-paribus sweeps from a small set of anchors, simulating an initial systematic scrape (e.g. 50 anchors ≈ 12k profiles)
   - The warm-start mix ratio is a configurable experiment parameter; `04_build_warm_start.py` builds and saves it

   **Weekly AL loop** (time-indexed by `profiles_per_week`):
   - Each week: sample `n_anchors_per_week` anchor rows from the real dataset → generate ceteris-paribus candidate profiles ad hoc → apply query strategy to select `profiles_per_week` candidates → label selected profiles via the oracle → retrain competitor model
   - No pre-built pool on disk — profiles are generated lazily on demand
   - Track convergence: RMSE on a fixed holdout + mean SHAP cosine similarity to the oracle

   **AL query strategies** (all implemented, no prior preference):
   - `random` — uniform random baseline
   - `uncertainty` — bootstrap variance across lightweight ensemble members
   - `error_based` — expected absolute error predicted by a proxy model trained on labeled residuals
   - `shap_divergence` — profiles where oracle and competitor SHAP vectors diverge most (L2 distance)

   **Tariff change simulation** (`PerturbedOracleEngine` in `perturbed_oracle.py`):
   - A perturbed oracle applies a systematic premium shift (e.g. young-driver surcharge +20%, uniform reprice, area repricing)
   - Injected mid-run at `tariff_change_week`; holdout labels switch to the new oracle so RMSE measures recovery of the new tariff
   - Lets practitioners compare *continuous scraping* vs *restart after tariff change* strategies
   - Continuous scraping is NOT always optimal — a major re-tariffing may require a restart + bulk scrape to converge faster than incrementally overriding stale labeled data

   **Core research question**: does the AL strategy rediscover systematic ceteris paribus profiling on its own? A good strategy should converge on varying one factor at a time across its range — this is the structure of a competitor tariff that scraping is trying to reveal.

   **Strategy redesign (implemented)**: strategies now select *anchor rows*, not individual profiles. The weekly budget determines how many anchors are selected (`n_anchors = weekly_budget // PROFILES_PER_ANCHOR`). All CP profiles from selected anchors are generated and labeled. A candidate pool of `candidate_multiplier × n_anchors` anchors is scored each week; the best are kept.

   **Simulation findings (10-week run, 5 000 profiles/week)**:
   - Random anchor sampling beats all three informativeness-based strategies on a population-representative holdout
   - `error_based` recovers the young-driver segment (age < 30) faster than random — the one segment where a sophisticated strategy wins
   - `shap_divergence` concentrates on edge cases (young drivers, expensive cars, high power), creating a distribution mismatch with the holdout; SHAP cosine similarity bottoms out around week 8
   - Root cause: greedy informativeness strategies are not representative — they starve mainstream segments of scraping budget
   - Next step: `segment_adaptive` strategy — allocate weekly budget proportionally to per-segment RMSE, then score anchors within each segment

   **Open question on warm start CP profiles**: the warm start mixes random real rows (A) and CP profiles (B). CP profiles help bootstrap marginal effects quickly but may not add much over random rows when the warm start is large (~20k profiles) — the model is already reasonably informed. CP profiles also create a distribution mismatch relative to the true joint feature distribution. Whether removing them from the warm start materially changes AL strategy performance is an open empirical question.

### No copula / generative model
The copula was dropped. The real dataset (~105k rows) is large enough to serve as anchor points directly. Drawing from real data is simpler and more principled — those profiles represent the true feature distribution by definition.

### SHAP
Used in two places:
- On the **oracle** — validate the learned tariff structure looks actuarially sensible (driver age curve, vehicle age, power, interactions)
- On the **competitor model** — track recovery of the oracle's SHAP structure across AL weeks (a richer convergence metric than MSE alone); measured as mean cosine similarity on a fixed holdout set

## Stack
Python 3.12, LightGBM, SHAP, Streamlit. All notebooks are `.py` files (numbered), not `.ipynb`. Virtual environment is `.venv` (not conda).
