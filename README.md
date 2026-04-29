# When Random Wins: A Simulation Study on Reverse-Engineering a Competitor Tariff

> Seventeen strategies were designed to reverse-engineer an insurance tariff from aggregator quotes using a Gradient Boosting Oracle. None outperformed random sampling.

&nbsp;

[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://market-model-al.streamlit.app/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/fischerdm/market-model-al/releases)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Data DOI](https://img.shields.io/badge/Data-10.17632%2F5cxyb5fp4f.2-orange.svg)](https://doi.org/10.17632/5cxyb5fp4f.2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project simulates the process of scraping competitor quotes from aggregator websites to build and continuously retrain a **competitor model** — a replica of a competitor's pricing engine in non-life insurance.

Seventeen strategies were designed to reverse-engineer a competitor tariff from synthetic aggregator quotes, mimicking the weekly scraping and retraining loop in practice: each week, a batch of policy profiles is submitted to the simulated aggregator, the returned quotes label the training set, and the competitor model is retrained. None of the strategies outperformed random sampling. Even the cube method, which achieves exact covariate balance by construction, is only marginally better in some settings.

## Concept

An insurer can train a competitor model by scraping quoted premiums from aggregator platforms. This project simulates that iterative process in a controlled synthetic environment where the ground truth is known.

The real dataset is treated as the competitor's actual tariff. A LightGBM oracle learns that tariff and can return a premium for any policy profile — simulating the function of an aggregator. The **Active Learning (AL) loop** then tests how efficiently a competitor model can recover the oracle from a limited scraping budget: each week, a strategy selects which profiles to query, the oracle labels them, and the competitor model is retrained.

The end deliverable is a **Streamlit dashboard** for interactively exploring and comparing AL query strategies.

### Phase 1 — Oracle: learn the competitor's pricing engine

- Fit a **LightGBM oracle** on `Premium ~ features` using all rows from the dataset (105,555 policy-year observations across 53,502 unique policies), excluding current-year claim outcomes (`Cost_claims_year`, `N_claims_year`) and claim history features (`N_claims_history`, `R_Claims_history`) → the competitor's pricing engine
- Dataset: Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, [doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2)
- Validate oracle structure with **SHAP dependence plots** (driver age U-curve, vehicle power, key interactions — actuarial sanity check)

### Company portfolio vs. market portfolio

The Lledó dataset represents a *company* portfolio — the policies actually held by one insurer. This is not the same as the full quote traffic an insurer receives via the aggregator — all profiles for which a premium is requested, regardless of whether they convert.

**The aggregator loop:** a user submits a profile on a price comparison website; the request is forwarded to every participating insurer; each insurer returns a quote. The company therefore sees **all quote requests arriving via the aggregator** — the full market, not just its own policyholders. Because premiums are price-elastic, segments where the company is uncompetitive generate few conversions and are under-represented in its portfolio, even though the company still receives those quote requests.

**The preparation step** (`create_market_supplement()`) constructs a **market portfolio** from the company portfolio by supplementing under-represented segments. This market portfolio is used as the anchor pool for `random_market` and `informed_market`, and for the warm start. The `market_supplement_ratio` parameter (default 10%) controls the supplement fraction. This correction is what gives the market strategies their structural advantage: they train on a distribution that more closely reflects the full aggregator traffic, not just the policies the company happened to write.

> **Important distinction.** The oracle (Phase 1) is trained on the *company portfolio* only — and intentionally so. The oracle represents the competitor's own pricing engine, learned from their own data. The market portfolio correction applies only to Phase 2: it adjusts the distribution of profiles that *our* competitor model is trained on during the AL loop, simulating what we would observe arriving via the aggregator.

### Phase 2 — Weekly scraping and retraining loop

**Three families of strategies** are compared:

- **Ceteris-paribus (CP):** for each anchor row, each continuous feature is swept one at a time across its full range while all other features are held fixed. Produces 254 profiles per anchor.
- **Gaussian perturbations:** for each anchor row, all continuous features are perturbed simultaneously with independent Gaussian noise (σ = `gaussian_sigma_frac × feature_range`). Values are clipped and constraint-validated. Produces 254 profiles per anchor (same budget). Tests whether joint feature variation allows LightGBM to learn interaction effects more efficiently than axis-aligned CP sweeps.
- **Market sampling** (`random_market`, `informed_market`, `cube_market`): draws real rows directly from the market-corrected pool, preserving natural feature correlations. No synthetic profile generation.

**Warm start** seeds the competitor model with `warmup_weeks × weekly_budget` rows (default: 1 × 5,000 = 5,000), simulating organic quote requests arriving before the systematic scraping loop begins. Composition mirrors `random_market`: real portfolio rows topped up with a synthetic supplement via `create_market_supplement()`. A `warmup_scale` factor (default 1.2) oversamples before validation to guarantee exact counts after constraint dropout.

The **weekly loop** labels a batch of profiles via the oracle and retrains the competitor model. For CP and Gaussian strategies, anchor selection works as follows:

**Budget and anchor pool (CP and Gaussian)**

```
n_anchors_base  = weekly_budget // profiles_per_anchor   # e.g. 5 000 ÷ 254 = 19
n_pool          = n_anchors_base × anchor_space_multiplier   # e.g. 19 × 30 = 570  (all scored)
n_selected      = round(n_pool × selection_fraction)          # e.g. 570 × 10% = 57  (profiled)
profiles        → trimmed to weekly_budget from the top-ranked anchors first
```

`anchor_space_multiplier` (default 30) controls how wide the candidate field is. `selection_fraction` (default 10%) controls what share is profiled — increase it if Gaussian validation dropout leaves the weekly budget under-utilised. Trimming by rank means the highest-scoring anchors always contribute their profiles before lower-ranked ones are cut.

For market sampling strategies, no anchor profiling is involved. `random_market` draws rows directly from the market-corrected pool. `informed_market` draws a large representative pool and selects the top `weekly_budget` rows by expected prediction error. `cube_market` builds a pool of 3× the weekly budget and applies the Tillé-Deville cube method to select a sample whose covariate means exactly match the population — balance by construction rather than in expectation.

Anchor-based strategies come in two variants — CP (`_cp`) and Gaussian (`_gauss`) — corresponding to the two profile generators described above:

| Strategy | Query criterion | Deployable in practice? |
|---|---|---|
| `random_cp` / `random_gauss` | Uniform random anchor selection — no model required | Yes |
| `random_market` | Uniform random selection from the market-corrected pool — no scoring required | Yes |
| `informed_market` | Error-based scoring on a large representative pool (same market composition as `random_market`); selects top `weekly_budget` rows — best-of-both-worlds hybrid | Yes |
| `cube_market` | Tillé-Deville cube method on a pool 3× the weekly budget: selects profiles balanced on all 7 continuous features by construction, not just in expectation | Yes |
| `uncertainty_cp` / `_gauss` | Anchors where bootstrap prediction variance is highest | Yes |
| `error_based_cp` / `_gauss` | Anchors with highest expected relative residual (proxy model on labeled data) | Yes |
| `segment_adaptive_cp` / `_gauss` | Anchors scored by global + per-segment relative RMSE; converges toward random as gaps close | Yes |
| `disruption_cp` / `_gauss` | Concentrates budget on segments with a sharp week-on-week RMSE spike; falls back to random otherwise | Yes |

Convergence is tracked in two metrics:
- **RMSE on holdout** — a fixed set of 5,000 real rows, oracle-labeled, never used during training. Measures prediction accuracy on a population-representative sample.
- **SHAP cosine similarity** *(simulation-only diagnostic)* — compares the competitor model's SHAP vectors to the oracle's on the holdout. Captures whether the tariff *structure* has been recovered, not just the premium levels. Requires oracle access, so it cannot be observed in real-world deployment.

**Tariff change simulation**: a `PerturbedOracleEngine` can be injected at one or more configurable weeks within a single simulation run to simulate a competitor repricing event (e.g. young-driver surcharge +20%, area repricing, or composed stacked shocks). Multiple shocks can be chained — the competitor model experiences all of them in one continuous timeline, with the RMSE curve measuring recovery of the *currently active* tariff at each point. Simulations and perturbation types are fully defined in YAML config files, with no code changes required to add new scenarios.

**Core research questions**:
1. Does an AL strategy rediscover systematic ceteris paribus profiling on its own — varying one factor at a time while holding all others fixed?
2. Do Gaussian joint perturbations — varying all features simultaneously around an anchor — outperform CP sweeps by exposing LightGBM to genuine multivariate variation within each anchor's batch?
3. Can a best-of-both-worlds hybrid — applying an informativeness filter within a representative pool — outperform pure random market scraping?
4. Does balanced sampling via the cube method (Tillé & Deville, 2004) improve on simple random market scraping — and how close is simple random sampling (SRS) to the theoretical optimum?

## Simulation findings (10-week run, 5 000 profiles/week)

The central tension is the **exploration-exploitation tradeoff**: informativeness strategies exploit high-error regions at the cost of mainstream coverage; representative strategies explore the full market space.

| Finding | Detail |
|---|---|
| **Random market outperforms all strategies** | Real portfolio rows carry natural joint feature correlations — LightGBM learns interaction effects far more efficiently from these than from any synthetic profiles (CP or Gaussian). Holds globally and in every segment. |
| **Informed market (hybrid) does not beat random market** | Applying error-based scoring within a representative pool — the best-of-both-worlds approach — still does not outperform pure random market. Representativeness is the dominant factor; the informativeness filter adds noise and cost without improving convergence. |
| **Gaussian profiles do not close the gap** | Joint Gaussian perturbations perform comparably to their CP counterparts. Varying all features simultaneously does not compensate for the absence of natural feature correlations present in real observed quotes. |
| **Random beats sophisticated CP strategies globally** | Among CP strategies, random anchor sampling matches or outperforms all informativeness-based strategies on global RMSE and SHAP cosine similarity at 10 weeks. |
| **Error-based recovers young drivers faster** | Segment-level RMSE reveals that `error_based_cp` converges faster on young drivers (age < 30) — the one segment where informativeness pays. |
| **Root cause: exploration vs. exploitation** | Greedy informativeness (exploitation) pulls budget toward high-signal edge cases, starving mainstream segments. Representative sampling (exploration) covers the market proportionally by construction — and that is sufficient. |
| **Continuous scraping outperforms restart** | After a targeted tariff change, a full restart discards valid labels from unchanged segments. Continuous scraping can win on global RMSE at week 10. |
| **Disruption-adaptive** | Uses the week-on-week *change* in segment RMSE as a signal, not the absolute level. Fires on disruption, reverts to random once the gap closes, discards no labels. |

## Survey sampling perspective

The central finding — that `random_market` beats every informativeness-based AL strategy — can be reframed through survey sampling theory. The problem is fundamentally about **estimating a finite population (the tariff surface) from a limited budget**, a problem survey sampling has optimised for decades.

| Survey sampling concept | Equivalent in this project |
|---|---|
| Simple random sampling (SRS) | `random_market` — achieves representativeness *in expectation* |
| Neyman allocation | `segment_adaptive_cp` / `error_based_cp` — oversample high-variance strata; the formal guarantee that these heuristics approximate |
| Balanced sampling (cube method) | `cube_market` — implemented; `cube_market` is sometimes marginally better than `random_market`, confirming SRS is already near the theoretical ceiling |

`random_market` wins because representativeness is the dominant factor. `cube_market` (Tillé & Deville, 2004) delivers exact covariate balance by construction — a strictly stronger property than SRS — yet the gain over `random_market` is only marginal. This confirms that SRS is robust: random deviation from the population mean at n=5,000 profiles/week is already small enough that eliminating it entirely provides little additional benefit.

## Open questions

All findings are specific to a **gradient boosting oracle**. Whether representativeness retains its advantage over informativeness under simpler, more separable tariff structures is an open question — GLM or GAM-based pricing engines (the traditional standard in non-life insurance) represent a particularly compelling comparison point, since ceteris-paribus profiling was originally motivated by the multiplicative structure of such models. Actuaries are encouraged to adapt this codebase to their own data and pricing engine.

## Project structure

```
market-model-al/
├── config/
│   ├── simulation.yaml         # n_weeks, budget, strategies, metrics, simulations list
│   │                           # advanced: anchor_space_multiplier, selection_fraction,
│   │                           #           gaussian_sigma_frac, market_supplement_ratio,
│   │                           #           market_profile_method, random_market.market_n_anchors,
│   │                           #           warmup_weeks, warmup_scale
│   └── tariff_changes.yaml     # named perturbation library (type + params, no timing)
├── data/
│   ├── raw/                # Lledó & Pavía (2024) data (not committed)
│   └── processed/          # intermediate processed data (not committed)
├── docs/
│   └── index.html          # GitHub Pages project page
├── notebooks/              # numbered .py scripts (run in order)
│   ├── 01_oracle.py                    # fit and validate the oracle
│   ├── 02_oracle_engine_smoke.py       # smoke test: OraclePricingEngine
│   ├── 03_profile_generator_smoke.py   # smoke test: ceteris-paribus generator
│   ├── 04_build_warm_start.py          # build warm start (warmup_weeks × weekly_budget rows)
│   ├── 05_al_simulation.py             # run all simulations, save results + figures
│   └── 06_segment_summary.py           # segment distribution analysis + threshold calibration
├── src/
│   └── market_model_al/
│       ├── features.py           # feature engineering
│       ├── constraints.py        # physical validity rules (MIN_AGE_AT_LICENSING=18)
│       ├── oracle_engine.py      # OraclePricingEngine: query(profiles) → prices
│       ├── profile_generator.py  # generate_ceteris_paribus, generate_gaussian_profiles,
│       │                         # create_market_supplement; PROFILES_PER_ANCHOR = 254
│       ├── competitor_model.py   # CompetitorModel: LightGBM, retrained each iteration
│       ├── cube_sampling.py      # Tillé-Deville cube method balanced sampling
│       ├── strategies.py         # AL query strategies (STRATEGIES list)
│       ├── segments.py           # four actuarial segments + segment_rmse(), segment_rel_rmse()
│       │                         #   young_driver <30 (8.8%), high_value >28k (11.8%),
│       │                         #   high_power >130hp (10.9%), senior_driver ≥65 (9.5%)
│       ├── al_loop.py            # ALSimulation: weekly loop, multi-shock tariff change + restart
│       ├── perturbed_oracle.py   # PerturbedOracleEngine + preset perturbation functions
│       └── config.py             # YAML loaders, resolve_simulations, perturbation factory
├── outputs/
│   ├── figures/            # saved plots (not committed)
│   ├── models/             # oracle.pkl (not committed)
│   ├── warm_start/         # warm_start_X.parquet, warm_start_y.npy (not committed)
│   └── al_results/         # {strategy}.parquet — one file per strategy (committed)
├── app.py                  # Streamlit dashboard (3 tabs)
└── pyproject.toml
```

## Setup

**Prerequisites:** [pyenv](https://github.com/pyenv/pyenv) must be installed to manage the Python version. On macOS: `brew install pyenv`. For other platforms see the [pyenv installation docs](https://github.com/pyenv/pyenv#installation).

Pin the Python version for this project (creates a `.python-version` file):

```bash
pyenv local 3.12.8
```

Create and activate the virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running

```bash
# Phase 1 — fit the oracle (requires data in data/raw/)
.venv/bin/python notebooks/01_oracle.py

# Phase 2 — build warm start, then run AL simulation
.venv/bin/python notebooks/04_build_warm_start.py
.venv/bin/python notebooks/05_al_simulation.py

# Dashboard
streamlit run app.py
```

## Data

Download the dataset from [Mendeley Data — doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2) and place the file(s) in `data/raw/`.

## Acknowledgments

This study implements the **Cube Method** (Tillé and Deville, 2004), a concept I first encountered in Tillé's lectures almost a decade ago.
