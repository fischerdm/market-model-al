# Recursive Model Improvement via Active Learning in Non-Life Pricing

Active learning simulation for non-life pricing, modelling the process of scraping competitor quotes from aggregator websites (e.g. comparis.ch) to build a *competitor model*.

## Concept

An insurer can train a competitor model by scraping quoted premiums from aggregator platforms. This project simulates that iterative process in a controlled synthetic environment where the ground truth is known.

The real dataset is treated as the competitor's actual tariff. A LightGBM oracle learns that tariff and can return a premium for any policy profile — simulating the function of an aggregator like comparis.ch. The AL loop then tests how efficiently a competitor model can recover the oracle from a limited scraping budget.

The end deliverable is a **Streamlit dashboard** for interactively exploring and comparing AL query strategies.

### Phase 1 — Oracle: learn the competitor's pricing engine

- Fit a **LightGBM oracle** on `Premium ~ features` across all renewal years → the competitor's pricing engine
- Dataset: Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, [doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2)
- Validate oracle structure with **SHAP dependence plots** (driver age U-curve, vehicle power, key interactions — actuarial sanity check)

### Phase 2 — Active learning loop

Candidate profiles are generated ad hoc each week via a ceteris-paribus approach: real rows serve as anchor points and continuous features are swept one at a time across their range (e.g. `driver_age` 18→80, all other features held fixed). This mirrors how scraping is done in practice.

**Warm start** seeds the competitor model with two components:
- **(A) Real rows** — a random sample from the Spanish dataset, simulating organic quote requests arriving via comparis
- **(B) CP profiles** — structured ceteris-paribus sweeps from a small set of anchors, simulating an initial systematic scrape

The **weekly AL loop** then generates fresh CP candidate profiles each week and uses a query strategy to select which ones to actually scrape (label via the oracle) before retraining the competitor model.

| AL strategy | Query criterion |
|---|---|
| Random | Uniform random baseline |
| Uncertainty sampling | Profiles where bootstrap prediction variance is highest |
| Error-based | Profiles where expected absolute residual is highest (proxy model) |
| SHAP divergence | Profiles where oracle and competitor SHAP vectors diverge most |

Convergence is tracked in two metrics: RMSE against the oracle, and SHAP cosine similarity — capturing whether the competitor model has learned the same tariff *factors*, not just the same premium levels.

**Tariff change simulation**: a `PerturbedOracleEngine` can be injected mid-run at a configurable week to simulate a competitor repricing event (e.g. young-driver surcharge +20%). This lets practitioners compare *continue* vs *restart* strategies and assess whether the weekly continuous scraping rate is sufficient to track the new tariff.

**Core research question**: does the AL strategy rediscover systematic ceteris paribus profiling on its own?

## Project structure

```
market-model-al/
├── data/
│   ├── raw/                # Lledó & Pavía (2024) data (not committed)
│   └── processed/          # intermediate processed data (not committed)
├── docs/
│   └── index.html          # GitHub Pages project page
├── notebooks/              # numbered .py scripts (run in order)
│   ├── 01_oracle.py                    # fit and validate the oracle
│   ├── 02_oracle_engine_smoke.py       # smoke test: OraclePricingEngine
│   ├── 03_profile_generator_smoke.py   # smoke test: ceteris-paribus generator
│   ├── 04_build_warm_start.py          # build warm start dataset (real + CP mix)
│   └── 05_al_simulation.py             # run AL strategies, save results + figures
├── src/
│   └── market_model_al/
│       ├── features.py           # feature engineering
│       ├── constraints.py        # physical validity rules (MIN_AGE_AT_LICENSING=18)
│       ├── oracle_engine.py      # OraclePricingEngine: query(profiles) → prices
│       ├── profile_generator.py  # ceteris-paribus candidate generation (ad hoc)
│       ├── competitor_model.py   # CompetitorModel: LightGBM, retrained each iteration
│       ├── strategies.py         # AL query strategies (random, uncertainty, error_based, shap_divergence)
│       ├── al_loop.py            # ALSimulation: weekly loop, tariff change support
│       └── perturbed_oracle.py   # PerturbedOracleEngine + preset perturbation functions
├── outputs/
│   ├── figures/            # saved plots (not committed)
│   ├── models/             # oracle.pkl (not committed)
│   ├── warm_start/         # warm_start_X.parquet, warm_start_y.npy (not committed)
│   └── al_results/         # results.parquet (not committed)
└── pyproject.toml
```

## Setup

**Prerequisites:** [pyenv](https://github.com/pyenv/pyenv) must be installed (`brew install pyenv`) to manage the Python version.

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
```

## Data

Download the dataset from [Mendeley Data — doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2) and place the file(s) in `data/raw/`.
