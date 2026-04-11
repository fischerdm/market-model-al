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

**Warm start** seeds the competitor model with ~5,000 real rows (≈ one week's scraping budget), simulating organic quote requests arriving via the aggregator before the systematic AL loop begins.

The **weekly AL loop** generates fresh CP candidate profiles each week and uses a query strategy to select which anchors to scrape (label via the oracle) before retraining the competitor model.

| AL strategy | Query criterion | Deployable in practice? |
|---|---|---|
| Random | Uniform random baseline — no model required | Yes |
| Uncertainty | Anchors where bootstrap prediction variance is highest | Yes |
| Error-based | Anchors with highest expected relative residual (proxy model on labeled data) | Yes |
| Segment-adaptive | Anchors scored by global + per-segment relative RMSE; converges toward random as gaps close | Yes |
| Disruption-adaptive | Concentrates budget on segments with a sharp week-on-week RMSE spike; falls back to random otherwise | Yes |

Convergence is tracked in two metrics:
- **RMSE on holdout** — a fixed set of 5,000 real rows, oracle-labeled, never used during training. Measures prediction accuracy on a population-representative sample.
- **SHAP cosine similarity** *(simulation-only diagnostic)* — compares the competitor model's SHAP vectors to the oracle's on the holdout. Captures whether the tariff *structure* has been recovered, not just the premium levels. Requires oracle access, so it cannot be observed in real-world deployment.

**Tariff change simulation**: a `PerturbedOracleEngine` can be injected at one or more configurable weeks within a single simulation run to simulate a competitor repricing event (e.g. young-driver surcharge +20%, area repricing, or composed stacked shocks). Multiple shocks can be chained — the competitor model experiences all of them in one continuous timeline, with the RMSE curve measuring recovery of the *currently active* tariff at each point. Simulations and perturbation types are fully defined in YAML config files, with no code changes required to add new scenarios.

**Core research question**: does the AL strategy rediscover systematic ceteris paribus profiling on its own?

## Simulation findings (10-week run, 5 000 profiles/week)

Strategies operate on **anchor selection**: each week a pool of real anchor rows is scored, the best anchors are selected, and all their ceteris-paribus profiles are generated and labeled. The weekly profile count is derived from the budget: `n_anchors = weekly_budget // 254`.

| Finding | Detail |
|---|---|
| **Random beats sophisticated strategies globally** | On a population-representative holdout, random anchor sampling is competitive with or better than all informativeness-based strategies at 10 weeks. |
| **Error-based recovers young drivers faster** | Segment-level RMSE reveals that `error_based` converges faster than random on young drivers (age < 30) — the one segment where a sophisticated strategy wins. |
| **Root cause: informativeness vs. representativeness** | Greedy informativeness strategies pull budget toward high-signal edge cases, starving mainstream segments that random covers proportionally. |
| **Restart is not always optimal** | After a targeted tariff change (young-driver surcharge), a full restart discards valid labels from unchanged segments. The continuous strategy retains that data and can outperform restart on global RMSE. |
| **Disruption-adaptive** | Uses the week-on-week *change* in segment RMSE as a signal, not the absolute level. Fires on disruption, ignores permanently hard segments, and reverts to random once the gap closes. No labels are discarded. |

## Project structure

```
market-model-al/
├── config/
│   ├── simulation.yaml         # n_weeks, budget, strategies, metrics, simulations list
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
│   ├── 04_build_warm_start.py          # build warm start dataset (~5k real rows)
│   ├── 05_al_simulation.py             # run all simulations, save results + figures
│   └── 06_segment_summary.py           # segment distribution analysis + threshold calibration
├── src/
│   └── market_model_al/
│       ├── features.py           # feature engineering
│       ├── constraints.py        # physical validity rules (MIN_AGE_AT_LICENSING=18)
│       ├── oracle_engine.py      # OraclePricingEngine: query(profiles) → prices
│       ├── profile_generator.py  # ceteris-paribus candidate generation (ad hoc)
│       ├── competitor_model.py   # CompetitorModel: LightGBM, retrained each iteration
│       ├── strategies.py         # AL query strategies
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
│   └── al_results/         # results.parquet (not committed)
├── app.py                  # Streamlit dashboard (3 tabs)
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

# Dashboard
streamlit run app.py
```

## Data

Download the dataset from [Mendeley Data — doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2) and place the file(s) in `data/raw/`.
