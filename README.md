# Recursive Model Improvement via Active Learning in Non-Life Pricing

Active learning simulation for non-life pricing, modelling the process of scraping competitor quotes from aggregator websites (e.g. comparis.ch) to build a *competitor model*.

## Concept

An insurer can train a competitor model by scraping quoted premiums from aggregator platforms. This project simulates that iterative process in a controlled synthetic environment where the ground truth is known.

The real dataset is treated as the competitor's actual tariff. A LightGBM oracle learns that tariff and can return a premium for any policy profile — simulating the function of an aggregator like comparis.ch. The AL loop then tests how efficiently a competitor model can recover the oracle from a limited scraping budget.

The end deliverable is a **Streamlit dashboard** for interactively exploring and comparing AL query strategies.

### Phase 1 — Oracle: learn the competitor's pricing engine

- Fit a **LightGBM oracle** on `Premium ~ features` across all renewal years → the competitor's pricing engine
- Dataset: Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, [doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2)
- Validate oracle structure with **SHAP dependence plots** (driver age curve, vehicle power, key interactions — actuarial sanity check)

### Phase 2 — Active learning loop

The profile pool is built by taking real rows as anchor points and varying continuous features in small steps (e.g. `driver_age` 18→70, keeping all other features fixed) — mirroring how scraping is done in practice on aggregators. The oracle labels every generated profile.

1. Warm start: ~50k labeled profiles from the pool
2. Train the competitor model on the warm-start budget
3. Apply an AL query strategy to identify the next profiles to query
4. Label via the oracle and retrain
5. Repeat — tracking convergence in MSE and SHAP structure similarity to the oracle
6. Multiple AL strategies compared: uncertainty sampling, error-based, SHAP divergence

**Core research question**: does the AL strategy rediscover systematic ceteris paribus profiling on its own?

## Project structure

```
market-model-al/
├── data/
│   ├── raw/              # Lledó & Pavía (2024) data (not committed)
│   └── processed/        # engineered datasets and synthetic samples
├── notebooks/            # analysis scripts (numbered .py files)
├── src/
│   └── market_model_al/  # reusable Python package
├── outputs/
│   ├── figures/          # saved plots (not committed)
│   └── models/           # saved model artefacts (not committed)
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

## Data

Download the dataset from [Mendeley Data — doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2) and place the file(s) in `data/raw/`.
