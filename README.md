# Recursive Model Improvement via Active Learning in Non-Life Pricing

Active learning simulation for non-life pricing, modelling the process of scraping competitor quotes from aggregator websites (e.g. comparis.ch) to build a *nanny model*.

## Concept

An insurer can train a competitor model by scraping quoted premiums from aggregator platforms. This project simulates that iterative process in a controlled synthetic environment where the ground truth is known.

The simulation runs in three phases:

The real dataset is treated as the competitor's actual tariff. The copula and oracle together simulate a competitor's quoting engine: generate any policy profile, get a quote. The AL loop then tests how efficiently a nanny model can recover that engine from a limited scraping budget.

### Phases 1 & 2 — Learn the competitor's quoting engine

- Fit a **Gaussian copula** on one row per policy (latest renewal) → represents the competitor's book of business
- Fit a **LightGBM oracle** on `Premium ~ features` across all renewal years → the competitor's pricing engine
- Dataset: Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, [doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2)
- Validate oracle structure with **SHAP** (actuarial sanity check)
- Validate copula fit across earlier renewal years (marginal overlays — sanity check only)
- Drifts representing competitor repricings deferred to a later extension

### Phase 3 — Active learning loop

1. Warm start: ~50k labeled profiles — random copula samples + ceteris paribus profiles (select base profiles from the sample, vary one factor at a time)
2. Train the nanny model on the warm-start budget
3. Apply an AL query strategy to identify the next profiles to query
4. Label via the oracle and retrain
5. Repeat — tracking convergence in MSE and SHAP structure similarity to the oracle
6. Multiple AL strategies compared: uncertainty sampling, error-based, SHAP divergence

## Project structure

```
nanny-model/
├── data/
│   ├── raw/          # Lledó & Pavía (2024) data (not committed)
│   └── processed/    # engineered datasets and synthetic samples
├── notebooks/        # analysis scripts (numbered)
├── src/
│   └── nanny_model/  # reusable Python package
├── outputs/
│   ├── figures/      # saved plots
│   └── models/       # saved model artefacts
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
