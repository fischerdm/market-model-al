# Recursive Model Improvement via Active Learning in Non-Life Pricing

Active learning simulation for non-life pricing, modelling the process of scraping competitor quotes from aggregator websites (e.g. comparis.ch) to build a *nanny model*.

## Concept

An insurer can train a competitor model by scraping quoted premiums from aggregator platforms. This project simulates that iterative process in a controlled synthetic environment where the ground truth is known.

The simulation runs in three phases:

### Phase 1 — Learn from real data

- Fit a **Gaussian copula** to the joint feature distribution of real motor policies
- Fit a **LightGBM regression** on `Premium ~ features` → this becomes the **oracle** (the "true" competitor tariff)
- Dataset: Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, [doi: 10.17632/5cxyb5fp4f.2](https://doi.org/10.17632/5cxyb5fp4f.2)
- The oracle's structure is analysed with **SHAP** to validate it looks actuarially sensible

### Phase 2 — Synthetic world

- Sample unlimited policy profiles from the copula
- Label them using the oracle (± noise)
- Controlled drifts can be injected: shift feature marginals, perturb oracle weights
- Known ground truth enables objective measurement of AL performance

### Phase 3 — Active learning loop

1. Start with a small labeled budget (sampled oracle quotes)
2. Train the nanny model on observed profiles
3. Apply an AL query strategy to select the next profiles to query
4. Re-label via the oracle and retrain
5. Repeat — tracking convergence in MSE and SHAP structure similarity to the oracle

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
