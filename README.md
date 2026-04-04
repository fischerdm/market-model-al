# Recursive Model Improvement via Active Learning in Non-Life Pricing

Active learning simulation for non-life pricing, using the **freMTPL2** French motor TPL dataset as a proxy for competitor model scraping (e.g. as done on aggregators like comparis.ch).

## Concept

In non-life insurance pricing, a *nanny model* (or competitor model) can be trained by scraping quoted premiums from aggregator websites. This project simulates that iterative process:

1. Train a model on observed data
2. Measure accuracy / uncertainty
3. Generate new profiles (active learning query strategy)
4. Re-train and repeat

## Project structure

```
nanny-model/
├── data/
│   ├── raw/          # freMTPL2freq.csv, freMTPL2sev.csv (not committed)
│   └── processed/    # engineered datasets
├── notebooks/        # analysis scripts (numbered)
│   └── 01_shap_interaction_analysis.py
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

Download from [Kaggle – French Motor Claims Datasets freMTPL2](https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq) and place the two CSVs in `data/raw/`.

## Usage

```bash
python notebooks/01_shap_interaction_analysis.py
```
