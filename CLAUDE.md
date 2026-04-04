# Project context for Claude

## What this project is

A simulation of the *nanny model* use case in non-life insurance pricing: an insurer scrapes competitor quotes from aggregator websites and trains a model on them. This project builds a controlled synthetic environment to test active learning (AL) query strategies, where the ground truth is known.

## Architecture decisions

### Dataset
Lledó, Josep; Pavía, Jose M. (2024), *Dataset of an actual motor vehicle insurance portfolio*, Mendeley Data V2, doi: 10.17632/5cxyb5fp4f.2.

Chosen because it contains an actual `Premium` (net premium) column — a direct proxy for a competitor quote. No frequency/severity modelling needed.

### Three-phase pipeline

1. **Learn from real data** — fit a generative model to the joint feature distribution; fit LightGBM on `Premium ~ features` as the **oracle** (the "true" competitor tariff)
2. **Synthetic world** — sample unlimited profiles from the generative model, label with the oracle; inject controlled drifts (feature marginal shifts, oracle weight perturbations)
3. **AL simulation loop** — start with a small labeled budget, apply a query strategy, retrain the nanny model, repeat; track convergence in MSE and SHAP structure similarity to the oracle

### Generative model choice
Start simple: **Gaussian copula + parametric marginals** (e.g. SDV `GaussianCopulaSynthesizer`). The synthetic world only needs to represent interesting relationships, not perfectly replicate reality.

Alternatives considered but deferred:
- **CTGAN / TVAE** — better at complex non-linear dependencies in mixed-type tabular data, but a black box; revisit if the copula fails to capture important structure

### SHAP
Used in two places:
- On the **oracle** — validate the learned tariff structure looks actuarially sensible
- On the **nanny model** — track recovery of the oracle's SHAP structure across AL iterations (a richer convergence metric than MSE alone)

## Stack
Python 3.12, LightGBM, SHAP, SDV (copula synthesis). All notebooks are `.py` files (numbered), not `.ipynb`.
