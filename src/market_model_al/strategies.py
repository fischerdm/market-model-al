"""
Active learning query strategies for the competitor-model simulation.

Each strategy selects the *indices into unlabeled_X* of the next batch to label.
The caller is responsible for translating those local indices back to pool indices.

Strategies
----------
random          — uniform random baseline (no model required)
uncertainty     — bootstrap variance across lightweight ensemble members
error_based     — expected absolute error predicted by a proxy model
shap_divergence — profiles where oracle SHAP and competitor SHAP diverge most

All strategies accept a NumPy RNG for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap


# ── helpers ───────────────────────────────────────────────────────────────────


def _drop_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Convert category columns to their string codes.

    The quick bootstrap and proxy models only need variance / error estimates —
    they don't require LightGBM's categorical optimisation.  Converting to
    strings avoids category-level mismatches between training and prediction
    data when candidate profiles cover a subset of category values.
    """
    result = X.copy()
    for col in X.select_dtypes("category").columns:
        result[col] = result[col].astype(str)
    return result


def _train_quick_lgb(
    X: pd.DataFrame,
    y: np.ndarray,
    rng: np.random.Generator,
) -> lgb.LGBMRegressor:
    """Train a small, fast LightGBM model (used for bootstrap and proxy models)."""
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=int(rng.integers(0, 2**31)),
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(_drop_categoricals(X), y)
    return model


# ── strategy functions ────────────────────────────────────────────────────────


def random_query(
    unlabeled_X: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n profiles uniformly at random."""
    return rng.choice(len(unlabeled_X), size=n, replace=False)


def uncertainty_query(
    competitor,  # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    unlabeled_X: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    n_boot: int = 7,
    uncertainty_pool_size: int = 20_000,
) -> np.ndarray:
    """Select n profiles with the highest prediction variance across bootstrap models.

    To keep runtime tractable, variance is estimated on a random subsample of the
    unlabeled pool (uncertainty_pool_size rows), then the top-n are returned.
    If unlabeled_X is smaller than uncertainty_pool_size, all rows are scored.
    """
    # Score a subsample for efficiency
    if len(unlabeled_X) > uncertainty_pool_size:
        sample_idx = rng.choice(len(unlabeled_X), size=uncertainty_pool_size, replace=False)
    else:
        sample_idx = np.arange(len(unlabeled_X))

    candidate_X = unlabeled_X.iloc[sample_idx]

    preds = np.zeros((n_boot, len(candidate_X)))
    for i in range(n_boot):
        boot_idx = rng.choice(len(labeled_X), size=len(labeled_X), replace=True)
        boot_model = _train_quick_lgb(
            labeled_X.iloc[boot_idx], labeled_y[boot_idx], rng
        )
        preds[i] = boot_model.predict(_drop_categoricals(candidate_X))

    scores = preds.std(axis=0)
    top_k = np.argsort(-scores)[: n]          # best candidates in subsample
    return sample_idx[top_k]                  # map back to unlabeled_X indices


def error_based_query(
    competitor,  # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    unlabeled_X: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    error_pool_size: int = 20_000,
) -> np.ndarray:
    """Select n profiles with the highest expected absolute error.

    Trains a proxy LightGBM model on (labeled_X, |residuals|), then predicts
    expected error magnitude on the unlabeled pool to rank candidates.
    """
    residuals = np.abs(labeled_y - competitor.predict(labeled_X))
    proxy = _train_quick_lgb(labeled_X, residuals, rng)

    if len(unlabeled_X) > error_pool_size:
        sample_idx = rng.choice(len(unlabeled_X), size=error_pool_size, replace=False)
    else:
        sample_idx = np.arange(len(unlabeled_X))

    candidate_X = unlabeled_X.iloc[sample_idx]
    scores = proxy.predict(_drop_categoricals(candidate_X))
    top_k = np.argsort(-scores)[:n]
    return sample_idx[top_k]


def shap_divergence_query(
    oracle_explainer: shap.TreeExplainer,
    competitor,  # CompetitorModel
    unlabeled_X: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    shap_pool_size: int = 5_000,
) -> np.ndarray:
    """Select n profiles where oracle and competitor SHAP vectors diverge most.

    SHAP is computed on a random subsample of the unlabeled pool (shap_pool_size)
    to keep runtime feasible.  Divergence is measured as the L2 norm of the
    per-row difference between oracle and competitor SHAP vectors.
    """
    if len(unlabeled_X) > shap_pool_size:
        sample_idx = rng.choice(len(unlabeled_X), size=shap_pool_size, replace=False)
    else:
        sample_idx = np.arange(len(unlabeled_X))

    candidate_X = unlabeled_X.iloc[sample_idx]

    oracle_shap = oracle_explainer.shap_values(candidate_X)      # (k, F)
    competitor_shap = competitor.shap_values(candidate_X)         # (k, F)

    divergence = np.linalg.norm(oracle_shap - competitor_shap, axis=1)
    top_k = np.argsort(-divergence)[:n]
    return sample_idx[top_k]


# ── registry ──────────────────────────────────────────────────────────────────

STRATEGIES = ["random", "uncertainty", "error_based", "shap_divergence"]
