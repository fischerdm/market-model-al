"""
Active learning query strategies for the competitor-model simulation.

Each strategy selects *which anchor rows* to scrape next week.  The caller
generates ceteris-paribus profiles from the selected anchors and labels them
via the oracle.  Strategies therefore decide *where in feature space* to focus
the scraping effort, not which individual profiles to label.

Strategies
----------
random          — select anchors uniformly at random (baseline)
uncertainty     — anchors where the competitor model is most uncertain
                  (bootstrap variance of predictions at the anchor point)
error_based     — anchors with the highest expected absolute prediction error
                  (proxy model trained on labeled residuals)
shap_divergence — anchors where oracle and competitor SHAP vectors diverge most
                  (L2 norm of per-feature SHAP difference at the anchor)

All strategies accept a NumPy RNG for reproducibility.

Interface
---------
Each function receives:
    anchor_pool   : pd.DataFrame  — candidate anchors to score (real data rows)
    n             : int           — number of anchors to select
    ...model args...
    rng           : np.random.Generator

Returns:
    np.ndarray of integer indices into anchor_pool (length n).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap


# ── helpers ───────────────────────────────────────────────────────────────────


def _encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Convert category columns to integer codes.

    Bootstrap and proxy models only need variance / error estimates and don't
    require LightGBM's categorical optimisation.  Integer codes avoid
    category-level mismatches when anchor pools cover only a subset of levels.
    """
    result = X.copy()
    for col in X.select_dtypes("category").columns:
        result[col] = result[col].cat.codes
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
    model.fit(_encode_categoricals(X), y)
    return model


# ── strategy functions ────────────────────────────────────────────────────────


def random_query(
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n anchors uniformly at random."""
    return rng.choice(len(anchor_pool), size=n, replace=False)


def uncertainty_query(
    competitor,             # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    n_boot: int = 7,
) -> np.ndarray:
    """Select n anchors where the competitor model is most uncertain.

    Uncertainty is estimated as the standard deviation of predictions across
    bootstrap-resampled models, evaluated directly at each candidate anchor.
    Anchors with high variance indicate regions where the competitor model
    lacks confidence — exactly where additional scraping is most valuable.
    """
    preds = np.zeros((n_boot, len(anchor_pool)))
    for i in range(n_boot):
        boot_idx = rng.choice(len(labeled_X), size=len(labeled_X), replace=True)
        boot_model = _train_quick_lgb(
            labeled_X.iloc[boot_idx], labeled_y[boot_idx], rng
        )
        preds[i] = boot_model.predict(_encode_categoricals(anchor_pool))

    scores = preds.std(axis=0)
    return np.argsort(-scores)[:n]


def error_based_query(
    competitor,             # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n anchors with the highest expected absolute prediction error.

    A proxy LightGBM model is trained on (labeled_X, |residuals|) and used to
    predict the expected error magnitude at each candidate anchor.  Anchors
    where the competitor model is likely most wrong are prioritised.
    """
    residuals = np.abs(labeled_y - competitor.predict(labeled_X))
    proxy = _train_quick_lgb(labeled_X, residuals, rng)
    scores = proxy.predict(_encode_categoricals(anchor_pool))
    return np.argsort(-scores)[:n]


def shap_divergence_query(
    oracle_explainer: shap.TreeExplainer,
    competitor,             # CompetitorModel
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n anchors where oracle and competitor SHAP vectors diverge most.

    SHAP values are computed at each candidate anchor for both the oracle and
    the competitor model.  Divergence is measured as the L2 norm of the
    per-feature SHAP difference.  High divergence means the competitor model
    has a fundamentally different attribution of risk factors in that region —
    the most informative region to scrape next.
    """
    oracle_shap     = oracle_explainer.shap_values(anchor_pool)       # (k, F)
    competitor_shap = competitor.shap_values(anchor_pool)              # (k, F)
    divergence      = np.linalg.norm(oracle_shap - competitor_shap, axis=1)
    return np.argsort(-divergence)[:n]


# ── registry ──────────────────────────────────────────────────────────────────

STRATEGIES = ["random", "uncertainty", "error_based", "shap_divergence"]
