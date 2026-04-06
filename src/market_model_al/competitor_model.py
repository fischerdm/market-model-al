"""
Competitor pricing model — retrained iteratively during the AL simulation.

Wraps a LightGBM regressor and exposes predict() and shap_values() so the
AL loop and convergence metrics can stay model-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from market_model_al.features import CAT_FEATURES, CAT_FEATURES_OBJ


# LightGBM hyperparameters for the competitor model.
# Intentionally simpler than the oracle to simulate an adversary working from
# limited scraped data.
_DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class CompetitorModel:
    """LightGBM-based competitor model, retrained from scratch on each call to fit().

    Parameters
    ----------
    params : dict, optional
        Override any LightGBM hyperparameter from the defaults above.
    """

    def __init__(self, params: dict | None = None) -> None:
        p = {**_DEFAULT_PARAMS}
        if params:
            p.update(params)
        self._params = p
        self._model: lgb.LGBMRegressor | None = None
        self._explainer: shap.TreeExplainer | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CompetitorModel":
        """Train the model on labeled profiles.  Clears any cached explainer."""
        self._model = lgb.LGBMRegressor(**self._params)
        self._model.fit(self._prepare(X), y)
        self._explainer = None  # invalidated after refit
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted premiums for profiles in X."""
        self._require_fitted()
        return self._model.predict(self._prepare(X))

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values (n_samples × n_features) for profiles in X.

        The explainer is cached after the first call and reused until the model
        is retrained with fit().
        """
        self._require_fitted()
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer.shap_values(self._prepare(X))

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ── internal ──────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in CAT_FEATURES + CAT_FEATURES_OBJ:
            if col in df.columns and df[col].dtype.name != "category":
                df[col] = df[col].astype("category")
        return df

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("CompetitorModel has not been fitted.  Call fit() first.")
