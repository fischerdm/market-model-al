"""
Oracle pricing engine — wraps the fitted oracle model.

Accepts engineered feature profiles (driver_age, licence_age, vehicle_age, …)
and returns oracle-predicted premiums, enforcing physical validity constraints.

Usage
-----
    from market_model_al.oracle_engine import OraclePricingEngine

    engine = OraclePricingEngine("outputs/models/oracle.pkl")

    mask = engine.validate(profiles)          # boolean Series, True = valid
    prices = engine.query(profiles[mask])     # np.ndarray of premiums
"""

import joblib
import numpy as np
import pandas as pd

from market_model_al.base_oracle import BaseOracle
from market_model_al.features_config import CAT_FEATURES, CAT_FEATURES_OBJ


class OraclePricingEngine(BaseOracle):
    """LightGBM-based pricing oracle: query(profiles) -> premium array.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialised LightGBM oracle (joblib format).
    """

    def __init__(self, model_path):
        self._oracle = joblib.load(model_path)

    # ── public API ────────────────────────────────────────────────────────────

    def query(self, profiles: pd.DataFrame) -> np.ndarray:
        """Return oracle-predicted premiums for *valid* profiles.

        Raises
        ------
        ValueError
            If any row fails physical validation.  Call ``validate()`` first to
            filter if needed.
        """
        mask = self.validate(profiles)
        if not mask.all():
            n_bad = (~mask).sum()
            raise ValueError(
                f"{n_bad} profile(s) failed physical validation. "
                "Call validate() to filter before querying."
            )

        X = self._prepare(profiles)
        return self._oracle.predict(X)

    # ── internal ──────────────────────────────────────────────────────────────

    def _prepare(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Cast categorical columns to the dtype expected by LightGBM."""
        X = profiles.copy()
        for col in CAT_FEATURES + CAT_FEATURES_OBJ:
            if col in X.columns and X[col].dtype.name != "category":
                X[col] = X[col].astype("category")
        return X
