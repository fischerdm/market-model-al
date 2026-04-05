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

from market_model_al.features import CAT_FEATURES, CAT_FEATURES_OBJ


# Physical lower bounds for continuous features.
# Upper bounds are not enforced — they are data-range artefacts, not laws.
_LOWER_BOUNDS: dict[str, float] = {
    "driver_age":        18.0,   # minimum legal driving age
    "licence_age":        0.0,   # can't have held a licence for negative time
    "vehicle_age":        0.0,   # car can't be newer than registration year
    "Power":              1.0,
    "Cylinder_capacity":  1.0,
    "Value_vehicle":      1.0,
    "Seniority":          0.0,
}

# Minimum age at which a driving licence can be obtained (used in cross-check).
_MIN_LICENCE_AGE = 16.0


class OraclePricingEngine:
    """Stateless pricing oracle: query(profiles) -> premium array.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialised LightGBM oracle (joblib format).
    """

    def __init__(self, model_path):
        self._oracle = joblib.load(model_path)

    # ── public API ────────────────────────────────────────────────────────────

    def validate(self, profiles: pd.DataFrame) -> pd.Series:
        """Return a boolean Series (index-aligned) — True where the profile
        is physically valid.

        Checks applied:
        - Each bounded continuous feature is >= its lower bound.
        - ``licence_age <= driver_age - _MIN_LICENCE_AGE`` (can't have obtained
          a licence before the minimum licence age).
        """
        mask = pd.Series(True, index=profiles.index)

        for col, lb in _LOWER_BOUNDS.items():
            if col in profiles.columns:
                mask &= profiles[col] >= lb

        if "driver_age" in profiles.columns and "licence_age" in profiles.columns:
            mask &= profiles["licence_age"] <= (
                profiles["driver_age"] - _MIN_LICENCE_AGE
            )

        return mask

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
