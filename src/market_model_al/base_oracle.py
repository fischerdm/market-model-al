"""
Base oracle interface.

Subclass BaseOracle and implement query() to plug in a custom tariff —
a multiplicative GLM, a GBM loaded from file, or any other pricing engine.

Minimal example (multiplicative tariff)
----------------------------------------
    from market_model_al.base_oracle import BaseOracle
    import numpy as np
    import pandas as pd

    class MyTariff(BaseOracle):
        def query(self, profiles: pd.DataFrame) -> np.ndarray:
            base = 300.0
            age_factor  = np.where(profiles["driver_age"] < 25, 1.5, 1.0)
            power_factor = 1.0 + profiles["Power"].values / 1000.0
            return base * age_factor * power_factor

    oracle = MyTariff()
    prices = oracle.query(profiles[oracle.validate(profiles)])

Touch points for a custom tariff
----------------------------------
  1. Subclass BaseOracle here and implement query()
  2. Update config/features.yaml with your feature names and sweep ranges
  3. Adapt features.py for your raw-to-engineered column transformations
     (e.g. date columns → ages, jurisdiction-specific constraint constants)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseOracle(ABC):
    """Abstract pricing oracle.

    Defines the interface expected by ALSimulation and the profile generators.
    Both OraclePricingEngine (LightGBM-based) and PerturbedOracleEngine
    (tariff-change decorator) inherit from this class.
    """

    def validate(self, profiles: pd.DataFrame) -> pd.Series:
        """Return a boolean Series (index-aligned) — True where the profile
        is physically valid.  Delegates to ``constraints.validate``."""
        from market_model_al.constraints import validate as _validate
        return _validate(profiles)

    @abstractmethod
    def query(self, profiles: pd.DataFrame) -> np.ndarray:
        """Return predicted premiums for the given profiles.

        Parameters
        ----------
        profiles : pd.DataFrame
            Rows in engineered-feature space (same columns as the oracle
            expects after features.engineer_features()).

        Returns
        -------
        np.ndarray
            1-D array of predicted premiums, one per row.

        Raises
        ------
        ValueError
            If any profile fails physical validation.  Call validate() first
            to filter if needed.
        """
        ...
