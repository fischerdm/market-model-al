"""
Perturbed oracle — simulates a competitor re-tariffing event.

A PerturbedOracleEngine wraps an existing OraclePricingEngine and applies
a systematic premium shift to its predictions.  This lets the AL simulation
inject a mid-run tariff change and test whether the competitor model can
recover under different query strategies.

Usage
-----
    from market_model_al.perturbed_oracle import PerturbedOracleEngine, young_driver_surcharge

    base = OraclePricingEngine("outputs/models/oracle.pkl")
    new_oracle = PerturbedOracleEngine(base, young_driver_surcharge(factor=0.20))

    # Drop-in replacement wherever an oracle is expected
    prices = new_oracle.query(profiles)

Preset perturbations
--------------------
    young_driver_surcharge(factor)   — drivers < 30 pay +factor (e.g. 0.20 = +20 %)
    high_value_surcharge(factor)     — Value_vehicle > threshold pay +factor
    uniform_reprice(factor)          — all premiums multiplied by (1 + factor)
    area_reprice(area_map)           — per-area multiplier dict {area_code: multiplier}
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


# ── Engine ─────────────────────────────────────────────────────────────────────

class PerturbedOracleEngine:
    """Wraps an OraclePricingEngine and applies a post-prediction perturbation.

    Parameters
    ----------
    base_engine : OraclePricingEngine
        The original (unperturbed) oracle.
    perturbation_fn : callable
        A function ``(profiles: pd.DataFrame, prices: np.ndarray) -> np.ndarray``
        that returns the perturbed premiums.
    """

    def __init__(self, base_engine, perturbation_fn: Callable) -> None:
        self._base = base_engine
        self._perturb = perturbation_fn
        # Expose the underlying LightGBM model so SHAP explainers built outside
        # this class can still access the original model structure.
        self._oracle = base_engine._oracle

    def validate(self, profiles: pd.DataFrame) -> pd.Series:
        return self._base.validate(profiles)

    def query(self, profiles: pd.DataFrame) -> np.ndarray:
        prices = self._base.query(profiles)
        return self._perturb(profiles, prices)


# ── Preset perturbation factories ─────────────────────────────────────────────

def young_driver_surcharge(factor: float = 0.20, age_threshold: float = 30.0) -> Callable:
    """Drivers younger than age_threshold pay (1 + factor) × their base premium.

    Simulates a competitor repositioning away from young-driver risk.
    """
    def _apply(profiles: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
        multiplier = np.where(profiles["driver_age"].values < age_threshold, 1.0 + factor, 1.0)
        return prices * multiplier
    return _apply


def high_value_surcharge(factor: float = 0.15, value_threshold: float = 50_000.0) -> Callable:
    """Vehicles worth more than value_threshold pay (1 + factor) × base premium.

    Simulates repricing of high-value / prestige vehicles.
    """
    def _apply(profiles: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
        multiplier = np.where(
            profiles["Value_vehicle"].values > value_threshold, 1.0 + factor, 1.0
        )
        return prices * multiplier
    return _apply


def uniform_reprice(factor: float = 0.10) -> Callable:
    """Multiply all premiums by (1 + factor).

    Simulates an across-the-board price increase or decrease.
    """
    def _apply(profiles: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
        return prices * (1.0 + factor)
    return _apply


def area_reprice(area_factors: dict[int, float]) -> Callable:
    """Per-area multiplicative adjustment.

    Parameters
    ----------
    area_factors : dict
        Mapping of area code (int) → multiplicative factor (e.g. {1: 1.10, 3: 0.95}).
        Areas not in the dict are unchanged.

    Example
    -------
    >>> fn = area_reprice({1: 1.15, 2: 1.05, 3: 0.90})
    """
    def _apply(profiles: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
        multiplier = np.ones(len(profiles))
        if "Area" in profiles.columns:
            area_vals = profiles["Area"].values
            for code, factor in area_factors.items():
                multiplier = np.where(area_vals == code, factor, multiplier)
        return prices * multiplier
    return _apply


# ── Convenience: compose two perturbations ────────────────────────────────────

def compose(*perturbation_fns: Callable) -> Callable:
    """Chain multiple perturbation functions: each is applied left-to-right."""
    def _apply(profiles: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
        for fn in perturbation_fns:
            prices = fn(profiles, prices)
        return prices
    return _apply
