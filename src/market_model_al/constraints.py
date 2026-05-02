"""
Physical validity constraints for motor insurance profiles.

These rules are properties of the feature space, independent of any model.
Both the oracle engine and the profile generator use this module.

Constraint values (lower bounds and min_age_at_licensing) are loaded from
config/features.yaml — edit that file to adapt to a different jurisdiction.
"""

import pandas as pd

from market_model_al.features_config import LOWER_BOUNDS, MIN_AGE_AT_LICENSING

# Kept for backward compatibility (same value as MIN_AGE_AT_LICENSING = 18).
MIN_DRIVING_AGE = MIN_AGE_AT_LICENSING


def validate(profiles: pd.DataFrame) -> pd.Series:
    """Return a boolean Series (index-aligned) — True where the profile
    is physically valid.

    Checks applied:
    - Each bounded continuous feature is >= its lower bound.
    - ``licence_age <= driver_age - MIN_AGE_AT_LICENSING`` (can't have held
      a licence for longer than driver_age - minimum licensing age).
    """
    mask = pd.Series(True, index=profiles.index)

    for col, lb in LOWER_BOUNDS.items():
        if col in profiles.columns:
            mask &= profiles[col] >= lb

    if "driver_age" in profiles.columns and "licence_age" in profiles.columns:
        mask &= profiles["licence_age"] <= (profiles["driver_age"] - MIN_AGE_AT_LICENSING)

    return mask
