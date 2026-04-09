"""
Physical validity constraints for motor insurance profiles.

These rules are properties of the feature space, independent of any model.
Both the oracle engine and the profile generator use this module.
"""

import pandas as pd

# Minimum legal driving age and minimum licence-obtaining age.
MIN_DRIVING_AGE = 18.0
MIN_AGE_AT_LICENSING = 18.0  # Spain: minimum age to obtain a driving licence

# Per-feature lower bounds (upper bounds are not enforced — they are data
# artefacts, not physical laws).
LOWER_BOUNDS = {
    "driver_age":        MIN_DRIVING_AGE,
    "licence_age":       0.0,
    "vehicle_age":       0.0,
    "Power":             1.0,
    "Cylinder_capacity": 1.0,
    "Value_vehicle":     1.0,
    "Seniority":         0.0,
}


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
