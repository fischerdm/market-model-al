"""
Feature engineering for the Lledó & Pavía (2024) motor insurance dataset.

The raw dataset contains one row per policy-year (renewal). This module
engineers date-derived features and drops columns that are not observable
at quote time (claim outcomes, raw dates, administrative lapse info).
"""

import pandas as pd

# Claim outcomes and admin columns not available at quote time
_DROP_COLS = [
    "Date_start_contract", "Date_next_renewal",
    "Date_birth", "Date_driving_licence", "Date_lapse", "Year_matriculation",
    "Lapse",
    "Cost_claims_year", "N_claims_year", "N_claims_history", "R_Claims_history",
]

# Integer-coded columns that are categorical in nature
CAT_FEATURES = [
    "Distribution_channel", "Payment", "Type_risk", "Area",
    "Second_driver", "N_doors",
]
# Type_fuel is object dtype and also categorical
CAT_FEATURES_OBJ = ["Type_fuel"]


def load_raw(path) -> pd.DataFrame:
    """Load raw CSV; parse date columns. Keeps ID and Date_last_renewal for
    downstream filtering (e.g. latest-renewal-per-policy selection)."""
    df = pd.read_csv(path, sep=";", na_values=["NA"])
    for col in ["Date_last_renewal", "Date_birth", "Date_driving_licence"]:
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a raw (loaded) dataframe into model-ready features.

    Expects Date_last_renewal, Date_birth, Date_driving_licence, and
    Year_matriculation to be present. ID and Date_last_renewal are dropped
    after use; caller should filter rows beforehand if needed.

    Returns a dataframe with engineered features plus Premium (target).
    Categorical columns are cast to pandas Categorical dtype.
    """
    df = df.copy()

    # Date-derived features; reference point is Date_last_renewal per row
    df["driver_age"] = (df["Date_last_renewal"] - df["Date_birth"]).dt.days / 365.25
    df["licence_age"] = (df["Date_last_renewal"] - df["Date_driving_licence"]).dt.days / 365.25
    df["vehicle_age"] = df["Date_last_renewal"].dt.year - df["Year_matriculation"]

    df = df.drop(columns=["ID", "Date_last_renewal"] + _DROP_COLS)

    for col in CAT_FEATURES + CAT_FEATURES_OBJ:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
