"""
Features configuration loader.

Reads config/features.yaml and exposes the computed constants used across
features.py, constraints.py, and profile_generator.py.  Edit features.yaml
to adapt the simulation to a different tariff or market — no Python changes
needed for feature names, sweep ranges, or constraint bounds.

Exported constants
------------------
CONTINUOUS_FEATURES      : list[str]            — ordered list of continuous feature names
DEFAULT_RANGES           : dict[str, np.ndarray] — CP sweep grid per feature
LOWER_BOUNDS             : dict[str, float]      — per-feature lower bounds for validation
CAT_FEATURES             : list[str]             — integer-coded categorical features
CAT_FEATURES_OBJ         : list[str]             — object-dtype categorical features
MIN_AGE_AT_LICENSING     : float                 — cross-feature constraint constant
"""

from pathlib import Path

import numpy as np
import yaml

_CFG_PATH = Path(__file__).parent.parent.parent / "config" / "features.yaml"


def _load():
    if not _CFG_PATH.exists():
        raise FileNotFoundError(
            f"features.yaml not found at {_CFG_PATH}. "
            "Make sure config/features.yaml exists relative to the project root."
        )
    with open(_CFG_PATH) as f:
        raw = yaml.safe_load(f)

    cont = raw["continuous"]
    cat  = raw["categorical"]
    cnst = raw.get("constraints", {})

    continuous_features: list[str] = list(cont.keys())
    default_ranges: dict[str, np.ndarray] = {}
    lower_bounds: dict[str, float] = {}

    for name, spec in cont.items():
        sw = spec["sweep"]
        stype = sw["type"]
        if stype == "arange":
            default_ranges[name] = np.arange(sw["start"], sw["stop"], sw["step"])
        elif stype == "geomspace":
            default_ranges[name] = np.geomspace(sw["start"], sw["stop"], int(sw["npoints"]))
        else:
            raise ValueError(
                f"features.yaml: unknown sweep type '{stype}' for '{name}'. "
                "Valid: 'arange', 'geomspace'."
            )
        if "lower_bound" in spec:
            lower_bounds[name] = float(spec["lower_bound"])

    cat_features     = list(cat.get("int_coded", []))
    cat_features_obj = list(cat.get("object_coded", []))
    min_age_at_licensing = float(cnst.get("min_age_at_licensing", 18.0))

    return (
        continuous_features,
        default_ranges,
        lower_bounds,
        cat_features,
        cat_features_obj,
        min_age_at_licensing,
    )


(
    CONTINUOUS_FEATURES,
    DEFAULT_RANGES,
    LOWER_BOUNDS,
    CAT_FEATURES,
    CAT_FEATURES_OBJ,
    MIN_AGE_AT_LICENSING,
) = _load()
