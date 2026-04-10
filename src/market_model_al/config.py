"""
Configuration loader for the AL simulation.

Reads simulation.yaml and tariff_changes.yaml, validates them, and provides
factory helpers that build perturbation functions and perturbed oracle engines
ready for use in the AL loop.

Usage
-----
    from market_model_al.config import load_simulation_cfg, load_tariff_changes_cfg
    from market_model_al.config import build_perturbed_oracle

    sim_cfg = load_simulation_cfg("config/simulation.yaml")
    tc_list = load_tariff_changes_cfg("config/tariff_changes.yaml")

    # sim_cfg keys: n_weeks, weekly_budget, candidate_multiplier, seed,
    #               strategies, metrics, compute_shap_similarity
    # tc_list: list of dicts with keys: name, label, week, perturbation,
    #           restart_strategies

    perturbed = build_perturbed_oracle(oracle_engine, tc["perturbation"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml


# ── Simulation config ─────────────────────────────────────────────────────────

_VALID_METRICS = {"rmse", "rel_rmse", "shap_cosine_similarity"}
_REQUIRED_METRICS = {"rmse"}   # always computed; cannot be turned off


def load_simulation_cfg(path: str | Path) -> dict[str, Any]:
    """Load and validate simulation.yaml.

    Returns a dict with keys:
        n_weeks, weekly_budget, candidate_multiplier, seed,
        strategies, metrics, compute_shap_similarity
    """
    raw = _load_yaml(path)

    cfg: dict[str, Any] = {}
    cfg["n_weeks"]             = int(raw["n_weeks"])
    cfg["weekly_budget"]       = int(raw["weekly_budget"])
    cfg["candidate_multiplier"] = int(raw["candidate_multiplier"])
    cfg["seed"]                = int(raw["seed"])
    cfg["strategies"]          = list(raw["strategies"])

    metrics = set(raw.get("metrics", list(_VALID_METRICS)))
    unknown = metrics - _VALID_METRICS
    if unknown:
        raise ValueError(f"simulation.yaml: unknown metrics {unknown}. "
                         f"Valid: {_VALID_METRICS}")
    metrics |= _REQUIRED_METRICS
    cfg["metrics"] = metrics
    cfg["compute_shap_similarity"] = "shap_cosine_similarity" in metrics

    return cfg


# ── Tariff-changes config ─────────────────────────────────────────────────────

_VALID_PERTURB_TYPES = {
    "young_driver_surcharge",
    "high_value_surcharge",
    "uniform_reprice",
    "area_reprice",
    "compose",
}


def load_tariff_changes_cfg(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate tariff_changes.yaml.

    Returns a list of dicts, each with keys:
        name, label, week, perturbation, restart_strategies
    """
    raw = _load_yaml(path)
    entries = raw.get("tariff_changes", [])
    result = []
    for i, entry in enumerate(entries):
        _validate_tc_entry(entry, index=i)
        result.append({
            "name":               str(entry["name"]),
            "label":              str(entry["label"]),
            "week":               int(entry["week"]),
            "perturbation":       dict(entry["perturbation"]),
            "restart_strategies": list(entry.get("restart_strategies") or []),
        })
    return result


# ── Perturbation factory ──────────────────────────────────────────────────────

def build_perturbation_fn(perturb_cfg: dict[str, Any]) -> Callable:
    """Build a perturbation callable from a perturbation config dict.

    The callable has signature:
        (profiles: pd.DataFrame, prices: np.ndarray) -> np.ndarray
    """
    from market_model_al.perturbed_oracle import (
        young_driver_surcharge,
        high_value_surcharge,
        uniform_reprice,
        area_reprice,
        compose,
    )

    ptype = perturb_cfg["type"]

    if ptype == "young_driver_surcharge":
        return young_driver_surcharge(
            factor=float(perturb_cfg["factor"]),
            age_threshold=float(perturb_cfg.get("age_threshold", 30.0)),
        )

    if ptype == "high_value_surcharge":
        return high_value_surcharge(
            factor=float(perturb_cfg["factor"]),
            value_threshold=float(perturb_cfg.get("value_threshold", 50_000.0)),
        )

    if ptype == "uniform_reprice":
        return uniform_reprice(factor=float(perturb_cfg["factor"]))

    if ptype == "area_reprice":
        raw_factors = perturb_cfg["area_factors"]
        area_factors = {int(k): float(v) for k, v in raw_factors.items()}
        return area_reprice(area_factors)

    if ptype == "compose":
        parts = [build_perturbation_fn(p) for p in perturb_cfg["parts"]]
        return compose(*parts)

    raise ValueError(f"Unknown perturbation type '{ptype}'. "
                     f"Valid types: {_VALID_PERTURB_TYPES}")


def build_perturbed_oracle(base_engine, perturb_cfg: dict[str, Any]):
    """Build a PerturbedOracleEngine from an oracle and a perturbation config dict."""
    from market_model_al.perturbed_oracle import PerturbedOracleEngine
    fn = build_perturbation_fn(perturb_cfg)
    return PerturbedOracleEngine(base_engine, fn)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_tc_entry(entry: dict, index: int) -> None:
    prefix = f"tariff_changes[{index}] (name={entry.get('name', '?')})"
    for key in ("name", "label", "week", "perturbation"):
        if key not in entry:
            raise ValueError(f"{prefix}: missing required field '{key}'")
    _validate_perturbation(entry["perturbation"], prefix)


def _validate_perturbation(perturb: dict, prefix: str) -> None:
    if "type" not in perturb:
        raise ValueError(f"{prefix}: perturbation is missing 'type'")
    ptype = perturb["type"]
    if ptype not in _VALID_PERTURB_TYPES:
        raise ValueError(f"{prefix}: unknown perturbation type '{ptype}'. "
                         f"Valid: {_VALID_PERTURB_TYPES}")
    if ptype == "compose":
        if "parts" not in perturb or not perturb["parts"]:
            raise ValueError(f"{prefix}: compose perturbation requires non-empty 'parts'")
        for i, part in enumerate(perturb["parts"]):
            _validate_perturbation(part, f"{prefix}.parts[{i}]")
    if ptype == "area_reprice" and "area_factors" not in perturb:
        raise ValueError(f"{prefix}: area_reprice requires 'area_factors'")
    if ptype in ("young_driver_surcharge", "high_value_surcharge",
                 "uniform_reprice") and "factor" not in perturb:
        raise ValueError(f"{prefix}: {ptype} requires 'factor'")
