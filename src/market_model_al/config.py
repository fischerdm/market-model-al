"""
Configuration loader for the AL simulation.

Reads simulation.yaml and tariff_changes.yaml, validates them, and provides
factory helpers that build perturbation functions and perturbed oracle engines.

Usage
-----
    from market_model_al.config import load_simulation_cfg, load_tariff_changes_cfg
    from market_model_al.config import resolve_simulations, build_perturbed_oracle

    sim_cfg    = load_simulation_cfg("config/simulation.yaml")
    tc_library = load_tariff_changes_cfg("config/tariff_changes.yaml")
    simulations = resolve_simulations(sim_cfg, tc_library)

    # sim_cfg keys:
    #   n_weeks, weekly_budget, anchor_space_multiplier, selection_fraction, seed,
    #   strategies, metrics, compute_shap_similarity, restart_strategies,
    #   market_supplement_ratio, market_profile_method, market_n_anchors,
    #   gaussian_sigma_frac

    # simulations: list of dicts with keys:
    #   name, label, has_tariff_changes,
    #   tariff_changes: list of (week: int, perturb_cfg: dict) sorted by week

    perturbed = build_perturbed_oracle(oracle, perturb_cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml


# ── Simulation config ─────────────────────────────────────────────────────────

_VALID_METRICS   = {"rmse", "rel_rmse", "shap_cosine_similarity"}
_REQUIRED_METRICS = {"rmse"}


def load_simulation_cfg(path: str | Path) -> dict[str, Any]:
    """Load and validate simulation.yaml.

    Returns a dict with keys:
        n_weeks, weekly_budget, candidate_multiplier, seed,
        strategies, restart_strategies, metrics, compute_shap_similarity,
        simulations (raw list — pass to resolve_simulations for full resolution)
    """
    raw = _load_yaml(path)

    cfg: dict[str, Any] = {}
    cfg["n_weeks"]             = int(raw["n_weeks"])
    cfg["weekly_budget"]       = int(raw["weekly_budget"])
    cfg["seed"]                = int(raw["seed"])
    cfg["strategies"]          = list(raw["strategies"])
    cfg["restart_strategies"]  = list(raw.get("restart_strategies") or [])
    cfg["simulations_raw"]     = list(raw.get("simulations") or [])

    advanced = raw.get("advanced") or {}
    cfg["anchor_space_multiplier"] = int(advanced.get("anchor_space_multiplier", 30))
    cfg["selection_fraction"]      = float(advanced.get("selection_fraction", 0.10))
    cfg["market_supplement_ratio"] = float(advanced.get("market_supplement_ratio", 0.10))
    cfg["market_profile_method"]   = str(advanced.get("market_profile_method", "cp"))

    rm = advanced.get("random_market") or {}
    cfg["market_n_anchors"] = int(rm.get("market_n_anchors", 50))

    cfg["gaussian_sigma_frac"]      = float(advanced.get("gaussian_sigma_frac", 0.3))

    metrics = set(raw.get("metrics", list(_VALID_METRICS)))
    unknown = metrics - _VALID_METRICS
    if unknown:
        raise ValueError(f"simulation.yaml: unknown metrics {unknown}. "
                         f"Valid: {_VALID_METRICS}")
    metrics |= _REQUIRED_METRICS
    cfg["metrics"] = metrics
    cfg["compute_shap_similarity"] = "shap_cosine_similarity" in metrics

    return cfg


# ── Tariff-changes library ────────────────────────────────────────────────────

_VALID_PERTURB_TYPES = {
    "young_driver_surcharge",
    "high_value_surcharge",
    "uniform_reprice",
    "area_reprice",
}


def load_tariff_changes_cfg(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate tariff_changes.yaml.

    Returns a dict of name → perturbation-param dict.
    """
    raw = _load_yaml(path)
    library: dict[str, dict] = {}
    for name, params in raw.items():
        _validate_perturbation(params, context=name)
        library[name] = dict(params)
    return library


# ── Simulation resolution ─────────────────────────────────────────────────────

def resolve_simulations(
    sim_cfg: dict[str, Any],
    tc_library: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve the simulations list from sim_cfg against the perturbation library.

    Returns a list of simulation dicts, each with keys:
        name               : slug used in results parquet and figure filenames
        label              : human-readable description for plots / dashboard
        has_tariff_changes : bool — False for baseline runs
        tariff_changes     : list of (week: int, perturb_cfg: dict) sorted by week
    """
    simulations = []
    for i, entry in enumerate(sim_cfg["simulations_raw"]):
        if "name" not in entry:
            raise ValueError(f"simulation.yaml simulations[{i}]: missing 'name'")
        if "label" not in entry:
            raise ValueError(f"simulation.yaml simulations[{i}] ({entry['name']}): missing 'label'")

        raw_changes = entry.get("tariff_changes") or []
        resolved_changes: list[tuple[int, dict]] = []

        for j, change in enumerate(raw_changes):
            if "week" not in change or "perturbations" not in change:
                raise ValueError(
                    f"simulation.yaml simulations[{i}].tariff_changes[{j}]: "
                    "each entry must have 'week' and 'perturbations'"
                )
            week  = int(change["week"])
            names = list(change["perturbations"])

            for n in names:
                if n not in tc_library:
                    raise ValueError(
                        f"simulation.yaml references unknown perturbation '{n}'. "
                        f"Defined in tariff_changes.yaml: {list(tc_library)}"
                    )

            if len(names) == 1:
                perturb_cfg = tc_library[names[0]]
            else:
                perturb_cfg = {"type": "compose", "parts": [tc_library[n] for n in names]}

            resolved_changes.append((week, perturb_cfg))

        # Sort by week so the loop can apply them in chronological order
        resolved_changes.sort(key=lambda x: x[0])

        simulations.append({
            "name":               str(entry["name"]),
            "label":              str(entry["label"]),
            "has_tariff_changes": bool(resolved_changes),
            "tariff_changes":     resolved_changes,
        })

    return simulations


# ── Perturbation factory ──────────────────────────────────────────────────────

def build_perturbation_fn(perturb_cfg: dict[str, Any]) -> Callable:
    """Build a perturbation callable from a resolved perturbation config dict."""
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
        area_factors = {int(k): float(v) for k, v in perturb_cfg["area_factors"].items()}
        return area_reprice(area_factors)
    if ptype == "compose":
        parts = [build_perturbation_fn(p) for p in perturb_cfg["parts"]]
        return compose(*parts)

    raise ValueError(f"Unknown perturbation type '{ptype}'.")


def build_perturbed_oracle(base_engine, perturb_cfg: dict[str, Any]):
    """Build a PerturbedOracleEngine from an oracle and a perturbation config."""
    from market_model_al.perturbed_oracle import PerturbedOracleEngine
    return PerturbedOracleEngine(base_engine, build_perturbation_fn(perturb_cfg))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_perturbation(params: dict, context: str) -> None:
    if "type" not in params:
        raise ValueError(f"tariff_changes.yaml '{context}': missing 'type'")
    ptype = params["type"]
    if ptype not in _VALID_PERTURB_TYPES:
        raise ValueError(f"tariff_changes.yaml '{context}': unknown type '{ptype}'. "
                         f"Valid: {_VALID_PERTURB_TYPES}")
    if ptype in ("young_driver_surcharge", "high_value_surcharge", "uniform_reprice"):
        if "factor" not in params:
            raise ValueError(f"tariff_changes.yaml '{context}': {ptype} requires 'factor'")
    if ptype == "area_reprice" and "area_factors" not in params:
        raise ValueError(f"tariff_changes.yaml '{context}': area_reprice requires 'area_factors'")
