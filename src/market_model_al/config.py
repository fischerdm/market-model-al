"""
Configuration loader for the AL simulation.

Reads simulation.yaml and tariff_changes.yaml, validates them, and provides
factory helpers that build perturbation functions and perturbed oracle engines.

Usage
-----
    from market_model_al.config import load_simulation_cfg, load_tariff_changes_cfg
    from market_model_al.config import resolve_scenarios, build_perturbed_oracle

    sim_cfg    = load_simulation_cfg("config/simulation.yaml")
    tc_library = load_tariff_changes_cfg("config/tariff_changes.yaml")
    scenarios  = resolve_scenarios(sim_cfg, tc_library)

    # sim_cfg keys:
    #   n_weeks, weekly_budget, candidate_multiplier, seed,
    #   strategies, metrics, compute_shap_similarity,
    #   perturbation_schedule, restart_strategies
    #
    # tc_library: dict of name → perturbation-param dict
    #
    # scenarios: list of dicts with keys:
    #   name, label, week, perturbation_names, perturb_cfg, restart_strategies

    perturbed = build_perturbed_oracle(oracle_engine, scenario["perturb_cfg"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml


# ── Simulation config ─────────────────────────────────────────────────────────

_VALID_METRICS = {"rmse", "rel_rmse", "shap_cosine_similarity"}
_REQUIRED_METRICS = {"rmse"}


def load_simulation_cfg(path: str | Path) -> dict[str, Any]:
    """Load and validate simulation.yaml.

    Returns a dict with keys:
        n_weeks, weekly_budget, candidate_multiplier, seed,
        strategies, metrics, compute_shap_similarity,
        perturbation_schedule, restart_strategies
    """
    raw = _load_yaml(path)

    cfg: dict[str, Any] = {}
    cfg["n_weeks"]              = int(raw["n_weeks"])
    cfg["weekly_budget"]        = int(raw["weekly_budget"])
    cfg["seed"]                 = int(raw["seed"])
    cfg["strategies"]           = list(raw["strategies"])
    cfg["restart_strategies"]   = list(raw.get("restart_strategies") or [])
    cfg["perturbation_schedule"] = list(raw.get("perturbation_schedule") or [])

    advanced = raw.get("advanced") or {}
    cfg["candidate_multiplier"] = int(advanced.get("candidate_multiplier", 10))

    metrics = set(raw.get("metrics", list(_VALID_METRICS)))
    unknown = metrics - _VALID_METRICS
    if unknown:
        raise ValueError(f"simulation.yaml: unknown metrics {unknown}. "
                         f"Valid: {_VALID_METRICS}")
    metrics |= _REQUIRED_METRICS
    cfg["metrics"] = metrics
    cfg["compute_shap_similarity"] = "shap_cosine_similarity" in metrics

    for i, entry in enumerate(cfg["perturbation_schedule"]):
        if "week" not in entry or "perturbations" not in entry:
            raise ValueError(
                f"simulation.yaml perturbation_schedule[{i}]: "
                "each entry must have 'week' and 'perturbations'"
            )

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


# ── Scenario resolution ───────────────────────────────────────────────────────

def resolve_scenarios(
    sim_cfg: dict[str, Any],
    tc_library: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine simulation schedule with the perturbation library.

    Returns a list of scenario dicts with keys:
        name               : slug used in results parquet and figure filenames
        label              : human-readable description for plots/dashboard
        week               : AL week the oracle switches
        perturbation_names : list of names from tariff_changes.yaml
        perturb_cfg        : resolved perturbation config (compose if multiple)
        restart_strategies : from sim_cfg (global)
    """
    scenarios = []
    name_counts: dict[str, int] = {}  # deduplicate names when the same perturbs appear twice

    for entry in sim_cfg["perturbation_schedule"]:
        week = int(entry["week"])
        names: list[str] = list(entry["perturbations"])

        # Validate all names exist in library
        for n in names:
            if n not in tc_library:
                raise ValueError(
                    f"simulation.yaml references unknown perturbation '{n}'. "
                    f"Defined in tariff_changes.yaml: {list(tc_library)}"
                )

        base_name = "_".join(names)
        count = name_counts.get(base_name, 0)
        name_counts[base_name] = count + 1
        unique_name = f"w{week}_{base_name}" + (f"_{count}" if count else "")

        if len(names) == 1:
            perturb_cfg = tc_library[names[0]]
            label = _make_label(names[0], tc_library[names[0]])
        else:
            # Compose: wrap in a synthetic compose config
            parts = [tc_library[n] for n in names]
            perturb_cfg = {"type": "compose", "parts": parts}
            label = " + ".join(_make_label(n, tc_library[n]) for n in names)

        scenarios.append({
            "name":               unique_name,
            "label":              label,
            "week":               week,
            "perturbation_names": names,
            "perturb_cfg":        perturb_cfg,
            "restart_strategies": sim_cfg["restart_strategies"],
        })

    return scenarios


def _make_label(name: str, params: dict) -> str:
    """Human-readable label derived from the perturbation definition."""
    ptype = params["type"]
    factor = params.get("factor")
    sign = "+" if factor and factor >= 0 else ""
    pct = f"{sign}{factor * 100:.0f} %" if factor is not None else ""

    if ptype == "young_driver_surcharge":
        thr = params.get("age_threshold", 30)
        return f"Young-driver surcharge {pct} (age < {thr:.0f})"
    if ptype == "high_value_surcharge":
        thr = params.get("value_threshold", 50_000)
        return f"High-value vehicle surcharge {pct} (> €{thr:,.0f})"
    if ptype == "uniform_reprice":
        return f"Uniform reprice {pct}"
    if ptype == "area_reprice":
        return f"Area repricing ({name})"
    return name


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
