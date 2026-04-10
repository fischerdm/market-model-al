"""
Active learning simulation loop — anchor-based, weekly scraping budget.

The simulation models the following real-world workflow:

  1. Warm start: the competitor model is seeded with a mix of
       (a) randomly sampled real policy rows  — organic quotes from comparis
       (b) structured ceteris-paribus profiles — initial systematic scraping
     Both are oracle-labeled before the loop starts.

  2. Weekly AL loop: each week the strategy selects which *anchor rows* to
     scrape next, within a fixed profile budget.
       - A candidate pool of anchor rows is drawn from the real dataset.
       - The query strategy selects the best n_anchors from that pool.
       - All ceteris-paribus profiles from the selected anchors are generated
         and labeled by the (current) oracle.
       - The competitor model is retrained on the growing labeled set.
       - Convergence metrics are recorded.

     Weekly budget:  n_anchors = weekly_budget // PROFILES_PER_ANCHOR
     Candidate pool: candidate_multiplier × n_anchors anchors are scored,
                     the top n_anchors are kept.

  3. Optional tariff change: at tariff_change_week the oracle is replaced by a
     PerturbedOracleEngine.  The holdout labels switch to the new oracle so
     RMSE measures recovery of the new tariff — not the old one.

Usage
-----
    from market_model_al.al_loop import ALSimulation
    from market_model_al.oracle_engine import OraclePricingEngine
    from market_model_al.perturbed_oracle import PerturbedOracleEngine, young_driver_surcharge

    oracle = OraclePricingEngine("outputs/models/oracle.pkl")
    new_oracle = PerturbedOracleEngine(oracle, young_driver_surcharge(0.20))

    sim = ALSimulation(oracle, real_X, seed=42)

    # Scenario A: no tariff change, 5 000 profiles/week
    results_a = sim.run("uncertainty", warm_start_X, warm_start_y,
                        weekly_budget=5_000, n_weeks=52)

    # Scenario B: tariff change at week 26
    results_b = sim.run("uncertainty", warm_start_X, warm_start_y,
                        weekly_budget=5_000, n_weeks=52,
                        tariff_change_week=26, perturbed_oracle=new_oracle)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error

from market_model_al.competitor_model import CompetitorModel
from market_model_al.profile_generator import generate_ceteris_paribus, PROFILES_PER_ANCHOR
from market_model_al.segments import segment_rmse
from market_model_al.strategies import (
    STRATEGIES,
    random_query,
    uncertainty_query,
    error_based_query,
    segment_adaptive_query,
    disruption_query,
)


_HOLDOUT_N = 2_000   # fixed holdout size, carved from real_X at construction time


class ALSimulation:
    """Run active learning experiments over the full real dataset.

    Parameters
    ----------
    oracle_engine : OraclePricingEngine
        The base (unperturbed) oracle.  Also used to build the SHAP explainer
        for convergence tracking and the shap_divergence strategy.
    real_X : pd.DataFrame
        Full engineered feature matrix (no target) from the Spanish dataset.
        Used as (a) anchor pool during the loop and (b) source of holdout rows.
    seed : int
        Master seed; each run() call derives its own sub-RNG from this.
    competitor_params : dict, optional
        Override default LightGBM params for the competitor model.
    """

    def __init__(
        self,
        oracle_engine,
        real_X: pd.DataFrame,
        seed: int = 42,
        competitor_params: dict | None = None,
    ) -> None:
        self._oracle = oracle_engine
        self._master_rng = np.random.default_rng(seed)
        self._competitor_params = competitor_params

        # Drop real rows with data-quality violations so the holdout and anchor
        # pool never contain physically invalid profiles.
        real_X = real_X.reset_index(drop=True)
        valid_mask = oracle_engine.validate(real_X)
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"  Dropped {n_invalid} invalid rows from real_X "
                  f"({n_invalid / len(real_X):.2%} of dataset).")
        self._real_X = real_X[valid_mask].reset_index(drop=True)

        # Carve a fixed holdout from real_X — never used as anchors or warm start.
        holdout_idx = self._master_rng.choice(
            len(self._real_X), size=_HOLDOUT_N, replace=False
        )
        self._holdout_mask = np.zeros(len(self._real_X), dtype=bool)
        self._holdout_mask[holdout_idx] = True
        self._holdout_X = self._real_X.iloc[holdout_idx].copy()

        # Label holdout with base oracle once; re-labelled at tariff change.
        self._holdout_y_base = oracle_engine.query(self._holdout_X)

        # Oracle SHAP explainer — always based on the original LightGBM model,
        # used for convergence tracking and the shap_divergence strategy.
        print("Pre-computing oracle SHAP on holdout set…", flush=True)
        self._oracle_explainer = shap.TreeExplainer(oracle_engine._oracle)
        self._oracle_shap = self._oracle_explainer.shap_values(self._holdout_X)
        print(f"  Oracle SHAP shape: {self._oracle_shap.shape}\n", flush=True)

        # Anchor pool = all valid real rows minus holdout
        self._anchor_pool_idx = np.where(~self._holdout_mask)[0]

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        strategy: str,
        warm_start_X: pd.DataFrame,
        warm_start_y: np.ndarray,
        weekly_budget: int = 5_000,
        n_weeks: int = 52,
        candidate_multiplier: int = 10,
        tariff_change_week: int | None = None,
        perturbed_oracle=None,
        restart_at_tariff_change: bool = False,
    ) -> pd.DataFrame:
        """Run a full AL experiment and return per-week metrics.

        Parameters
        ----------
        strategy : str
            One of: 'random', 'uncertainty', 'error_based', 'shap_divergence'.
        warm_start_X : pd.DataFrame
            Warm-start profiles (real rows + CP profiles combined).
        warm_start_y : np.ndarray
            Oracle labels for warm_start_X (aligned 1-D array).
        weekly_budget : int
            Target number of profiles to scrape per week.
            Converted to anchors as: n_anchors = weekly_budget // PROFILES_PER_ANCHOR.
        n_weeks : int
            Number of AL rounds after the warm start.
        candidate_multiplier : int
            The strategy scores (candidate_multiplier × n_anchors) candidate
            anchors each week and selects the best n_anchors from that pool.
            Higher values give strategies more room to differentiate but
            increase scoring time.  Ignored for 'random'.
        tariff_change_week : int, optional
            If set, the oracle switches to perturbed_oracle at this week.
            Holdout labels are re-computed with the new oracle at that point.
        perturbed_oracle : PerturbedOracleEngine, optional
            Required when tariff_change_week is set.
        restart_at_tariff_change : bool
            If True, the entire labeled set (including warm start) is discarded
            at tariff_change_week after that week's evaluation.  The model
            re-learns exclusively from profiles labeled by the new oracle.
            The RMSE spike at tariff_change_week is still recorded honestly.

        Returns
        -------
        pd.DataFrame
            Columns: strategy, week, n_labeled, n_anchors_selected,
                     profiles_added, rmse, rel_rmse, shap_cosine_similarity,
                     post_tariff_change, elapsed_s
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'.  Choose from {STRATEGIES}.")
        if tariff_change_week is not None and perturbed_oracle is None:
            raise ValueError("perturbed_oracle must be supplied when tariff_change_week is set.")

        n_anchors = max(1, weekly_budget // PROFILES_PER_ANCHOR)
        n_candidates = n_anchors * candidate_multiplier

        rng = np.random.default_rng(int(self._master_rng.integers(0, 2**31)))

        # Working labeled set — starts from warm start
        labeled_X = warm_start_X.copy().reset_index(drop=True)
        labeled_y = np.asarray(warm_start_y, dtype=float).copy()

        # Current oracle (may switch at tariff_change_week)
        current_oracle = self._oracle
        holdout_y = self._holdout_y_base.copy()
        post_change = False

        records = []
        competitor = CompetitorModel(params=self._competitor_params)
        prev_seg_rmses: dict[str, float] | None = None   # for disruption_query

        print(f"Strategy : {strategy}")
        print(f"  warm_start={len(labeled_X):,}  weekly_budget={weekly_budget:,}"
              f"  n_anchors/week={n_anchors}  candidates/week={n_candidates}  weeks={n_weeks}")
        if tariff_change_week is not None:
            print(f"  tariff change at week {tariff_change_week}")
        print()

        # Week 0 = warm-start evaluation (no queries yet)
        for week in range(n_weeks + 1):
            t0 = time.perf_counter()

            # ── Switch oracle at tariff_change_week ───────────────────────────
            if tariff_change_week is not None and week == tariff_change_week and not post_change:
                current_oracle = perturbed_oracle
                holdout_y = perturbed_oracle.query(self._holdout_X)
                post_change = True
                print(f"  [week {week}] Tariff change injected — holdout labels updated.")

            # ── Train competitor on current labeled set ────────────────────────
            competitor.fit(labeled_X, labeled_y)

            # ── Evaluate on holdout ───────────────────────────────────────────
            preds   = competitor.predict(self._holdout_X)
            rmse    = float(np.sqrt(mean_squared_error(holdout_y, preds)))
            rel_rmse = rmse / float(holdout_y.mean())
            shap_sim = self._shap_similarity(competitor)

            seg_rmse = segment_rmse(self._holdout_X, holdout_y, preds)

            elapsed = time.perf_counter() - t0
            records.append(dict(
                strategy=strategy,
                week=week,
                n_labeled=len(labeled_X),
                rmse=rmse,
                rel_rmse=rel_rmse,
                shap_cosine_similarity=shap_sim,
                post_tariff_change=post_change,
                elapsed_s=elapsed,
                **{f"rmse_{k}": v for k, v in seg_rmse.items()},
            ))
            print(
                f"  week {week:3d} | labeled={len(labeled_X):6,} | "
                f"RMSE={rmse:7.2f} | rel={rel_rmse:.4f} | "
                f"SHAP-sim={shap_sim:.4f} | {elapsed:.1f}s",
                flush=True,
            )

            if week == n_weeks:
                break

            # ── Carry segment RMSEs forward for disruption detection ──────────
            prev_seg_rmses = seg_rmse

            # ── Restart: discard all stale labels after tariff-change evaluation ─
            if restart_at_tariff_change and post_change and week == tariff_change_week:
                labeled_X = labeled_X.iloc[:0].copy()
                labeled_y = np.array([], dtype=float)
                prev_seg_rmses = None   # reset state so disruption fires cleanly next week
                print(f"  [week {week}] Restart — labeled set cleared, "
                      "re-learning from new oracle only.", flush=True)

            # ── Sample candidate anchor pool ──────────────────────────────────
            cand_idx = rng.choice(self._anchor_pool_idx, size=n_candidates, replace=False)
            candidate_anchors = self._real_X.iloc[cand_idx].reset_index(drop=True)

            # ── Select best anchors via strategy ──────────────────────────────
            chosen_local = self._apply_strategy(
                strategy, competitor, labeled_X, labeled_y,
                candidate_anchors, n_anchors, rng,
                prev_seg_rmses=prev_seg_rmses,
            )
            selected_anchors = candidate_anchors.iloc[chosen_local]

            # ── Generate all CP profiles from selected anchors ────────────────
            profiles = generate_ceteris_paribus(selected_anchors, validate=True)
            if len(profiles) == 0:
                continue

            # ── Label with current oracle ─────────────────────────────────────
            labels = current_oracle.query(profiles)

            # ── Add to labeled set ────────────────────────────────────────────
            labeled_X = pd.concat([labeled_X, profiles], ignore_index=True)
            labeled_y = np.concatenate([labeled_y, labels])

        return pd.DataFrame(records)

    # ── internal ──────────────────────────────────────────────────────────────

    def _apply_strategy(
        self,
        strategy: str,
        competitor: CompetitorModel,
        labeled_X: pd.DataFrame,
        labeled_y: np.ndarray,
        candidate_anchors: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
        prev_seg_rmses: dict | None = None,
    ) -> np.ndarray:
        if strategy == "random":
            return random_query(candidate_anchors, n, rng)
        elif strategy == "uncertainty":
            return uncertainty_query(competitor, labeled_X, labeled_y, candidate_anchors, n, rng)
        elif strategy == "error_based":
            return error_based_query(competitor, labeled_X, labeled_y, candidate_anchors, n, rng)
        elif strategy == "segment_adaptive":
            return segment_adaptive_query(
                competitor, labeled_X, labeled_y, candidate_anchors, n, rng
            )
        elif strategy == "disruption":
            return disruption_query(
                competitor, labeled_X, labeled_y, candidate_anchors, n, rng,
                prev_seg_rmses=prev_seg_rmses,
            )

    def _shap_similarity(self, competitor: CompetitorModel) -> float:
        """Mean cosine similarity between oracle and competitor SHAP on holdout."""
        comp_shap = competitor.shap_values(self._holdout_X)
        dot   = (self._oracle_shap * comp_shap).sum(axis=1)
        norm_o = np.linalg.norm(self._oracle_shap, axis=1) + 1e-12
        norm_c = np.linalg.norm(comp_shap, axis=1) + 1e-12
        return float((dot / (norm_o * norm_c)).mean())
