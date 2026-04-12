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
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error

from market_model_al.competitor_model import CompetitorModel
from market_model_al.profile_generator import generate_ceteris_paribus, PROFILES_PER_ANCHOR
from market_model_al.segments import segment_rmse, segment_rel_rmse
from market_model_al.strategies import (
    STRATEGIES,
    random_query,
    uncertainty_query,
    error_based_query,
    segment_adaptive_query,
    disruption_query,
)

# Default random_market hyperparameters (overridable via run())
_RANDOM_MARKET_N_CP_ANCHORS = 50
_RANDOM_MARKET_CP_RATIO     = 0.10


_HOLDOUT_N = 5_000   # fixed holdout size, carved from real_X at construction time


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
        compute_shap_similarity: bool = True,
    ) -> None:
        self._oracle = oracle_engine
        self._master_rng = np.random.default_rng(seed)
        self._competitor_params = competitor_params
        self._compute_shap_similarity = compute_shap_similarity

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

        # Oracle SHAP explainer — only pre-computed when SHAP similarity is needed,
        # as it adds ~10 s at startup and is not observable in real-world deployment.
        if self._compute_shap_similarity:
            print("Pre-computing oracle SHAP on holdout set…", flush=True)
            self._oracle_explainer = shap.TreeExplainer(oracle_engine._oracle)
            self._oracle_shap = self._oracle_explainer.shap_values(self._holdout_X)
            print(f"  Oracle SHAP shape: {self._oracle_shap.shape}\n", flush=True)
        else:
            self._oracle_explainer = None
            self._oracle_shap = None
            print("SHAP similarity disabled — skipping oracle SHAP precomputation.\n",
                  flush=True)

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
        tariff_changes: list[tuple[int, Any]] | None = None,
        restart_at_tariff_change: bool = False,
        random_market_n_cp_anchors: int = _RANDOM_MARKET_N_CP_ANCHORS,
        market_cp_ratio: float = _RANDOM_MARKET_CP_RATIO,
    ) -> pd.DataFrame:
        """Run a full AL experiment and return per-week metrics.

        Parameters
        ----------
        strategy : str
            One of: 'random', 'uncertainty', 'error_based',
                    'segment_adaptive', 'disruption'.
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
        tariff_changes : list of (week, PerturbedOracleEngine), optional
            Sequence of oracle switch events applied within this single run,
            sorted by week.  At each scheduled week the oracle switches and
            holdout labels are re-computed with the new oracle, so RMSE
            measures recovery of the *currently active* tariff.
            Multiple events produce a multi-shock timeline within one run.
        restart_at_tariff_change : bool
            If True, the entire labeled set is discarded after each tariff-change
            week's evaluation.  The model re-learns from scratch using only
            profiles labeled by the new oracle.  Applied at *every* change week,
            not just the first.
        random_market_n_cp_anchors : int
            (random_market only) Number of random anchors from which CP profiles
            are generated each week.  More anchors → broader CP coverage of the
            aggregator space, but higher generation cost.
        market_cp_ratio : float
            Fraction of weekly_budget drawn from the CP pool (random_market
            strategy only); the remainder is drawn from the real anchor pool.
            Represents the degree to which the competitor's portfolio
            under-covers the aggregator space.  Should match the value used to
            build the warm start.  E.g. 0.10 → 10 % CP profiles, 90 % real rows.

        Returns
        -------
        pd.DataFrame
            Columns: strategy, week, n_labeled, rmse, rel_rmse,
                     shap_cosine_similarity, post_tariff_change,
                     tariff_change_applied, elapsed_s, rmse_<segment>…
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'.  Choose from {STRATEGIES}.")

        # Sort change events by week and build a fast lookup set
        tc_schedule: list[tuple[int, Any]] = sorted(tariff_changes or [], key=lambda x: x[0])
        tc_weeks = {w for w, _ in tc_schedule}

        n_anchors    = max(1, weekly_budget // PROFILES_PER_ANCHOR)
        n_candidates = n_anchors * candidate_multiplier

        rng = np.random.default_rng(int(self._master_rng.integers(0, 2**31)))

        # Working labeled set — starts from warm start
        labeled_X = warm_start_X.copy().reset_index(drop=True)
        labeled_y = np.asarray(warm_start_y, dtype=float).copy()

        # Current oracle — updated at each scheduled tariff change
        current_oracle = self._oracle
        holdout_y      = self._holdout_y_base.copy()
        post_change    = False          # True after the first change

        records = []
        competitor = CompetitorModel(params=self._competitor_params)
        prev_seg_rmses: dict[str, float] | None = None   # for disruption_query

        print(f"Strategy : {strategy}")
        if strategy == "random_market":
            n_cp = int(weekly_budget * market_cp_ratio)
            n_real = weekly_budget - n_cp
            print(f"  warm_start={len(labeled_X):,}  weekly_budget={weekly_budget:,}"
                  f"  n_cp_anchors={random_market_n_cp_anchors}"
                  f"  cp_ratio={market_cp_ratio}  weeks={n_weeks}")
            print(f"  per week: {n_cp} from CP pool + {n_real} from real pool")
        else:
            print(f"  warm_start={len(labeled_X):,}  weekly_budget={weekly_budget:,}"
                  f"  n_anchors/week={n_anchors}  candidates/week={n_candidates}  weeks={n_weeks}")
        if tc_schedule:
            changes_str = ", ".join(f"week {w}" for w, _ in tc_schedule)
            print(f"  tariff changes at: {changes_str}"
                  + (" (with restart)" if restart_at_tariff_change else ""))
        print()

        # Week 0 = warm-start evaluation (no queries yet)
        for week in range(n_weeks + 1):
            t0 = time.perf_counter()

            # ── Apply any tariff change(s) scheduled for this week ────────────
            change_applied_this_week = False
            for tc_week, tc_oracle in tc_schedule:
                if week == tc_week:
                    current_oracle = tc_oracle
                    holdout_y      = tc_oracle.query(self._holdout_X)
                    post_change    = True
                    change_applied_this_week = True
                    print(f"  [week {week}] Tariff change injected — holdout labels updated.",
                          flush=True)

            # ── Train competitor on current labeled set ────────────────────────
            competitor.fit(labeled_X, labeled_y)

            # ── Evaluate on holdout ───────────────────────────────────────────
            preds    = competitor.predict(self._holdout_X)
            rmse     = float(np.sqrt(mean_squared_error(holdout_y, preds)))
            rel_rmse = rmse / float(holdout_y.mean())
            shap_sim = (self._shap_similarity(competitor)
                        if self._compute_shap_similarity else float("nan"))

            seg_rmse     = segment_rmse(self._holdout_X, holdout_y, preds)
            seg_rel_rmse = segment_rel_rmse(self._holdout_X, holdout_y, preds)

            elapsed = time.perf_counter() - t0
            records.append(dict(
                strategy=strategy,
                week=week,
                n_labeled=len(labeled_X),
                rmse=rmse,
                rel_rmse=rel_rmse,
                shap_cosine_similarity=shap_sim,
                post_tariff_change=post_change,
                tariff_change_applied=change_applied_this_week,
                elapsed_s=elapsed,
                **{f"rmse_{k}": v for k, v in seg_rmse.items()},
                **{f"rel_rmse_{k}": v for k, v in seg_rel_rmse.items()},
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

            # ── Restart after evaluation at every tariff-change week ──────────
            if restart_at_tariff_change and change_applied_this_week:
                labeled_X      = labeled_X.iloc[:0].copy()
                labeled_y      = np.array([], dtype=float)
                prev_seg_rmses = None   # reset so disruption fires cleanly next week
                print(f"  [week {week}] Restart — labeled set cleared, "
                      "re-learning from new oracle only.", flush=True)

            if strategy == "random_market":
                # ── random_market: stratified sampling from market space ──────
                # Build CP pool from n_cp_anchors random anchors, then sample
                # cp_ratio of weekly_budget from it and the remainder from the
                # real anchor pool.  Mirrors aggregator traffic: mostly real
                # portfolio rows, topped up with synthetic profiles that cover
                # segments the competitor doesn't write.
                n_cp_sample   = max(1, int(weekly_budget * market_cp_ratio))
                n_real_sample = weekly_budget - n_cp_sample

                cp_anchor_idx = rng.choice(
                    self._anchor_pool_idx,
                    size=random_market_n_cp_anchors,
                    replace=False,
                )
                cp_profiles = generate_ceteris_paribus(
                    self._real_X.iloc[cp_anchor_idx].reset_index(drop=True),
                    validate=True,
                )

                real_pool = self._real_X.iloc[self._anchor_pool_idx].reset_index(drop=True)

                # Draw from CP pool (replace=True if pool smaller than budget slice)
                if len(cp_profiles) > 0:
                    cp_sample_idx = rng.choice(
                        len(cp_profiles),
                        size=min(n_cp_sample, len(cp_profiles)),
                        replace=n_cp_sample > len(cp_profiles),
                    )
                    cp_sample = cp_profiles.iloc[cp_sample_idx]
                else:
                    cp_sample = cp_profiles  # empty

                real_sample_idx = rng.choice(len(real_pool), size=n_real_sample, replace=False)
                real_sample = real_pool.iloc[real_sample_idx]

                profiles = pd.concat([cp_sample, real_sample], ignore_index=True)
                if len(profiles) == 0:
                    continue

            else:
                # ── CP-anchor strategies ──────────────────────────────────────
                cand_idx = rng.choice(self._anchor_pool_idx, size=n_candidates, replace=False)
                candidate_anchors = self._real_X.iloc[cand_idx].reset_index(drop=True)

                chosen_local = self._apply_strategy(
                    strategy, competitor, labeled_X, labeled_y,
                    candidate_anchors, n_anchors, rng,
                    prev_seg_rmses=prev_seg_rmses,
                )
                selected_anchors = candidate_anchors.iloc[chosen_local]

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
