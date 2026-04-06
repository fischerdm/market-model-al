"""
Active learning simulation loop.

ALSimulation coordinates the full experiment:
  1. Draw a warm-start labeled set from the pre-labeled pool.
  2. Train the competitor model on the labeled set.
  3. Record convergence metrics (MSE, RMSE, SHAP cosine similarity to oracle).
  4. Use the chosen query strategy to select the next batch from the unlabeled pool.
  5. Reveal oracle labels (already computed) and move to labeled.
  6. Repeat for n_iterations.

All oracle labels are pre-computed (pool_y), so the loop never calls the oracle
engine at runtime — it only "pays" by incrementing the labeled budget counter.

Usage
-----
    from market_model_al.al_loop import ALSimulation
    from market_model_al.oracle_engine import OraclePricingEngine

    oracle = OraclePricingEngine("outputs/models/oracle.pkl")
    sim = ALSimulation(oracle, pool_X, pool_y, seed=42)
    results = sim.run("uncertainty", warm_start_n=10_000, batch_size=500, n_iterations=40)
    # results is a pd.DataFrame with columns:
    #   iteration, n_labeled, rmse, rel_rmse, shap_cosine_similarity
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error

from market_model_al.competitor_model import CompetitorModel
from market_model_al.strategies import (
    random_query,
    uncertainty_query,
    error_based_query,
    shap_divergence_query,
    STRATEGIES,
)


# Fixed holdout size for metric evaluation (drawn once from the pool, not from
# the unlabeled set — it's a separate evaluation partition).
_HOLDOUT_N = 2_000


class ALSimulation:
    """Run active learning experiments over a pre-labeled profile pool.

    Parameters
    ----------
    oracle_engine : OraclePricingEngine
        Needed to build the oracle SHAP explainer for SHAP-divergence metrics
        and for the shap_divergence query strategy.
    pool_X : pd.DataFrame
        All candidate profiles (engineered features, no target).
    pool_y : np.ndarray
        Oracle-predicted premium for every row in pool_X (1-D, aligned).
    seed : int
        Master random seed; all sub-operations derive from it.
    competitor_params : dict, optional
        Override default LightGBM hyperparameters for the competitor model.
    """

    def __init__(
        self,
        oracle_engine,
        pool_X: pd.DataFrame,
        pool_y: np.ndarray,
        seed: int = 42,
        competitor_params: dict | None = None,
    ) -> None:
        self._oracle_engine = oracle_engine
        self._pool_X = pool_X.reset_index(drop=True)
        self._pool_y = np.asarray(pool_y, dtype=float)
        self._rng = np.random.default_rng(seed)
        self._competitor_params = competitor_params

        # Carve out a fixed holdout partition for metric evaluation.
        # This is drawn *before* any strategy runs so it stays constant.
        holdout_idx = self._rng.choice(
            len(self._pool_X), size=_HOLDOUT_N, replace=False
        )
        self._holdout_mask = np.zeros(len(self._pool_X), dtype=bool)
        self._holdout_mask[holdout_idx] = True
        self._holdout_X = self._pool_X.iloc[holdout_idx].copy()
        self._holdout_y = self._pool_y[holdout_idx].copy()

        # Pre-compute oracle SHAP on holdout (fixed reference — computed once).
        print("Pre-computing oracle SHAP on holdout set…", flush=True)
        oracle_explainer = shap.TreeExplainer(oracle_engine._oracle)
        self._oracle_shap = oracle_explainer.shap_values(self._holdout_X)
        self._oracle_explainer = oracle_explainer
        print(f"  Oracle SHAP shape: {self._oracle_shap.shape}\n", flush=True)

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        strategy: str,
        warm_start_n: int = 10_000,
        batch_size: int = 500,
        n_iterations: int = 40,
    ) -> pd.DataFrame:
        """Run a full AL experiment and return a DataFrame of per-iteration metrics.

        Parameters
        ----------
        strategy : str
            One of 'random', 'uncertainty', 'error_based', 'shap_divergence'.
        warm_start_n : int
            Number of labeled profiles to seed the experiment with.
        batch_size : int
            Profiles added to the labeled set per iteration.
        n_iterations : int
            Number of AL rounds (not counting the warm-start evaluation).

        Returns
        -------
        pd.DataFrame
            Columns: strategy, iteration, n_labeled, rmse, rel_rmse,
                     shap_cosine_similarity, elapsed_s
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'.  Choose from {STRATEGIES}.")

        rng = np.random.default_rng(int(self._rng.integers(0, 2**31)))

        # Pool rows available for AL (excludes the fixed holdout).
        pool_avail = np.where(~self._holdout_mask)[0]
        rng.shuffle(pool_avail)

        labeled_local = pool_avail[:warm_start_n]
        unlabeled_local = pool_avail[warm_start_n:]

        records = []
        competitor = CompetitorModel(params=self._competitor_params)

        print(f"Strategy: {strategy}")
        print(f"  warm_start={warm_start_n:,}  batch={batch_size}  iters={n_iterations}\n")

        for it in range(n_iterations + 1):  # +1: evaluate warm-start before first query
            t0 = time.perf_counter()

            # ── 1. train competitor ───────────────────────────────────────────
            X_lab = self._pool_X.iloc[labeled_local]
            y_lab = self._pool_y[labeled_local]
            competitor.fit(X_lab, y_lab)

            # ── 2. evaluate on holdout ────────────────────────────────────────
            preds = competitor.predict(self._holdout_X)
            rmse = float(np.sqrt(mean_squared_error(self._holdout_y, preds)))
            rel_rmse = rmse / float(self._holdout_y.mean())
            shap_sim = self._shap_similarity(competitor)

            elapsed = time.perf_counter() - t0
            records.append(
                dict(
                    strategy=strategy,
                    iteration=it,
                    n_labeled=len(labeled_local),
                    rmse=rmse,
                    rel_rmse=rel_rmse,
                    shap_cosine_similarity=shap_sim,
                    elapsed_s=elapsed,
                )
            )
            print(
                f"  iter {it:3d} | labeled={len(labeled_local):6,} | "
                f"RMSE={rmse:7.2f} | rel={rel_rmse:.4f} | "
                f"SHAP-sim={shap_sim:.4f} | {elapsed:.1f}s",
                flush=True,
            )

            if it == n_iterations:
                break  # final eval done, no need to query

            # ── 3. select next batch via strategy ─────────────────────────────
            unlabeled_X = self._pool_X.iloc[unlabeled_local]

            if strategy == "random":
                chosen_local_idx = random_query(unlabeled_X, batch_size, rng)

            elif strategy == "uncertainty":
                chosen_local_idx = uncertainty_query(
                    competitor, X_lab, y_lab, unlabeled_X, batch_size, rng
                )

            elif strategy == "error_based":
                chosen_local_idx = error_based_query(
                    competitor, X_lab, y_lab, unlabeled_X, batch_size, rng
                )

            elif strategy == "shap_divergence":
                chosen_local_idx = shap_divergence_query(
                    self._oracle_explainer, competitor, unlabeled_X, batch_size, rng
                )

            # Move selected from unlabeled → labeled
            chosen_pool_idx = unlabeled_local[chosen_local_idx]
            labeled_local = np.concatenate([labeled_local, chosen_pool_idx])
            unlabeled_local = np.delete(unlabeled_local, chosen_local_idx)

        return pd.DataFrame(records)

    # ── internal ──────────────────────────────────────────────────────────────

    def _shap_similarity(self, competitor: CompetitorModel) -> float:
        """Mean cosine similarity between oracle and competitor SHAP vectors on holdout."""
        comp_shap = competitor.shap_values(self._holdout_X)   # (n, F)

        # Row-wise cosine similarity, averaged across the holdout.
        dot = (self._oracle_shap * comp_shap).sum(axis=1)
        norm_o = np.linalg.norm(self._oracle_shap, axis=1) + 1e-12
        norm_c = np.linalg.norm(comp_shap, axis=1) + 1e-12
        cos_sim = dot / (norm_o * norm_c)
        return float(cos_sim.mean())
