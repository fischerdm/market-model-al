"""
Active learning query strategies for the competitor-model simulation.

Each strategy selects *which anchor rows* to scrape next week.  The caller
generates ceteris-paribus profiles from the selected anchors and labels them
via the oracle.  Strategies therefore decide *where in feature space* to focus
the scraping effort, not which individual profiles to label.

Strategies
----------
random          — select anchors uniformly at random (baseline)
uncertainty     — anchors where the competitor model is most uncertain
                  (bootstrap variance of predictions at the anchor point)
error_based     — anchors with the highest expected absolute prediction error
                  (proxy model trained on labeled residuals)
shap_divergence — anchors where oracle and competitor SHAP vectors diverge most
                  (L2 norm of per-feature SHAP difference at the anchor)

All strategies accept a NumPy RNG for reproducibility.

Interface
---------
Each function receives:
    anchor_pool   : pd.DataFrame  — candidate anchors to score (real data rows)
    n             : int           — number of anchors to select
    ...model args...
    rng           : np.random.Generator

Returns:
    np.ndarray of integer indices into anchor_pool (length n).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


# ── helpers ───────────────────────────────────────────────────────────────────


def _encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Convert category and object columns to integer codes.

    Bootstrap and proxy models only need variance / error estimates and don't
    require LightGBM's categorical optimisation.  Integer codes avoid
    category-level mismatches when anchor pools cover only a subset of levels.

    Object columns are included because rows originating from the warm start or
    CP profiles may carry string-dtype categoricals that were not cast to the
    pandas Categorical dtype before being appended to labeled_X.
    """
    result = X.copy()
    for col in X.select_dtypes(["category", "object"]).columns:
        result[col] = result[col].astype("category").cat.codes
    return result


def _train_quick_lgb(
    X: pd.DataFrame,
    y: np.ndarray,
    rng: np.random.Generator,
) -> lgb.LGBMRegressor:
    """Train a small, fast LightGBM model (used for bootstrap and proxy models)."""
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=int(rng.integers(0, 2**31)),
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(_encode_categoricals(X), y)
    return model


# ── strategy functions ────────────────────────────────────────────────────────


def random_query(
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n anchors uniformly at random."""
    return rng.choice(len(anchor_pool), size=n, replace=False)


def uncertainty_query(
    competitor,             # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    n_boot: int = 7,
) -> np.ndarray:
    """Select n anchors where the competitor model is most uncertain.

    Falls back to random selection when the labeled set is empty (e.g.
    immediately after a restart).

    Uncertainty is estimated as the standard deviation of predictions across
    bootstrap-resampled models, evaluated directly at each candidate anchor.
    Anchors with high variance indicate regions where the competitor model
    lacks confidence — exactly where additional scraping is most valuable.
    """
    if len(labeled_X) == 0:
        return rng.choice(len(anchor_pool), size=n, replace=False)

    preds = np.zeros((n_boot, len(anchor_pool)))
    for i in range(n_boot):
        boot_idx = rng.choice(len(labeled_X), size=len(labeled_X), replace=True)
        boot_model = _train_quick_lgb(
            labeled_X.iloc[boot_idx], labeled_y[boot_idx], rng
        )
        preds[i] = boot_model.predict(_encode_categoricals(anchor_pool))

    scores = preds.std(axis=0)
    return np.argsort(-scores)[:n]


def error_based_query(
    competitor,             # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n anchors with the highest expected relative prediction error.

    A proxy LightGBM model is trained on (labeled_X, |residuals| / label) and
    used to predict the expected relative error magnitude at each candidate
    anchor.  Using relative rather than absolute residuals avoids systematically
    over-sampling high-premium policies whose absolute errors are large purely
    due to their premium level.

    Falls back to random selection when the labeled set is empty (e.g.
    immediately after a restart).
    """
    if len(labeled_X) == 0:
        return rng.choice(len(anchor_pool), size=n, replace=False)

    abs_residuals = np.abs(labeled_y - competitor.predict(labeled_X))
    residuals = abs_residuals / (np.abs(labeled_y) + 1e-6)
    proxy = _train_quick_lgb(labeled_X, residuals, rng)
    scores = proxy.predict(_encode_categoricals(anchor_pool))
    return np.argsort(-scores)[:n]



def segment_adaptive_query(
    competitor,             # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n anchors with budget allocated proportionally to per-segment RMSE.

    Per-segment RMSE is estimated on the **labeled set** (where ground truth is
    available).  Each candidate anchor is scored as the global RMSE (baseline)
    plus the RMSE of every named segment it belongs to.  This means:

    - Anchors in high-error segments are prioritised over low-error ones.
    - Anchors outside all named segments still compete via the global baseline,
      preventing mainstream-population starvation.
    - Anchors that fall into multiple high-error segments (e.g. a young driver
      with a high-power car) receive extra priority.

    As segment gaps close over time the RMSE differentials shrink and the
    strategy naturally converges toward uniform random sampling.

    Falls back to random selection when the labeled set is empty (e.g.
    immediately after a restart).
    """
    if len(labeled_X) == 0:
        return rng.choice(len(anchor_pool), size=n, replace=False)

    from market_model_al.segments import SEGMENTS, segment_rmse

    # Estimate per-segment RMSE on the labeled set
    labeled_preds  = competitor.predict(labeled_X)
    residuals_sq   = (labeled_y - labeled_preds) ** 2
    global_rmse    = float(np.sqrt(residuals_sq.mean()))
    global_mean    = float(np.abs(labeled_y).mean()) + 1e-6
    global_rel     = global_rmse / global_mean
    seg_rmses      = segment_rmse(labeled_X, labeled_y, labeled_preds)

    # Score each candidate anchor: baseline + cumulative per-segment relative RMSE.
    # Using relative RMSE (RMSE / mean_premium) rather than absolute RMSE avoids
    # systematically over-sampling segments with inherently high premium levels.
    scores = np.full(len(anchor_pool), global_rel, dtype=float)
    for seg in SEGMENTS:
        rmse_val = seg_rmses.get(seg.key, float("nan"))
        if np.isnan(rmse_val):
            continue
        seg_mask   = seg.filter_fn(labeled_X).values
        seg_mean   = float(np.abs(labeled_y[seg_mask]).mean()) + 1e-6
        rel_val    = rmse_val / seg_mean
        mask = seg.filter_fn(anchor_pool).values
        scores[mask] += rel_val

    # Small jitter so anchors within the same score bucket are selected randomly
    # rather than by their position in the DataFrame.
    scores += rng.uniform(0, 1e-6, size=len(scores))

    return np.argsort(-scores)[:n]


def disruption_query(
    competitor,             # CompetitorModel
    labeled_X: pd.DataFrame,
    labeled_y: np.ndarray,
    anchor_pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    prev_seg_rmses: dict[str, float] | None = None,
    disruption_threshold: float = 0.15,
) -> np.ndarray:
    """Select n anchors by detecting and responding to disrupted segments.

    Each week, per-segment RMSE is compared to the previous week.  A segment is
    flagged as disrupted when its relative RMSE *increase* exceeds
    disruption_threshold (default 15 %).  When disruption is detected, the
    entire weekly budget is concentrated on random sampling within the union of
    all flagged segments.  When no disruption is detected, the strategy falls
    back to global random sampling.

    This uses the *derivative* of RMSE rather than its absolute level, making it
    robust to segments that are permanently harder (e.g. young drivers always
    have a higher absolute RMSE).  It fires exactly when needed and stops once
    the gap closes — a natural on/off signal rather than a permanent bias.

    Parameters
    ----------
    prev_seg_rmses : dict[str, float] | None
        Per-segment RMSE from the previous week (keys = segment.key).
        None on the first week — falls back to global random.
    disruption_threshold : float
        Relative RMSE increase that triggers disruption mode.
        E.g. 0.15 means a ≥15 % week-on-week RMSE increase in a segment.

    Falls back to random selection when the labeled set or prev_seg_rmses is
    unavailable (first week, or immediately after a restart).
    """
    from market_model_al.segments import SEGMENTS, segment_rmse

    if len(labeled_X) == 0 or prev_seg_rmses is None:
        return rng.choice(len(anchor_pool), size=n, replace=False)

    labeled_preds = competitor.predict(labeled_X)
    curr_seg_rmses = segment_rmse(labeled_X, labeled_y, labeled_preds)

    # Detect disrupted segments: relative RMSE increase > threshold
    disrupted_masks = []
    for seg in SEGMENTS:
        prev = prev_seg_rmses.get(seg.key, float("nan"))
        curr = curr_seg_rmses.get(seg.key, float("nan"))
        if np.isnan(prev) or np.isnan(curr) or prev < 1e-6:
            continue
        if (curr - prev) / prev > disruption_threshold:
            disrupted_masks.append(seg.filter_fn(anchor_pool).values)

    if not disrupted_masks:
        # No disruption detected — global random
        return rng.choice(len(anchor_pool), size=n, replace=False)

    # Concentrate on the union of all disrupted segments
    disrupted = np.zeros(len(anchor_pool), dtype=bool)
    for mask in disrupted_masks:
        disrupted |= mask

    disrupted_idx = np.where(disrupted)[0]
    if len(disrupted_idx) < n:
        # Disrupted segment too small — sample with replacement from it
        return rng.choice(disrupted_idx, size=n, replace=True)

    return rng.choice(disrupted_idx, size=n, replace=False)


# ── registry ──────────────────────────────────────────────────────────────────

STRATEGIES = [
    "random", "random_market",
    "uncertainty", "error_based", "segment_adaptive", "disruption",
    # Gaussian-perturbation variants: same selection logic, joint profile generator.
    "random_gauss", "uncertainty_gauss", "error_based_gauss",
    "segment_adaptive_gauss", "disruption_gauss",
]
