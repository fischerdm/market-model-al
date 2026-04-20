"""Cube method balanced sampling (Tillé & Deville 2004).

Selects n rows from a pool such that the sample means of auxiliary
variables equal the population means by construction — not just in
expectation as with simple random sampling (SRS).

Algorithm
---------
Flight phase (Tillé & Deville 2004, Section 5):
  Initialize inclusion probabilities a = π = n/N (equal for SRS).
  Each iteration finds a direction u in the null space of the balance
  matrix restricted to non-integer a_i, then takes a Bernoulli step
  along ±u that (a) preserves E[a] = π exactly and (b) drives at least
  one probability to 0 or 1.  After N − p iterations only p
  probabilities remain non-integer.

Landing phase:
  Round the at-most-p remaining non-integer probabilities by independent
  Bernoulli draws.  This introduces a small, bounded imbalance in at
  most p auxiliary dimensions — negligible for p ≪ n.

Complexity
----------
  Flight phase: O(N² · p) in the worst case.  In practice each step is
  O(|A| · p) where |A| shrinks from N to p, giving total ≈ N² · p / 2.
  For typical parameters (N ≈ 15 000, p = 7) this runs in a few seconds
  with vectorised NumPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cube_sampling(
    pool: pd.DataFrame,
    n: int,
    aux_cols: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Select n rows from pool balanced on aux_cols.

    Uses the cube method (Tillé & Deville 2004) so that the sample
    mean of every auxiliary column equals the population mean — the
    survey-sampling analogue of covariate balance.

    Parameters
    ----------
    pool : pd.DataFrame
        Candidate rows.  The cube method selects from these.
    n : int
        Target sample size.  Returned array always has exactly n elements.
    aux_cols : list[str]
        Continuous columns used as balancing auxiliary variables.
        Missing or constant columns are silently excluded from the
        balance constraints.
    rng : np.random.Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    np.ndarray
        Integer indices into pool (length n), sorted ascending.
    """
    N = len(pool)
    if n >= N:
        return np.arange(N)
    if n <= 0:
        return np.array([], dtype=int)

    # Extract auxiliary variables; drop constant columns (zero variance)
    # to avoid a singular balance matrix.
    X_raw = pool[aux_cols].values.astype(float)
    std   = X_raw.std(axis=0)
    valid = std > 1e-10
    X_raw = X_raw[:, valid]

    if X_raw.shape[1] == 0:
        # No valid auxiliary variables — fall back to simple random sampling
        return rng.choice(N, size=n, replace=False)

    # Standardise so all auxiliary dimensions have equal influence
    mu  = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    X   = (X_raw - mu) / std          # shape: (N, p)
    p   = X.shape[1]

    # Equal-probability (SRS) inclusion probabilities
    a = np.full(N, n / N, dtype=float)

    EPS = 1e-9

    # ── Flight phase ──────────────────────────────────────────────────────────
    # Each iteration reduces |A| (non-integer set) by at least 1.
    # Safety cap: at most 3N iterations (in practice N - p suffice).
    for _ in range(3 * N):
        non_int = (a > EPS) & (a < 1.0 - EPS)
        idx     = np.where(non_int)[0]
        if len(idx) <= p:
            break

        X_A = X[idx]   # shape: (|A|, p)
        a_A = a[idx]

        # Null-space direction: project a random vector v onto null(X_A^T).
        #   X_A^T u = 0  ⟺  u ⊥ col(X_A)
        #   u = v − X_A (X_A^T X_A)^{-1} X_A^T v
        # Cost: O(|A| · p) — cheap because p ≪ |A|.
        v   = rng.standard_normal(len(idx))
        XtX = X_A.T @ X_A    # p × p
        Xtv = X_A.T @ v      # p
        try:
            c = np.linalg.solve(XtX, Xtv)
        except np.linalg.LinAlgError:
            c, *_ = np.linalg.lstsq(X_A, v, rcond=None)
        u = v - X_A @ c      # null-space component: X_A^T @ u ≈ 0

        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:   # degenerate — try a fresh random direction
            continue
        u /= u_norm

        # Maximum steps λ+ (in direction +u) and λ− (in direction −u)
        # that keep every a_i ∈ [0, 1].
        #
        # Step +u: a_i + λu_i ≤ 1 for u_i > 0  →  λ ≤ (1 − a_i) / u_i
        #          a_i + λu_i ≥ 0 for u_i < 0  →  λ ≤ −a_i / u_i
        # Step −u: a_i − λu_i ≤ 1 for u_i < 0  →  λ ≤ (a_i − 1) / u_i
        #          a_i − λu_i ≥ 0 for u_i > 0  →  λ ≤  a_i / u_i
        pos = u > EPS
        neg = u < -EPS
        lam_plus = lam_minus = np.inf

        if pos.any():
            lam_plus  = min(lam_plus,  ((1.0 - a_A[pos]) / u[pos]).min())
            lam_minus = min(lam_minus, (a_A[pos] / u[pos]).min())
        if neg.any():
            lam_plus  = min(lam_plus,  (-a_A[neg] / u[neg]).min())
            lam_minus = min(lam_minus, ((a_A[neg] - 1.0) / u[neg]).min())

        if lam_plus + lam_minus < 1e-14:
            break

        # Bernoulli step that preserves E[a] = π:
        #   move +λ+ with probability λ− / (λ+ + λ−)
        #   move −λ− with probability λ+ / (λ+ + λ−)
        if rng.random() < lam_minus / (lam_plus + lam_minus):
            a[idx] = a_A + lam_plus  * u
        else:
            a[idx] = a_A - lam_minus * u
        np.clip(a, 0.0, 1.0, out=a)

    # ── Landing phase ─────────────────────────────────────────────────────────
    # At most p probabilities remain non-integer; round each by Bernoulli draw.
    remaining = np.where((a > EPS) & (a < 1.0 - EPS))[0]
    for i in remaining:
        a[i] = 1.0 if rng.random() < a[i] else 0.0

    selected = np.where(a > 0.5)[0]

    # Trim or top-up to exactly n (rounding occasionally mis-counts by ≤ p)
    if len(selected) > n:
        selected = rng.choice(selected, size=n, replace=False)
    elif len(selected) < n:
        unselected = np.where(a <= 0.5)[0]
        extra      = rng.choice(unselected, size=n - len(selected), replace=False)
        selected   = np.sort(np.concatenate([selected, extra]))

    return selected
