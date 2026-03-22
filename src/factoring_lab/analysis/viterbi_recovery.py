"""Viterbi decoding with SVD prior for factor recovery from the carry chain.

The spectral recovery experiment shows that SVD of the carry-constrained
least-squares solution Z* correlates with true factors (|corr| = 0.48 for
base 2, 0.83 for base 10), but naive rounding recovers 0% of factors.

Theorem C proves sequential rounding with UNIFORM prior fails at 2^{-d^2/4}.
But the SVD prior is NOT uniform -- it concentrates mass near the true digits.

This module combines:
1. The carry chain Markov structure (transition matrices from carry_channel.py)
2. The SVD estimate from the spectral recovery least-squares system
3. Viterbi algorithm to find the maximum-likelihood carry sequence
4. Constrained projection (water-filling) to recover z-values at each position

The key insight: the carry chain has states t_k in {0, ..., max_carry_at[k]}.
At each position k, the transition from t_{k-1} to t_k involves choosing
z-values for all (i,j) with i+j=k.  The SVD prior scores each z-tuple via:

    score(z) = exp(-lambda * sum (z_{ij} - z_est_{ij})^2)

where z_est_{ij} = x_est_i * y_est_j from the SVD decomposition.

The transition score for (t_{k-1}, t_k) is computed by projecting the SVD
estimates onto the carry-constraint simplex (water-filling), giving a
tractable approximation to the optimal z-tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

import numpy as np

from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    from_digits,
    to_digits,
)


@dataclass
class ViterbiRecoveryResult:
    """Result of Viterbi factor recovery attempt."""

    n: int
    p_true: int
    q_true: int
    base: int
    d: int
    dx: int
    dy: int

    # Viterbi recovery
    recovered_p: int | None
    recovered_q: int | None
    recovery_success: bool
    viterbi_log_likelihood: float

    # Greedy recovery (baseline)
    greedy_p: int | None
    greedy_q: int | None
    greedy_success: bool

    # Naive rounding recovery (from spectral experiment)
    naive_p: int | None
    naive_q: int | None
    naive_success: bool

    # Correlation metrics
    viterbi_corr_x: float
    viterbi_corr_y: float
    greedy_corr_x: float
    greedy_corr_y: float
    naive_corr_x: float
    naive_corr_y: float
    svd_corr_x: float  # raw SVD correlation (continuous)
    svd_corr_y: float

    # Carry sequence
    recovered_carry_sequence: list[int]
    true_carry_sequence: list[int] | None

    # Lambda parameter used
    lambda_param: float


def _build_carry_system(n: int, base: int, dx: int, dy: int):
    """Build the linear system for carry constraints.

    Returns A, b_vec, z_vars, z_idx, num_z, num_t.
    Identical to spectral_recovery_experiment._build_carry_system.
    """
    c = to_digits(n, base)
    d = len(c)

    z_vars = []
    z_idx = {}
    for i in range(dx):
        for j in range(dy):
            if i + j < d:
                z_idx[(i, j)] = len(z_vars)
                z_vars.append((i, j))

    num_z = len(z_vars)
    num_t = d
    num_vars = num_z + num_t

    A = np.zeros((d, num_vars))
    b_vec = np.zeros(d)

    for k in range(d):
        b_vec[k] = c[k]
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy and (i, j) in z_idx:
                A[k, z_idx[(i, j)]] = 1.0
        if k > 0:
            A[k, num_z + k - 1] = 1.0
        A[k, num_z + k] = -base

    return A, b_vec, z_vars, z_idx, num_z, num_t


def _solve_svd_estimates(
    n: int, base: int, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the least-squares carry system and extract SVD estimates.

    Returns (x_est, y_est, Z_star) where:
    - x_est: continuous estimate of factor x digits (length dx)
    - y_est: continuous estimate of factor y digits (length dy)
    - Z_star: the least-squares Z matrix (dx x dy)
    """
    A, b_vec, z_vars, z_idx, num_z, num_t = _build_carry_system(n, base, dx, dy)

    v_star, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

    Z_star = np.zeros((dx, dy))
    for (i, j), idx in z_idx.items():
        Z_star[i, j] = v_star[idx]

    U, S, Vt = np.linalg.svd(Z_star, full_matrices=False)

    u1 = U[:, 0]
    v1 = Vt[0, :]
    s1 = S[0]

    scale = np.sqrt(s1) if s1 > 0 else 1.0
    x_est = scale * u1
    y_est = scale * v1

    return x_est, y_est, Z_star


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation, handling degenerate cases."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _project_to_simplex_with_bounds(
    z_est: np.ndarray, target_sum: int, max_val: float
) -> np.ndarray:
    """Project z_est onto the set {z : sum(z) = target_sum, 0 <= z_i <= max_val}.

    This is the water-filling / simplex projection with box constraints.
    Minimizes ||z - z_est||^2 subject to sum(z) = target_sum and 0 <= z_i <= max_val.

    Uses iterative clipping: project to the hyperplane sum(z) = target_sum,
    clip to [0, max_val], repeat until convergence.
    """
    n = len(z_est)
    if n == 0:
        return np.array([])

    z = z_est.copy().astype(float)
    max_val = float(max_val)

    for _ in range(100):  # iterate until convergence
        # Project onto hyperplane sum(z) = target_sum
        residual = z.sum() - target_sum
        z -= residual / n

        # Clip to [0, max_val]
        old_z = z.copy()
        z = np.clip(z, 0.0, max_val)

        # Check convergence
        if np.allclose(z, old_z, atol=1e-10) and abs(z.sum() - target_sum) < 1e-6:
            break

    # Final correction to ensure exact sum
    diff = target_sum - z.sum()
    if abs(diff) > 1e-10:
        # Distribute residual among non-boundary variables
        free = (z > 1e-10) & (z < max_val - 1e-10)
        n_free = free.sum()
        if n_free > 0:
            z[free] += diff / n_free
        else:
            # All variables at boundary - adjust the one closest to interior
            margins = np.minimum(z, max_val - z)
            idx = np.argmax(margins)
            z[idx] += diff

    z = np.clip(z, 0.0, max_val)
    return z


def _round_to_integers_with_sum(
    z_continuous: np.ndarray, target_sum: int, max_val: int
) -> np.ndarray:
    """Round continuous z-values to integers while maintaining target sum.

    Greedy approach: round each value to the nearest integer, then adjust
    to hit the target sum by modifying the values with the smallest rounding error.
    """
    n = len(z_continuous)
    if n == 0:
        return np.array([], dtype=int)

    z_round = np.clip(np.round(z_continuous), 0, max_val).astype(int)
    current_sum = z_round.sum()
    diff = target_sum - current_sum

    if diff > 0:
        # Need to increase some values
        # Prefer increasing values that were rounded down the most
        fractional = z_continuous - z_round
        order = np.argsort(-fractional)  # most under-rounded first
        for idx in order:
            if diff <= 0:
                break
            increase = min(diff, max_val - z_round[idx])
            z_round[idx] += increase
            diff -= increase
    elif diff < 0:
        # Need to decrease some values
        fractional = z_round - z_continuous
        order = np.argsort(-fractional)  # most over-rounded first
        for idx in order:
            if diff >= 0:
                break
            decrease = min(-diff, z_round[idx])
            z_round[idx] -= decrease
            diff += decrease

    return z_round


def _get_pairs_at_position(k: int, dx: int, dy: int) -> list[tuple[int, int]]:
    """Get all (i, j) pairs contributing to digit position k."""
    pairs = []
    for i in range(min(k + 1, dx)):
        j = k - i
        if 0 <= j < dy:
            pairs.append((i, j))
    return pairs


def viterbi_factor_recovery(
    n: int,
    base: int,
    p: int | None = None,
    q: int | None = None,
    lambda_param: float = 1.0,
) -> ViterbiRecoveryResult:
    """Recover factors using Viterbi decoding with SVD prior over the carry chain.

    Algorithm:
    1. Solve least-squares carry system, take SVD to get x_est, y_est
    2. Build SVD-based z estimates: z_est_{ij} = x_est_i * y_est_j
    3. Run Viterbi over carry states, scoring transitions by the projection
       cost of z-values onto the carry-constraint simplex
    4. Backtrack to get the best carry sequence
    5. Recover z-values at each position by constrained projection + rounding
    6. Extract factors from Z matrix via SVD

    Parameters
    ----------
    n : int
        The semiprime to factor.
    base : int
        Base for digit representation.
    p, q : int, optional
        True factors (for evaluation only, not used in recovery).
    lambda_param : float
        Controls how much to trust the SVD prior vs carry constraints.
        Higher = trust SVD more.

    Returns
    -------
    ViterbiRecoveryResult
    """
    c = to_digits(n, base)
    d = len(c)
    _, dx, dy = _compute_digit_sizes(n, base)
    max_z_val = (base - 1) ** 2

    # Step 1: SVD estimates
    x_est, y_est, Z_star = _solve_svd_estimates(n, base, dx, dy)

    # Build z_est matrix
    z_est = np.outer(x_est, y_est)

    # Compute num_terms and max_carry at each position
    pairs_at: list[list[tuple[int, int]]] = []
    num_terms_at: list[int] = []
    for k in range(d):
        pairs = _get_pairs_at_position(k, dx, dy)
        pairs_at.append(pairs)
        num_terms_at.append(len(pairs))

    max_carry_at: list[int] = []
    max_t = 0
    for k in range(d):
        max_sum = num_terms_at[k] * max_z_val + max_t
        max_t = max_sum // base
        max_carry_at.append(max_t)

    # Step 2: Viterbi algorithm
    # States at position k: carry values t_k in {0, ..., max_carry_at[k]}
    # Transition from t_{k-1} to t_k at position k requires:
    #   sum z_{ij} = c_k - t_{k-1} + base * t_k  (the target sum S)
    #   with z_{ij} in [0, max_z_val]
    # Score = -lambda * min ||z - z_est||^2 subject to sum z = S, 0 <= z <= max_z_val

    # viterbi[k][t_k] = (best_log_score, best_t_prev)
    viterbi: list[dict[int, tuple[float, int]]] = []

    # Position 0: carry_in = 0 (no previous carry)
    vit0: dict[int, tuple[float, int]] = {}
    for t_out in range(max_carry_at[0] + 1):
        target_sum = c[0] + base * t_out
        if target_sum < 0 or target_sum > num_terms_at[0] * max_z_val:
            continue

        # Get z_est values for this position
        pairs = pairs_at[0]
        if len(pairs) == 0:
            if target_sum == 0:
                vit0[t_out] = (0.0, -1)
            continue

        z_est_k = np.array([z_est[i, j] for i, j in pairs])
        z_proj = _project_to_simplex_with_bounds(z_est_k, target_sum, max_z_val)
        cost = float(np.sum((z_proj - z_est_k) ** 2))
        log_score = -lambda_param * cost
        vit0[t_out] = (log_score, -1)

    viterbi.append(vit0)

    # Positions 1..d-1
    for k in range(1, d):
        vit_k: dict[int, tuple[float, int]] = {}
        prev = viterbi[k - 1]

        for t_out in range(max_carry_at[k] + 1):
            best_score = -np.inf
            best_t_in = -1

            for t_in, (score_in, _) in prev.items():
                target_sum = c[k] - t_in + base * t_out
                if target_sum < 0 or target_sum > num_terms_at[k] * max_z_val:
                    continue

                pairs = pairs_at[k]
                if len(pairs) == 0:
                    if target_sum == 0:
                        total_score = score_in
                        if total_score > best_score:
                            best_score = total_score
                            best_t_in = t_in
                    continue

                z_est_k = np.array([z_est[i, j] for i, j in pairs])
                z_proj = _project_to_simplex_with_bounds(
                    z_est_k, target_sum, max_z_val
                )
                cost = float(np.sum((z_proj - z_est_k) ** 2))
                total_score = score_in + (-lambda_param * cost)

                if total_score > best_score:
                    best_score = total_score
                    best_t_in = t_in

            if best_t_in >= 0:
                vit_k[t_out] = (best_score, best_t_in)

        viterbi.append(vit_k)

    # Step 3: Backtrack to get best carry sequence
    # Final carry must be 0
    if 0 not in viterbi[d - 1]:
        # If exact 0 not reachable, find the closest
        best_final = min(viterbi[d - 1].keys(), key=lambda t: abs(t))
    else:
        best_final = 0

    viterbi_log_likelihood = viterbi[d - 1].get(
        best_final, (-np.inf, -1)
    )[0]

    carry_sequence = [0] * d
    carry_sequence[d - 1] = best_final
    for k in range(d - 1, 0, -1):
        _, t_prev = viterbi[k].get(carry_sequence[k], (0.0, 0))
        carry_sequence[k - 1] = t_prev

    # Step 4: Recover z-values at each position using the recovered carry sequence
    Z_recovered = np.zeros((dx, dy))
    carry_in = 0
    for k in range(d):
        pairs = pairs_at[k]
        t_out = carry_sequence[k]
        target_sum = c[k] - carry_in + base * t_out

        if len(pairs) > 0 and 0 <= target_sum <= num_terms_at[k] * max_z_val:
            z_est_k = np.array([z_est[i, j] for i, j in pairs])
            z_proj = _project_to_simplex_with_bounds(z_est_k, target_sum, max_z_val)
            z_rounded = _round_to_integers_with_sum(z_proj, target_sum, max_z_val)

            for idx, (i, j) in enumerate(pairs):
                Z_recovered[i, j] = z_rounded[idx]

        carry_in = t_out

    # Step 5: Extract factors from recovered Z matrix via SVD
    U_r, S_r, Vt_r = np.linalg.svd(Z_recovered, full_matrices=False)
    s1_r = S_r[0] if len(S_r) > 0 else 0.0
    scale_r = np.sqrt(s1_r) if s1_r > 0 else 1.0
    x_viterbi = scale_r * U_r[:, 0]
    y_viterbi = scale_r * Vt_r[0, :]

    # Try all sign combinations for factor recovery
    viterbi_p, viterbi_q, viterbi_success = _try_recover_factors(
        x_viterbi, y_viterbi, n, base, dx, dy
    )

    # Also try direct column/row extraction from Z_recovered
    if not viterbi_success:
        viterbi_p, viterbi_q, viterbi_success = _try_extract_from_Z(
            Z_recovered, n, base, dx, dy
        )

    # Step 6: Greedy recovery baseline
    greedy_p, greedy_q, greedy_success, x_greedy, y_greedy = _greedy_svd_recovery(
        n, base, dx, dy, x_est, y_est, z_est
    )

    # Step 7: Naive rounding baseline
    naive_p, naive_q, naive_success, x_naive, y_naive = _naive_rounding(
        x_est, y_est, n, base
    )

    # Compute correlations with true factors
    true_carry_seq = None
    viterbi_cx = viterbi_cy = 0.0
    greedy_cx = greedy_cy = 0.0
    naive_cx = naive_cy = 0.0
    svd_cx = svd_cy = 0.0

    if p is not None and q is not None:
        x_true = np.array(to_digits(p, base), dtype=float)
        y_true = np.array(to_digits(q, base), dtype=float)
        while len(x_true) < dx:
            x_true = np.append(x_true, 0.0)
        while len(y_true) < dy:
            y_true = np.append(y_true, 0.0)

        # SVD correlations (pick best sign)
        svd_cx = max(abs(_corr(x_est, x_true)), abs(_corr(y_est, x_true)))
        svd_cy = max(abs(_corr(y_est, y_true)), abs(_corr(x_est, y_true)))

        # Viterbi correlations
        viterbi_cx = max(
            abs(_corr(x_viterbi, x_true)), abs(_corr(y_viterbi, x_true))
        )
        viterbi_cy = max(
            abs(_corr(y_viterbi, y_true)), abs(_corr(x_viterbi, y_true))
        )

        # Greedy correlations
        greedy_cx = max(abs(_corr(x_greedy, x_true)), abs(_corr(y_greedy, x_true)))
        greedy_cy = max(abs(_corr(y_greedy, y_true)), abs(_corr(x_greedy, y_true)))

        # Naive correlations
        naive_cx = max(abs(_corr(x_naive, x_true)), abs(_corr(y_naive, x_true)))
        naive_cy = max(abs(_corr(y_naive, y_true)), abs(_corr(x_naive, y_true)))

        # True carry sequence
        true_carry_seq = _compute_true_carry_sequence(p, q, base, dx, dy, d, c)

    return ViterbiRecoveryResult(
        n=n,
        p_true=p if p is not None else 0,
        q_true=q if q is not None else 0,
        base=base,
        d=d,
        dx=dx,
        dy=dy,
        recovered_p=viterbi_p,
        recovered_q=viterbi_q,
        recovery_success=viterbi_success,
        viterbi_log_likelihood=viterbi_log_likelihood,
        greedy_p=greedy_p,
        greedy_q=greedy_q,
        greedy_success=greedy_success,
        naive_p=naive_p,
        naive_q=naive_q,
        naive_success=naive_success,
        viterbi_corr_x=viterbi_cx,
        viterbi_corr_y=viterbi_cy,
        greedy_corr_x=greedy_cx,
        greedy_corr_y=greedy_cy,
        naive_corr_x=naive_cx,
        naive_corr_y=naive_cy,
        svd_corr_x=svd_cx,
        svd_corr_y=svd_cy,
        recovered_carry_sequence=carry_sequence,
        true_carry_sequence=true_carry_seq,
        lambda_param=lambda_param,
    )


def _try_recover_factors(
    x_est: np.ndarray,
    y_est: np.ndarray,
    n: int,
    base: int,
    dx: int,
    dy: int,
) -> tuple[int | None, int | None, bool]:
    """Try recovering factors from continuous estimates with sign flips."""
    for x_sign in [1, -1]:
        for y_sign in [1, -1]:
            xr = [max(0, min(base - 1, int(round(x_sign * v)))) for v in x_est[:dx]]
            yr = [max(0, min(base - 1, int(round(y_sign * v)))) for v in y_est[:dy]]
            pr = from_digits(xr, base)
            qr = from_digits(yr, base)
            if pr * qr == n and pr > 1 and qr > 1:
                return (min(pr, qr), max(pr, qr), True)
            # Also try swapping x and y roles
            xr2 = [max(0, min(base - 1, int(round(x_sign * v)))) for v in y_est[:dx]]
            yr2 = [max(0, min(base - 1, int(round(y_sign * v)))) for v in x_est[:dy]]
            pr2 = from_digits(xr2, base)
            qr2 = from_digits(yr2, base)
            if pr2 * qr2 == n and pr2 > 1 and qr2 > 1:
                return (min(pr2, qr2), max(pr2, qr2), True)
    return (None, None, False)


def _try_extract_from_Z(
    Z: np.ndarray,
    n: int,
    base: int,
    dx: int,
    dy: int,
) -> tuple[int | None, int | None, bool]:
    """Try extracting factors directly from Z matrix rows/columns.

    If Z is close to rank-1 with Z = x * y^T, then:
    - Z[0, :] = x_0 * y  (so y = Z[0, :] / x_0 if x_0 != 0)
    - Z[:, 0] = x * y_0
    We can try multiple rows/columns.
    """
    for pivot_row in range(min(dx, 3)):
        row = Z[pivot_row, :]
        if abs(row).max() < 0.5:
            continue
        # Treat this as proportional to y
        for pivot_col in range(min(dy, 3)):
            col = Z[:, pivot_col]
            if abs(col).max() < 0.5:
                continue
            # col is proportional to x
            xr = [max(0, min(base - 1, int(round(v)))) for v in col[:dx]]
            yr = [max(0, min(base - 1, int(round(v)))) for v in row[:dy]]
            pr = from_digits(xr, base)
            qr = from_digits(yr, base)
            if pr * qr == n and pr > 1 and qr > 1:
                return (min(pr, qr), max(pr, qr), True)
    return (None, None, False)


def _greedy_svd_recovery(
    n: int,
    base: int,
    dx: int,
    dy: int,
    x_est: np.ndarray,
    y_est: np.ndarray,
    z_est: np.ndarray,
) -> tuple[int | None, int | None, bool, np.ndarray, np.ndarray]:
    """Greedy carry-constrained rounding of SVD estimates.

    Process digit positions left-to-right (k=0, 1, ..., d-1).
    At each position, project z_est values onto the carry constraint
    and round to integers greedily.

    Returns (p, q, success, x_recovered, y_recovered).
    """
    c = to_digits(n, base)
    d = len(c)
    max_z_val = (base - 1) ** 2

    Z_greedy = np.zeros((dx, dy))
    carry_in = 0

    for k in range(d):
        pairs = _get_pairs_at_position(k, dx, dy)
        if len(pairs) == 0:
            continue

        # Find the best carry_out such that the target sum is feasible
        # and minimizes deviation from z_est
        z_est_k = np.array([z_est[i, j] for i, j in pairs])
        natural_sum = float(z_est_k.sum())

        # Target sum = c[k] - carry_in + base * carry_out
        # Want carry_out such that target_sum is closest to natural_sum
        # and 0 <= target_sum <= len(pairs) * max_z_val
        best_cost = np.inf
        best_t_out = 0
        best_z = z_est_k.copy()

        max_carry_out = (num_terms_at_k := len(pairs)) * max_z_val + carry_in
        max_t_out = max_carry_out // base

        for t_out in range(max_t_out + 1):
            target_sum = c[k] - carry_in + base * t_out
            if target_sum < 0 or target_sum > num_terms_at_k * max_z_val:
                continue

            z_proj = _project_to_simplex_with_bounds(z_est_k, target_sum, max_z_val)
            cost = float(np.sum((z_proj - z_est_k) ** 2))
            if cost < best_cost:
                best_cost = cost
                best_t_out = t_out
                best_z = z_proj

        # Round to integers
        target_sum = c[k] - carry_in + base * best_t_out
        if 0 <= target_sum <= num_terms_at_k * max_z_val:
            z_rounded = _round_to_integers_with_sum(best_z, target_sum, max_z_val)
            for idx, (i, j) in enumerate(pairs):
                Z_greedy[i, j] = z_rounded[idx]

        carry_in = best_t_out

    # Extract factors via SVD
    U_g, S_g, Vt_g = np.linalg.svd(Z_greedy, full_matrices=False)
    s1_g = S_g[0] if len(S_g) > 0 else 0.0
    scale_g = np.sqrt(s1_g) if s1_g > 0 else 1.0
    x_greedy = scale_g * U_g[:, 0]
    y_greedy = scale_g * Vt_g[0, :]

    p_rec, q_rec, success = _try_recover_factors(x_greedy, y_greedy, n, base, dx, dy)
    if not success:
        p_rec, q_rec, success = _try_extract_from_Z(Z_greedy, n, base, dx, dy)

    return (p_rec, q_rec, success, x_greedy, y_greedy)


def _naive_rounding(
    x_est: np.ndarray,
    y_est: np.ndarray,
    n: int,
    base: int,
) -> tuple[int | None, int | None, bool, np.ndarray, np.ndarray]:
    """Naive rounding of SVD estimates to nearest valid digit values.

    Returns (p, q, success, x_rounded_arr, y_rounded_arr).
    """
    dx = len(x_est)
    dy = len(y_est)

    best_p = None
    best_q = None
    success = False
    best_x = np.zeros(dx)
    best_y = np.zeros(dy)

    for x_sign in [1, -1]:
        for y_sign in [1, -1]:
            xr = np.array(
                [max(0, min(base - 1, int(round(x_sign * v)))) for v in x_est],
                dtype=float,
            )
            yr = np.array(
                [max(0, min(base - 1, int(round(y_sign * v)))) for v in y_est],
                dtype=float,
            )
            pr = from_digits([int(v) for v in xr], base)
            qr = from_digits([int(v) for v in yr], base)
            if pr * qr == n and pr > 1 and qr > 1:
                return (min(pr, qr), max(pr, qr), True, xr, yr)
            # Track the best attempt for correlation purposes
            if best_p is None:
                best_p = pr
                best_q = qr
                best_x = xr
                best_y = yr

    return (None, None, False, best_x, best_y)


def _compute_true_carry_sequence(
    p: int, q: int, base: int, dx: int, dy: int, d: int, c: list[int]
) -> list[int]:
    """Compute the true carry sequence for p * q in the given base."""
    x_digits = to_digits(p, base)
    y_digits = to_digits(q, base)
    while len(x_digits) < dx:
        x_digits.append(0)
    while len(y_digits) < dy:
        y_digits.append(0)

    carry_seq = []
    carry = 0
    for k in range(d):
        conv_sum = 0
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy:
                conv_sum += x_digits[i] * y_digits[j]
        total_at_k = conv_sum + carry
        t_k = (total_at_k - c[k]) // base
        carry_seq.append(t_k)
        carry = t_k

    return carry_seq


def sweep_lambda(
    n: int,
    base: int,
    p: int | None = None,
    q: int | None = None,
    lambdas: list[float] | None = None,
) -> list[ViterbiRecoveryResult]:
    """Run Viterbi recovery over a range of lambda values.

    Returns list of ViterbiRecoveryResult, one per lambda.
    """
    if lambdas is None:
        lambdas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = []
    for lam in lambdas:
        r = viterbi_factor_recovery(n, base, p, q, lambda_param=lam)
        results.append(r)
    return results


def run_experiment():
    """Run the full Viterbi recovery experiment and print comparison table."""
    cases = [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
        (1073, 29, 37),
        (5183, 71, 73),
        (10403, 101, 103),
        (25117, 139, 181),
    ]

    bases = [2, 10]
    lambdas_to_try = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("=" * 110)
    print("VITERBI RECOVERY EXPERIMENT: SVD prior + carry-chain Viterbi decoding")
    print("=" * 110)
    print()

    # Part 1: Find best lambda per case
    print("--- Part 1: Lambda sweep (base 2) ---")
    print()
    print(
        f"{'n':>10} {'lambda':>8} {'viterbi':>8} {'greedy':>8} "
        f"{'naive':>8} {'vit_cx':>8} {'vit_cy':>8} {'svd_cx':>8} {'svd_cy':>8}"
    )
    print("-" * 95)

    for n, p, q in cases:
        best_result = None
        best_lambda = 1.0
        best_corr_sum = -np.inf

        for lam in lambdas_to_try:
            r = viterbi_factor_recovery(n, 2, p, q, lambda_param=lam)
            corr_sum = r.viterbi_corr_x + r.viterbi_corr_y
            if r.recovery_success or (
                not (best_result and best_result.recovery_success)
                and corr_sum > best_corr_sum
            ):
                best_result = r
                best_lambda = lam
                best_corr_sum = corr_sum

        r = best_result
        v_str = "YES" if r.recovery_success else "no"
        g_str = "YES" if r.greedy_success else "no"
        n_str = "YES" if r.naive_success else "no"
        print(
            f"{n:>10} {best_lambda:>8.1f} {v_str:>8} {g_str:>8} "
            f"{n_str:>8} {r.viterbi_corr_x:>8.3f} {r.viterbi_corr_y:>8.3f} "
            f"{r.svd_corr_x:>8.3f} {r.svd_corr_y:>8.3f}"
        )

    # Part 2: Main comparison table across bases
    print()
    print("--- Part 2: Comparison across bases (lambda=1.0) ---")
    print()
    header = (
        f"{'n':>10} {'base':>5} {'naive':>8} {'greedy':>8} {'viterbi':>8} "
        f"{'svd_cx':>8} {'vit_cx':>8} {'corr_impr':>10}"
    )
    print(header)
    print("-" * len(header))

    for n, p, q in cases:
        for base in bases:
            r = viterbi_factor_recovery(n, base, p, q, lambda_param=1.0)
            n_str = "YES" if r.naive_success else "no"
            g_str = "YES" if r.greedy_success else "no"
            v_str = "YES" if r.recovery_success else "no"

            svd_avg = (r.svd_corr_x + r.svd_corr_y) / 2
            vit_avg = (r.viterbi_corr_x + r.viterbi_corr_y) / 2
            improvement = vit_avg - svd_avg

            print(
                f"{n:>10} {base:>5} {n_str:>8} {g_str:>8} {v_str:>8} "
                f"{svd_avg:>8.3f} {vit_avg:>8.3f} {improvement:>+10.3f}"
            )

    # Part 3: Carry sequence analysis
    print()
    print("--- Part 3: Carry sequence accuracy ---")
    print()
    print(
        f"{'n':>10} {'base':>5} {'carry_match':>12} {'carries_correct':>16} "
        f"{'viterbi_LL':>12}"
    )
    print("-" * 70)

    for n, p, q in cases[:8]:
        for base in [2, 10]:
            r = viterbi_factor_recovery(n, base, p, q, lambda_param=1.0)
            if r.true_carry_sequence is not None:
                matches = sum(
                    1
                    for a, b in zip(
                        r.recovered_carry_sequence, r.true_carry_sequence
                    )
                    if a == b
                )
                total = len(r.true_carry_sequence)
                pct = matches / total * 100 if total > 0 else 0
                print(
                    f"{n:>10} {base:>5} {pct:>11.1f}% "
                    f"{matches:>7}/{total:<7} "
                    f"{r.viterbi_log_likelihood:>12.2f}"
                )

    # Summary
    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print()
    print(
        "Viterbi decoding with SVD prior combines the spectral signal from the"
    )
    print(
        "least-squares solution with the Markov structure of carry propagation."
    )
    print()
    print("Key findings:")
    print(
        "- If Viterbi improves correlation over raw SVD: the carry chain structure"
    )
    print("  provides useful inductive bias beyond the spectral estimate alone.")
    print(
        "- If Viterbi recovers factors that naive/greedy miss: the prior+structure"
    )
    print("  combination overcomes the 2^{-d^2/4} barrier of Theorem C.")
    print(
        "- If Viterbi still fails: the rank-1 constraint remains the bottleneck,"
    )
    print("  even with structured search + good prior.")


if __name__ == "__main__":
    run_experiment()
