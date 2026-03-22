"""Belief Propagation / Approximate Message Passing on the carry chain factor graph.

Viterbi (sequential) fails because carry ambiguity cascades: errors at early
positions propagate unrecoverably to later positions. BP processes all digit
positions SIMULTANEOUSLY and passes messages bidirectionally.

The factor graph structure:
  - Variable nodes: x_0, ..., x_{dx-1}  (each in {0, ..., b-1})
  - Variable nodes: y_0, ..., y_{dy-1}  (each in {0, ..., b-1})
  - Factor nodes: for each digit position k, a factor connecting all (x_i, y_j)
    with i+j=k, plus carries t_{k-1} and t_k.
  - Constraint at position k:
        sum_{i+j=k} x_i * y_j + t_{k-1} - b * t_k = c_k

KEY INSIGHT: BP's advantage over Viterbi is that when x_3 participates in carry
constraints at positions k=3,4,5,..., BP receives messages from ALL those
positions simultaneously, not just one. This global consistency check is what
sequential methods lack.

APPROACH: We use a two-level message passing scheme:
  1. DISCRETE messages for carry factors with few variables (exact marginalization)
  2. GAUSSIAN approximate messages for carry factors with many variables

For small factors (few pairs at position k), we can exactly marginalize over
all partner variable beliefs and carry states. For large factors, we use the
mean-field Gaussian approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    from_digits,
    to_digits,
)
from factoring_lab.analysis.viterbi_recovery import (
    _corr,
    _get_pairs_at_position,
    _solve_svd_estimates,
    _try_extract_from_Z,
    _try_recover_factors,
)


@dataclass
class BPRecoveryResult:
    """Result of Belief Propagation factor recovery attempt."""

    n: int
    p_true: int
    q_true: int
    base: int
    d: int
    dx: int
    dy: int

    # BP recovery
    recovered_p: int | None
    recovered_q: int | None
    recovery_success: bool

    # Iteration info
    num_iterations: int
    converged: bool

    # Per-iteration correlation tracking (average of x and y correlations)
    per_iteration_correlation: list[float]

    # Final beliefs: probability tables for each variable
    # x_beliefs[i] is array of shape (base,) giving P(x_i = v)
    # y_beliefs[j] is array of shape (base,) giving P(y_j = v)
    x_beliefs: list[np.ndarray] = field(repr=False, default_factory=list)
    y_beliefs: list[np.ndarray] = field(repr=False, default_factory=list)

    # Final MAP estimates (argmax of beliefs)
    x_map: list[int] = field(default_factory=list)
    y_map: list[int] = field(default_factory=list)

    # Correlation metrics (only if true factors provided)
    bp_corr_x: float = 0.0
    bp_corr_y: float = 0.0
    svd_corr_x: float = 0.0
    svd_corr_y: float = 0.0

    # SVD lambda parameter used
    lambda_svd: float = 1.0


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to sum to 1 (probability distribution)."""
    s = arr.sum()
    if s > 0:
        return arr / s
    # Uniform fallback
    return np.ones_like(arr) / len(arr)


def _beliefs_to_map(beliefs: list[np.ndarray]) -> list[int]:
    """Extract MAP estimates (argmax) from belief arrays."""
    return [int(np.argmax(b)) for b in beliefs]


def _beliefs_to_means(beliefs: list[np.ndarray]) -> np.ndarray:
    """Extract means E[x_i] from belief arrays."""
    means = np.zeros(len(beliefs))
    for i, b in enumerate(beliefs):
        vals = np.arange(len(b), dtype=float)
        means[i] = np.dot(vals, b)
    return means


def _beliefs_to_variances(beliefs: list[np.ndarray]) -> np.ndarray:
    """Extract variances Var[x_i] from belief arrays."""
    variances = np.zeros(len(beliefs))
    for i, b in enumerate(beliefs):
        vals = np.arange(len(b), dtype=float)
        mean = np.dot(vals, b)
        variances[i] = np.dot(vals**2, b) - mean**2
    return variances


def _compute_carry_expectations(
    c: list[int],
    base: int,
    d: int,
    dx: int,
    dy: int,
    x_means: np.ndarray,
    y_means: np.ndarray,
) -> np.ndarray:
    """Compute the expected carry sequence t_0, ..., t_{d-1} under mean-field.

    Forward sweep: at each position k,
        expected_total_k = sum_{i+j=k} E[x_i]*E[y_j] + E[t_{k-1}]
        E[t_k] = (expected_total_k - c_k) / base
    """
    carries = np.zeros(d)
    carry_in = 0.0
    for k in range(d):
        pairs = _get_pairs_at_position(k, dx, dy)
        conv_sum = sum(x_means[i] * y_means[j] for i, j in pairs)
        total = conv_sum + carry_in
        carry_out = (total - c[k]) / base
        carries[k] = carry_out
        carry_in = carry_out
    return carries


def _compute_discrete_message_to_x(
    target_var_idx: int,
    pairs: list[tuple[int, int]],
    c_k: int,
    base: int,
    max_carry_in: int,
    max_carry_out: int,
    x_beliefs: list[np.ndarray],
    y_beliefs: list[np.ndarray],
    carry_in_dist: np.ndarray,
) -> np.ndarray:
    """Compute exact BP message from carry factor at position k to variable x_i.

    This marginalizes over all other variables at this position and over
    carry-in/carry-out states.

    For position k with pairs [(i0,j0), (i1,j1), ...], the factor enforces:
        sum_{(i,j) in pairs} x_i * y_j + t_{k-1} = c_k + base * t_k

    The message to x_{target_var_idx} is obtained by summing over all
    assignments to the OTHER variables and the carries, weighted by their
    beliefs.

    Parameters
    ----------
    target_var_idx : int
        The index i of the x variable we're computing the message for.
    pairs : list of (i, j)
        All (i, j) pairs at this position.
    c_k : int
        The k-th digit of n.
    base : int
        The base.
    max_carry_in, max_carry_out : int
        Range of carry values.
    x_beliefs, y_beliefs : current beliefs for all variables.
    carry_in_dist : distribution over carry-in values.

    Returns
    -------
    np.ndarray of shape (base,)
        The message (unnormalized).
    """
    msg = np.zeros(base)

    # Find which pair involves our target variable
    target_j = -1
    other_pairs = []
    for i, j in pairs:
        if i == target_var_idx:
            target_j = j
        else:
            other_pairs.append((i, j))

    if target_j < 0:
        return np.ones(base)  # variable not in this factor

    # For each value v of x_{target_var_idx}:
    # msg[v] = sum over all other assignments of
    #   P(y_{target_j} = w) * prod_{other pairs} P(x_a=va)*P(y_b=vb)
    #   * P(carry_in) * [constraint satisfied]
    #
    # We compute this by first computing the distribution of
    # "other_sum" = sum of x_a*y_b over other pairs, then checking
    # which carry values make the constraint work.

    # Compute the distribution of the sum of products over OTHER pairs
    # Using convolution of per-pair product distributions
    other_sum_dist = _compute_product_sum_distribution(
        other_pairs, x_beliefs, y_beliefs, base
    )

    # For each value v of x_i and w of y_j:
    y_belief = y_beliefs[target_j]
    for v in range(base):
        score = 0.0
        for w in range(base):
            prod_val = v * w
            # For each possible other_sum and carry_in:
            # constraint: prod_val + other_sum + carry_in = c_k + base * carry_out
            # carry_out = (prod_val + other_sum + carry_in - c_k) / base
            # Must be a non-negative integer
            for other_sum in range(len(other_sum_dist)):
                if other_sum_dist[other_sum] < 1e-300:
                    continue
                for t_in in range(max_carry_in + 1):
                    if carry_in_dist[t_in] < 1e-300:
                        continue
                    total = prod_val + other_sum + t_in
                    remainder = total - c_k
                    if remainder < 0:
                        continue
                    if remainder % base != 0:
                        continue
                    t_out = remainder // base
                    if t_out > max_carry_out:
                        continue
                    score += (
                        y_belief[w]
                        * other_sum_dist[other_sum]
                        * carry_in_dist[t_in]
                    )
        msg[v] = score

    return msg


def _compute_discrete_message_to_y(
    target_var_idx: int,
    pairs: list[tuple[int, int]],
    c_k: int,
    base: int,
    max_carry_in: int,
    max_carry_out: int,
    x_beliefs: list[np.ndarray],
    y_beliefs: list[np.ndarray],
    carry_in_dist: np.ndarray,
) -> np.ndarray:
    """Compute exact BP message from carry factor at position k to variable y_j.

    Same as _compute_discrete_message_to_x but for a y variable.
    """
    msg = np.zeros(base)

    target_i = -1
    other_pairs = []
    for i, j in pairs:
        if j == target_var_idx:
            target_i = i
        else:
            other_pairs.append((i, j))

    if target_i < 0:
        return np.ones(base)

    other_sum_dist = _compute_product_sum_distribution(
        other_pairs, x_beliefs, y_beliefs, base
    )

    x_belief = x_beliefs[target_i]
    for w in range(base):
        score = 0.0
        for v in range(base):
            prod_val = v * w
            for other_sum in range(len(other_sum_dist)):
                if other_sum_dist[other_sum] < 1e-300:
                    continue
                for t_in in range(max_carry_in + 1):
                    if carry_in_dist[t_in] < 1e-300:
                        continue
                    total = prod_val + other_sum + t_in
                    remainder = total - c_k
                    if remainder < 0:
                        continue
                    if remainder % base != 0:
                        continue
                    t_out = remainder // base
                    if t_out > max_carry_out:
                        continue
                    score += (
                        x_belief[v]
                        * other_sum_dist[other_sum]
                        * carry_in_dist[t_in]
                    )
        msg[w] = score

    return msg


def _compute_product_sum_distribution(
    pairs: list[tuple[int, int]],
    x_beliefs: list[np.ndarray],
    y_beliefs: list[np.ndarray],
    base: int,
) -> np.ndarray:
    """Compute the distribution of sum_{(i,j) in pairs} x_i * y_j.

    Uses convolution: start with delta at 0, for each pair convolve with
    the distribution of x_i * y_j.

    The maximum value of x_i * y_j is (base-1)^2, so the max sum is
    len(pairs) * (base-1)^2.
    """
    if not pairs:
        return np.array([1.0])  # delta at 0

    max_product = (base - 1) ** 2
    max_sum = len(pairs) * max_product

    # Start with delta at 0
    dist = np.zeros(max_sum + 1)
    dist[0] = 1.0

    for i, j in pairs:
        # Compute distribution of x_i * y_j
        prod_dist = np.zeros(max_product + 1)
        for v in range(base):
            for w in range(base):
                prod_dist[v * w] += x_beliefs[i][v] * y_beliefs[j][w]

        # Convolve dist with prod_dist
        new_dist = np.convolve(dist, prod_dist)
        # Trim to max_sum + 1
        dist = new_dist[: max_sum + 1]

    return dist


def _compute_gaussian_messages(
    k: int,
    pairs: list[tuple[int, int]],
    c_k: int,
    base: int,
    dx: int,
    dy: int,
    x_means: np.ndarray,
    y_means: np.ndarray,
    x_vars: np.ndarray,
    y_vars: np.ndarray,
    carries: np.ndarray,
    msg_to_x: list[dict[int, np.ndarray]],
    msg_to_y: list[dict[int, np.ndarray]],
    vals: np.ndarray,
) -> None:
    """Compute Gaussian approximate messages for position k (in-place).

    Used when the factor has too many variables for exact marginalization.
    """
    carry_in = carries[k - 1] if k > 0 else 0.0
    conv_sum = sum(x_means[i] * y_means[j] for i, j in pairs)
    total_expected = conv_sum + carry_in
    residual = total_expected - c_k - base * carries[k]

    conv_var = 0.0
    for i, j in pairs:
        conv_var += (
            x_vars[i] * y_means[j] ** 2
            + x_means[i] ** 2 * y_vars[j]
            + x_vars[i] * y_vars[j]
        )
    conv_var = max(conv_var, 1e-10)

    for i, j in pairs:
        ey = y_means[j]
        ex = x_means[i]

        pair_var = (
            x_vars[i] * ey**2
            + ex**2 * y_vars[j]
            + x_vars[i] * y_vars[j]
        )
        weight = pair_var / conv_var if conv_var > 1e-10 else 1.0 / len(pairs)
        credit = residual * weight
        ideal_product = ex * ey - credit

        if abs(ey) > 1e-10:
            ideal_x = ideal_product / ey
            precision = ey**2 / max(conv_var, 1e-10)
            log_msg = -0.5 * precision * (vals - ideal_x) ** 2
        else:
            log_msg = np.zeros(len(vals))

        log_msg -= log_msg.max()
        msg = _normalize(np.exp(log_msg))
        msg_to_x[i][k] = msg

        if abs(ex) > 1e-10:
            ideal_y = ideal_product / ex
            precision = ex**2 / max(conv_var, 1e-10)
            log_msg = -0.5 * precision * (vals - ideal_y) ** 2
        else:
            log_msg = np.zeros(len(vals))

        log_msg -= log_msg.max()
        msg = _normalize(np.exp(log_msg))
        msg_to_y[j][k] = msg


def bp_factor_recovery(
    n: int,
    base: int,
    p: int | None = None,
    q: int | None = None,
    max_iters: int = 100,
    damping: float = 0.5,
    lambda_svd: float = 1.0,
    convergence_tol: float = 1e-6,
    exact_threshold: int = 4,
) -> BPRecoveryResult:
    """Recover factors using Belief Propagation on the carry chain factor graph.

    Algorithm:
    1. Solve least-squares carry system, take SVD to get x_est, y_est
    2. Initialize beliefs: P(x_i = v) proportional to exp(-lambda * (v - x_est_i)^2)
    3. Iterative BP with carry factor messages:
       - For factors with few variables (<= exact_threshold pairs): exact discrete
         marginalization over partner beliefs and carry states
       - For factors with many variables: mean-field Gaussian approximation
    4. Update beliefs by combining SVD prior with all incoming messages (damped)
    5. Extract MAP estimates from converged beliefs.

    Parameters
    ----------
    n : int
        The semiprime to factor.
    base : int
        Base for digit representation.
    p, q : int, optional
        True factors (for evaluation only, not used in recovery).
    max_iters : int
        Maximum number of BP iterations.
    damping : float
        Damping factor in (0, 1]. new_belief = damping * old + (1-damping) * update.
    lambda_svd : float
        Controls how strongly the SVD prior influences initial beliefs.
    convergence_tol : float
        Stop when max belief change is below this threshold.
    exact_threshold : int
        Use exact discrete messages for factors with <= this many pairs.
        Set to 0 to force all-Gaussian mode.

    Returns
    -------
    BPRecoveryResult
    """
    c = to_digits(n, base)
    d = len(c)
    _, dx, dy = _compute_digit_sizes(n, base)

    # Step 1: SVD estimates
    x_est, y_est, Z_star = _solve_svd_estimates(n, base, dx, dy)

    # Step 2: Initialize beliefs from SVD prior
    x_beliefs: list[np.ndarray] = []
    y_beliefs: list[np.ndarray] = []

    vals = np.arange(base, dtype=float)

    # Try both sign orientations of SVD and pick the best one
    best_sign_x, best_sign_y = _pick_best_svd_signs(
        x_est, y_est, c, base, d, dx, dy, lambda_svd
    )

    x_est_signed = best_sign_x * x_est
    y_est_signed = best_sign_y * y_est

    for i in range(dx):
        log_prior = -lambda_svd * (vals - x_est_signed[i]) ** 2
        log_prior -= log_prior.max()
        belief = np.exp(log_prior)
        x_beliefs.append(_normalize(belief))

    for j in range(dy):
        log_prior = -lambda_svd * (vals - y_est_signed[j]) ** 2
        log_prior -= log_prior.max()
        belief = np.exp(log_prior)
        y_beliefs.append(_normalize(belief))

    # Build the SVD priors (fixed throughout iterations)
    x_priors = [b.copy() for b in x_beliefs]
    y_priors = [b.copy() for b in y_beliefs]

    # Precompute which positions each variable participates in
    x_positions: list[list[tuple[int, int]]] = [[] for _ in range(dx)]
    y_positions: list[list[tuple[int, int]]] = [[] for _ in range(dy)]
    position_pairs: list[list[tuple[int, int]]] = []

    for k in range(d):
        pairs = _get_pairs_at_position(k, dx, dy)
        position_pairs.append(pairs)
        for i, j in pairs:
            x_positions[i].append((k, j))
            y_positions[j].append((k, i))

    # Compute max carry at each position
    max_z_val = (base - 1) ** 2
    max_carry_at: list[int] = []
    max_t = 0
    for k in range(d):
        num_terms = len(position_pairs[k])
        max_sum = num_terms * max_z_val + max_t
        max_t = max_sum // base
        max_carry_at.append(max_t)

    # Initialize messages
    msg_to_x: list[dict[int, np.ndarray]] = [{} for _ in range(dx)]
    msg_to_y: list[dict[int, np.ndarray]] = [{} for _ in range(dy)]

    uniform = np.ones(base) / base
    for i in range(dx):
        for k, _ in x_positions[i]:
            msg_to_x[i][k] = uniform.copy()
    for j in range(dy):
        for k, _ in y_positions[j]:
            msg_to_y[j][k] = uniform.copy()

    # Prepare for tracking
    per_iteration_corr: list[float] = []
    converged = False

    # Compute true digit arrays for correlation tracking
    x_true_digits = None
    y_true_digits = None
    if p is not None and q is not None:
        x_true_digits = np.array(to_digits(p, base), dtype=float)
        y_true_digits = np.array(to_digits(q, base), dtype=float)
        while len(x_true_digits) < dx:
            x_true_digits = np.append(x_true_digits, 0.0)
        while len(y_true_digits) < dy:
            y_true_digits = np.append(y_true_digits, 0.0)

    # Step 3: BP iterations
    num_iters = 0
    for iteration in range(max_iters):
        num_iters = iteration + 1
        old_x_beliefs = [b.copy() for b in x_beliefs]
        old_y_beliefs = [b.copy() for b in y_beliefs]

        # Compute current means and variances
        x_means = _beliefs_to_means(x_beliefs)
        y_means = _beliefs_to_means(y_beliefs)
        x_vars = _beliefs_to_variances(x_beliefs)
        y_vars = _beliefs_to_variances(y_beliefs)

        # Compute expected carry sequence (for Gaussian mode and carry-in dist)
        carries = _compute_carry_expectations(
            c, base, d, dx, dy, x_means, y_means
        )

        # Compute carry-in distributions for discrete mode
        # Use forward propagation of the sum distribution
        carry_in_dists = _compute_carry_distributions(
            c, base, d, dx, dy, x_beliefs, y_beliefs,
            position_pairs, max_carry_at, exact_threshold,
        )

        # For each position k, compute factor-to-variable messages
        for k in range(d):
            pairs = position_pairs[k]
            if not pairs:
                continue

            num_pairs = len(pairs)
            max_cin = max_carry_at[k - 1] if k > 0 else 0
            max_cout = max_carry_at[k]

            # Decide whether to use exact discrete or Gaussian messages
            use_exact = (num_pairs <= exact_threshold and base <= 10)

            if use_exact:
                carry_in_dist = carry_in_dists[k]

                # Compute messages to each x variable at this position
                x_vars_at_k = set()
                y_vars_at_k = set()
                for i, j in pairs:
                    x_vars_at_k.add(i)
                    y_vars_at_k.add(j)

                for i in x_vars_at_k:
                    msg = _compute_discrete_message_to_x(
                        i, pairs, c[k], base,
                        max_cin, max_cout,
                        x_beliefs, y_beliefs,
                        carry_in_dist,
                    )
                    msg = _normalize(msg)
                    msg_to_x[i][k] = msg

                for j in y_vars_at_k:
                    msg = _compute_discrete_message_to_y(
                        j, pairs, c[k], base,
                        max_cin, max_cout,
                        x_beliefs, y_beliefs,
                        carry_in_dist,
                    )
                    msg = _normalize(msg)
                    msg_to_y[j][k] = msg
            else:
                _compute_gaussian_messages(
                    k, pairs, c[k], base, dx, dy,
                    x_means, y_means, x_vars, y_vars, carries,
                    msg_to_x, msg_to_y, vals,
                )

        # Update beliefs by combining prior with all incoming messages
        for i in range(dx):
            log_belief = np.log(np.maximum(x_priors[i], 1e-300))
            for k_pos, msg in msg_to_x[i].items():
                log_belief += np.log(np.maximum(msg, 1e-300))
            log_belief -= log_belief.max()
            new_belief = _normalize(np.exp(log_belief))

            # Damping
            x_beliefs[i] = damping * old_x_beliefs[i] + (1 - damping) * new_belief

        for j in range(dy):
            log_belief = np.log(np.maximum(y_priors[j], 1e-300))
            for k_pos, msg in msg_to_y[j].items():
                log_belief += np.log(np.maximum(msg, 1e-300))
            log_belief -= log_belief.max()
            new_belief = _normalize(np.exp(log_belief))

            # Damping
            y_beliefs[j] = damping * old_y_beliefs[j] + (1 - damping) * new_belief

        # Track correlation with true factors
        if x_true_digits is not None and y_true_digits is not None:
            x_map_arr = np.array(_beliefs_to_means(x_beliefs))
            y_map_arr = np.array(_beliefs_to_means(y_beliefs))
            cx = max(
                abs(_corr(x_map_arr, x_true_digits)),
                abs(_corr(y_map_arr, x_true_digits)),
            )
            cy = max(
                abs(_corr(y_map_arr, y_true_digits)),
                abs(_corr(x_map_arr, y_true_digits)),
            )
            per_iteration_corr.append((cx + cy) / 2)
        else:
            per_iteration_corr.append(0.0)

        # Check convergence
        max_change = 0.0
        for i in range(dx):
            max_change = max(
                max_change,
                float(np.max(np.abs(x_beliefs[i] - old_x_beliefs[i]))),
            )
        for j in range(dy):
            max_change = max(
                max_change,
                float(np.max(np.abs(y_beliefs[j] - old_y_beliefs[j]))),
            )

        if max_change < convergence_tol:
            converged = True
            break

    # Step 4: Extract MAP estimates
    x_map = _beliefs_to_map(x_beliefs)
    y_map = _beliefs_to_map(y_beliefs)

    # Try to recover factors from MAP estimates
    p_rec, q_rec, success = _try_map_recovery(x_map, y_map, n, base, dx, dy)

    # Also try from belief means (continuous) via rounding
    if not success:
        x_cont = _beliefs_to_means(x_beliefs)
        y_cont = _beliefs_to_means(y_beliefs)
        p_rec, q_rec, success = _try_recover_factors(
            x_cont, y_cont, n, base, dx, dy
        )

    # Also try building Z from belief means and using SVD/extraction
    if not success:
        x_cont = _beliefs_to_means(x_beliefs)
        y_cont = _beliefs_to_means(y_beliefs)
        Z_bp = np.outer(x_cont, y_cont)
        p_rec, q_rec, success = _try_extract_from_Z(Z_bp, n, base, dx, dy)

    # Compute correlation metrics
    bp_cx = bp_cy = svd_cx = svd_cy = 0.0
    if p is not None and q is not None and x_true_digits is not None:
        x_map_arr = np.array(_beliefs_to_means(x_beliefs))
        y_map_arr = np.array(_beliefs_to_means(y_beliefs))

        bp_cx = max(
            abs(_corr(x_map_arr, x_true_digits)),
            abs(_corr(y_map_arr, x_true_digits)),
        )
        bp_cy = max(
            abs(_corr(y_map_arr, y_true_digits)),
            abs(_corr(x_map_arr, y_true_digits)),
        )

        svd_cx = max(
            abs(_corr(x_est_signed, x_true_digits)),
            abs(_corr(y_est_signed, x_true_digits)),
        )
        svd_cy = max(
            abs(_corr(y_est_signed, y_true_digits)),
            abs(_corr(x_est_signed, y_true_digits)),
        )

    return BPRecoveryResult(
        n=n,
        p_true=p if p is not None else 0,
        q_true=q if q is not None else 0,
        base=base,
        d=d,
        dx=dx,
        dy=dy,
        recovered_p=p_rec,
        recovered_q=q_rec,
        recovery_success=success,
        num_iterations=num_iters,
        converged=converged,
        per_iteration_correlation=per_iteration_corr,
        x_beliefs=x_beliefs,
        y_beliefs=y_beliefs,
        x_map=x_map,
        y_map=y_map,
        bp_corr_x=bp_cx,
        bp_corr_y=bp_cy,
        svd_corr_x=svd_cx,
        svd_corr_y=svd_cy,
        lambda_svd=lambda_svd,
    )


def _compute_carry_distributions(
    c: list[int],
    base: int,
    d: int,
    dx: int,
    dy: int,
    x_beliefs: list[np.ndarray],
    y_beliefs: list[np.ndarray],
    position_pairs: list[list[tuple[int, int]]],
    max_carry_at: list[int],
    exact_threshold: int,
) -> list[np.ndarray]:
    """Compute approximate carry-in distributions at each position.

    For positions using exact discrete messages, we need the distribution
    P(t_{k-1} = s) to marginalize over carry-in states.

    We use a forward sweep: at each position, compute the distribution
    of the convolution sum + carry-in, then derive carry-out distribution.
    """
    carry_dists: list[np.ndarray] = []

    # Position 0: carry-in is deterministically 0
    carry_in_dist = np.array([1.0])  # P(t_{-1} = 0) = 1
    carry_dists.append(carry_in_dist)

    for k in range(d):
        pairs = position_pairs[k]
        max_cout = max_carry_at[k]

        if not pairs:
            # No convolution terms; carry propagates directly
            new_carry_dist = np.zeros(max_cout + 1)
            for t_in in range(len(carry_in_dist)):
                if carry_in_dist[t_in] < 1e-300:
                    continue
                remainder = t_in - c[k]
                if remainder >= 0 and remainder % base == 0:
                    t_out = remainder // base
                    if t_out <= max_cout:
                        new_carry_dist[t_out] += carry_in_dist[t_in]
            s = new_carry_dist.sum()
            if s > 0:
                new_carry_dist /= s
            else:
                new_carry_dist = np.zeros(max_cout + 1)
                new_carry_dist[0] = 1.0

            if k + 1 < d:
                carry_dists.append(new_carry_dist)
            carry_in_dist = new_carry_dist
            continue

        # Compute distribution of convolution sum at this position
        num_pairs = len(pairs)
        use_exact_sum = (num_pairs <= exact_threshold and base <= 10)

        if use_exact_sum:
            sum_dist = _compute_product_sum_distribution(
                pairs, x_beliefs, y_beliefs, base
            )
        else:
            # Gaussian approximation for the sum
            x_means = _beliefs_to_means(x_beliefs)
            y_means = _beliefs_to_means(y_beliefs)
            x_vars_arr = _beliefs_to_variances(x_beliefs)
            y_vars_arr = _beliefs_to_variances(y_beliefs)

            mean_sum = sum(x_means[i] * y_means[j] for i, j in pairs)
            var_sum = sum(
                x_vars_arr[i] * y_means[j] ** 2
                + x_means[i] ** 2 * y_vars_arr[j]
                + x_vars_arr[i] * y_vars_arr[j]
                for i, j in pairs
            )
            var_sum = max(var_sum, 0.01)

            max_sum_val = num_pairs * (base - 1) ** 2
            sum_dist = np.zeros(max_sum_val + 1)
            for s_val in range(max_sum_val + 1):
                sum_dist[s_val] = np.exp(
                    -0.5 * (s_val - mean_sum) ** 2 / var_sum
                )
            s = sum_dist.sum()
            if s > 0:
                sum_dist /= s

        # Compute carry-out distribution from sum_dist and carry_in_dist
        new_carry_dist = np.zeros(max_cout + 1)
        for t_in in range(len(carry_in_dist)):
            if carry_in_dist[t_in] < 1e-300:
                continue
            for s_val in range(len(sum_dist)):
                if sum_dist[s_val] < 1e-300:
                    continue
                total = s_val + t_in
                remainder = total - c[k]
                if remainder < 0:
                    continue
                if remainder % base != 0:
                    continue
                t_out = remainder // base
                if t_out <= max_cout:
                    new_carry_dist[t_out] += (
                        carry_in_dist[t_in] * sum_dist[s_val]
                    )

        s = new_carry_dist.sum()
        if s > 0:
            new_carry_dist /= s
        else:
            # Fallback: concentrate at the mean-field expected carry
            x_means = _beliefs_to_means(x_beliefs)
            y_means = _beliefs_to_means(y_beliefs)
            mean_cin = sum(
                t * carry_in_dist[t] for t in range(len(carry_in_dist))
            )
            mean_sum = sum(x_means[i] * y_means[j] for i, j in pairs)
            expected_cout = max(0, (mean_sum + mean_cin - c[k]) / base)
            t_est = int(round(expected_cout))
            t_est = max(0, min(max_cout, t_est))
            new_carry_dist = np.zeros(max_cout + 1)
            new_carry_dist[t_est] = 1.0

        if k + 1 < d:
            carry_dists.append(new_carry_dist)
        carry_in_dist = new_carry_dist

    return carry_dists


def _pick_best_svd_signs(
    x_est: np.ndarray,
    y_est: np.ndarray,
    c: list[int],
    base: int,
    d: int,
    dx: int,
    dy: int,
    lambda_svd: float,
) -> tuple[float, float]:
    """Pick the SVD sign orientation that best matches the digit constraints.

    The SVD decomposition has sign ambiguity (x, y) vs (-x, -y) vs (-x, y) etc.
    We pick the signs that make the SVD estimates closest to valid digit values
    (in [0, base-1]).
    """
    best_score = -np.inf
    best_sx, best_sy = 1.0, 1.0

    for sx in [1.0, -1.0]:
        for sy in [1.0, -1.0]:
            x_signed = sx * x_est
            y_signed = sy * y_est
            score = 0.0
            for i in range(dx):
                v = x_signed[i]
                if 0 <= v <= base - 1:
                    score += 1.0
                else:
                    score -= min(abs(v), abs(v - (base - 1)))
            for j in range(dy):
                v = y_signed[j]
                if 0 <= v <= base - 1:
                    score += 1.0
                else:
                    score -= min(abs(v), abs(v - (base - 1)))
            # Bonus for matching LSD constraint
            x0 = max(0, min(base - 1, round(x_signed[0])))
            y0 = max(0, min(base - 1, round(y_signed[0])))
            if x0 * y0 % base == c[0]:
                score += 5.0

            if score > best_score:
                best_score = score
                best_sx, best_sy = sx, sy

    return best_sx, best_sy


def _try_map_recovery(
    x_map: list[int],
    y_map: list[int],
    n: int,
    base: int,
    dx: int,
    dy: int,
) -> tuple[int | None, int | None, bool]:
    """Try recovering factors from MAP digit estimates."""
    pr = from_digits(x_map[:dx], base)
    qr = from_digits(y_map[:dy], base)
    if pr * qr == n and pr > 1 and qr > 1:
        return (min(pr, qr), max(pr, qr), True)

    # Try swapping x and y
    pr2 = from_digits(y_map[:dx], base)
    qr2 = from_digits(x_map[:dy], base)
    if pr2 * qr2 == n and pr2 > 1 and qr2 > 1:
        return (min(pr2, qr2), max(pr2, qr2), True)

    return (None, None, False)


def run_experiment():
    """Run BP recovery experiment and compare with Viterbi and naive rounding."""
    from factoring_lab.analysis.viterbi_recovery import viterbi_factor_recovery

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

    print("=" * 120)
    print("BELIEF PROPAGATION RECOVERY EXPERIMENT")
    print("=" * 120)
    print()
    print("Comparing BP (loopy belief propagation on carry factor graph)")
    print("vs Viterbi (sequential carry chain decoding) vs naive SVD rounding")
    print()

    header = (
        f"{'n':>10} {'base':>5} {'naive':>8} {'viterbi':>8} {'BP':>8} "
        f"{'svd_corr':>10} {'bp_corr':>10} {'vit_corr':>10} "
        f"{'bp_iters':>8} {'converged':>9}"
    )
    print(header)
    print("-" * len(header))

    bp_wins = 0
    vit_wins = 0
    bp_corr_total = 0.0
    vit_corr_total = 0.0
    svd_corr_total = 0.0
    count = 0

    for n_val, p_val, q_val in cases:
        for base in bases:
            bp_result = bp_factor_recovery(
                n_val, base, p_val, q_val,
                max_iters=100, damping=0.5, lambda_svd=1.0,
            )

            vit_result = viterbi_factor_recovery(
                n_val, base, p_val, q_val, lambda_param=1.0,
            )

            naive_str = "YES" if vit_result.naive_success else "no"
            vit_str = "YES" if vit_result.recovery_success else "no"
            bp_str = "YES" if bp_result.recovery_success else "no"

            svd_corr = (bp_result.svd_corr_x + bp_result.svd_corr_y) / 2
            bp_corr = (bp_result.bp_corr_x + bp_result.bp_corr_y) / 2
            vit_corr = (
                vit_result.viterbi_corr_x + vit_result.viterbi_corr_y
            ) / 2

            conv_str = "yes" if bp_result.converged else "no"

            print(
                f"{n_val:>10} {base:>5} {naive_str:>8} {vit_str:>8} "
                f"{bp_str:>8} "
                f"{svd_corr:>10.3f} {bp_corr:>10.3f} {vit_corr:>10.3f} "
                f"{bp_result.num_iterations:>8} {conv_str:>9}"
            )

            if bp_corr > vit_corr + 0.01:
                bp_wins += 1
            elif vit_corr > bp_corr + 0.01:
                vit_wins += 1
            bp_corr_total += bp_corr
            vit_corr_total += vit_corr
            svd_corr_total += svd_corr
            count += 1

    # Correlation evolution for a selected case
    print()
    print("--- Correlation Evolution (n=221, base=10) ---")
    print()
    bp_sel = bp_factor_recovery(
        221, 10, 13, 17, max_iters=50, damping=0.5
    )
    if bp_sel.per_iteration_correlation:
        print(f"{'Iter':>6} {'Avg Corr':>10}")
        print("-" * 20)
        step = max(1, len(bp_sel.per_iteration_correlation) // 15)
        for idx in range(
            0, len(bp_sel.per_iteration_correlation), step
        ):
            print(
                f"{idx + 1:>6} "
                f"{bp_sel.per_iteration_correlation[idx]:>10.4f}"
            )

    # Summary statistics
    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()
    print(f"BP wins (higher correlation): {bp_wins}/{count}")
    print(f"Viterbi wins: {vit_wins}/{count}")
    print(f"Average correlation -- SVD: {svd_corr_total/count:.3f}  "
          f"BP: {bp_corr_total/count:.3f}  "
          f"Viterbi: {vit_corr_total/count:.3f}")
    print()
    print("Belief Propagation on the carry chain factor graph provides global")
    print("consistency: each variable x_i receives messages from ALL digit")
    print("positions k where it participates (k = i, i+1, ..., i+dy-1).")
    print()
    print("Key observations:")
    print("- BP processes all positions simultaneously (vs Viterbi's L->R)")
    print("- Exact discrete messages for small factors, Gaussian for large")
    print("- Damped updates prevent oscillation in the loopy factor graph")
    print("- Convergence indicates consistent beliefs across constraints")


if __name__ == "__main__":
    run_experiment()
