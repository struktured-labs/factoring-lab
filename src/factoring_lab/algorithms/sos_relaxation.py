"""Sum-of-Squares (SOS) / Lasserre hierarchy relaxations for digit convolution factoring.

The plain SDP relaxation (rank-1 -> PSD) yields ~30-39% integrality gap.
The Lasserre/SOS hierarchy systematically tightens this:

  - Degree-2 SOS = standard SDP relaxation (moment matrix on monomials of degree <= 1)
  - Degree-4 SOS = adds constraints on products of pairs (moment matrix on degree <= 2)
  - Degree-2k SOS captures all polynomial constraints up to degree 2k

At sufficiently high degree the relaxation becomes exact, but the moment matrix
grows exponentially: for d variables, the degree-2k moment matrix is
C(d+k, k) x C(d+k, k).

Research question: at what SOS degree does the relaxation become tight enough to
distinguish the true factorization from spurious lattice points?

We use cvxpy with its SDP backend (SCS) for small instances.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_digits(n: int, base: int) -> list[int]:
    """Convert n to base-b digits, least significant first."""
    if n == 0:
        return [0]
    digits: list[int] = []
    while n > 0:
        digits.append(n % base)
        n //= base
    return digits


def _from_digits(digits: list[int], base: int) -> int:
    """Convert base-b digits back to integer."""
    result = 0
    for i in reversed(range(len(digits))):
        result = result * base + digits[i]
    return result


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SOSResult:
    """Result of an SOS relaxation solve."""

    n: int
    base: int
    degree: int  # SOS degree (2 or 4)
    num_digit_vars: int  # number of x_i and y_j variables
    moment_matrix_size: int  # dimension of the moment matrix
    solve_time_seconds: float = 0.0
    solver_status: str = ""
    sos_gap: float = float("inf")  # distance from rank-1
    recovered_p: int | None = None
    recovered_q: int | None = None
    recovery_success: bool = False
    objective_value: float = float("inf")
    eigenvalues_top5: list[float] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Variable index management for moment matrices
# ---------------------------------------------------------------------------


def _monomial_indices_deg1(dx: int, dy: int) -> list[tuple[int, ...]]:
    """Generate degree-0 and degree-1 monomial multi-indices.

    Variables are x_0, ..., x_{dx-1}, y_0, ..., y_{dy-1}.
    Returns list of tuples where each tuple is a multi-index (var_index,).
    The empty tuple () represents the constant monomial 1.
    """
    monomials = [()]  # constant = 1
    for i in range(dx + dy):
        monomials.append((i,))
    return monomials


def _monomial_indices_deg2(dx: int, dy: int) -> list[tuple[int, ...]]:
    """Generate degree-0, 1, and 2 monomial multi-indices.

    For degree-4 SOS, the moment matrix is indexed by monomials up to degree 2.
    """
    d = dx + dy
    monomials = [()]  # constant = 1
    # degree 1
    for i in range(d):
        monomials.append((i,))
    # degree 2: pairs (i, j) with i <= j
    for i in range(d):
        for j in range(i, d):
            monomials.append((i, j))
    return monomials


# ---------------------------------------------------------------------------
# Degree-2 SOS relaxation (standard SDP)
# ---------------------------------------------------------------------------


def solve_sos_degree2(
    n: int,
    base: int,
    known_p: int | None = None,
    known_q: int | None = None,
) -> SOSResult:
    """Solve the degree-2 SOS (standard SDP) relaxation.

    The moment matrix M is indexed by {1, x_0, ..., x_{dx-1}, y_0, ..., y_{dy-1}}.
    M[0,0] = 1 (normalization).
    M[0, i+1] = E[x_i] or E[y_j] (first moments).
    M[i+1, j+1] = E[x_i * x_j] or E[x_i * y_j] or E[y_i * y_j] (second moments).

    Constraints:
    - M >> 0 (PSD)
    - Digit bounds: 0 <= E[x_i] <= b-1, 0 <= E[y_j] <= b-1
    - Carry propagation: linear in E[x_i * y_j] and carries
    - Bound constraints: E[x_i^2] <= (b-1) * E[x_i] (from x_i * (b-1-x_i) >= 0)
    """
    if not HAS_CVXPY:
        return SOSResult(
            n=n, base=base, degree=2, num_digit_vars=0,
            moment_matrix_size=0, notes="cvxpy not available",
        )

    c_digits = _to_digits(n, base)
    d = len(c_digits)
    dx = (d + 1) // 2
    dy = (d + 1) // 2

    # Total digit variables
    num_vars = dx + dy
    monomials = _monomial_indices_deg1(dx, dy)
    mm_size = len(monomials)  # 1 + dx + dy

    t0 = time.perf_counter()

    # Moment matrix variable (symmetric PSD)
    M = cp.Variable((mm_size, mm_size), symmetric=True)

    constraints = []

    # M >> 0
    constraints.append(M >> 0)

    # M[0,0] = 1 (normalization: E[1] = 1)
    constraints.append(M[0, 0] == 1)

    # Variable mapping: index i in monomials -> position in M
    # monomials[0] = () -> constant
    # monomials[1..dx] = (0,), (1,), ..., (dx-1,) -> x_i
    # monomials[dx+1..dx+dy] = (dx,), (dx+1,), ..., (dx+dy-1,) -> y_j

    # Digit bound constraints: 0 <= E[x_i] <= b-1
    for i in range(dx):
        idx = 1 + i  # position in moment matrix
        constraints.append(M[0, idx] >= 0)
        constraints.append(M[0, idx] <= base - 1)

    for j in range(dy):
        idx = 1 + dx + j
        constraints.append(M[0, idx] >= 0)
        constraints.append(M[0, idx] <= base - 1)

    # Tighter bounds from x_i * (b-1-x_i) >= 0:
    # E[x_i^2] <= (b-1) * E[x_i], i.e. M[i+1, i+1] <= (b-1) * M[0, i+1]
    for i in range(dx):
        idx = 1 + i
        constraints.append(M[idx, idx] <= (base - 1) * M[0, idx])
        constraints.append(M[idx, idx] >= 0)

    for j in range(dy):
        idx = 1 + dx + j
        constraints.append(M[idx, idx] <= (base - 1) * M[0, idx])
        constraints.append(M[idx, idx] >= 0)

    # Cross-moment bounds: 0 <= E[x_i * y_j] <= (b-1)^2
    for i in range(dx):
        for j in range(dy):
            idx_x = 1 + i
            idx_y = 1 + dx + j
            constraints.append(M[idx_x, idx_y] >= 0)
            constraints.append(M[idx_x, idx_y] <= (base - 1) ** 2)

    # Carry propagation constraints
    # At position k: sum_{i+j=k} E[x_i * y_j] + t_{k-1} - b*t_k = c_k
    # The carries t_k are auxiliary variables
    t = cp.Variable(d, nonneg=True)

    for k in range(d):
        conv_sum = 0
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy:
                idx_x = 1 + i
                idx_y = 1 + dx + j
                conv_sum = conv_sum + M[idx_x, idx_y]

        carry_in = t[k - 1] if k > 0 else 0
        carry_out = base * t[k]
        constraints.append(conv_sum + carry_in - carry_out == c_digits[k])

    # Upper bound on carries: t_k <= (b-1)^2 * min(k+1, dx, dy) / b + previous carry
    # Loose bound: t_k <= (dx * (b-1)^2) for simplicity
    max_carry = dx * (base - 1) ** 2
    for k in range(d):
        constraints.append(t[k] <= max_carry)

    # Objective: minimize trace of Z submatrix (proxy for rank minimization)
    # The Z submatrix is M[1:, 1:] — minimizing trace encourages low rank
    objective = cp.Minimize(cp.trace(M[1:, 1:]))

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000, eps=1e-6)
    except cp.SolverError as e:
        return SOSResult(
            n=n, base=base, degree=2, num_digit_vars=num_vars,
            moment_matrix_size=mm_size,
            solve_time_seconds=time.perf_counter() - t0,
            solver_status=f"SolverError: {e}",
            notes="SDP solver failed",
        )

    solve_time = time.perf_counter() - t0
    status = prob.status

    if M.value is None:
        return SOSResult(
            n=n, base=base, degree=2, num_digit_vars=num_vars,
            moment_matrix_size=mm_size,
            solve_time_seconds=solve_time,
            solver_status=status,
            notes="No solution returned by solver",
        )

    M_val = np.array(M.value)

    # Extract solution
    x_moments = M_val[0, 1:1 + dx]
    y_moments = M_val[0, 1 + dx:1 + dx + dy]

    # Compute SOS gap: how far is M from rank-1?
    eigvals = np.linalg.eigvalsh(M_val)
    eigvals_sorted = sorted(eigvals, reverse=True)

    # For a rank-1 M, only the top eigenvalue should be nonzero
    total_eig = sum(max(0, e) for e in eigvals_sorted)
    if total_eig > 1e-12:
        sos_gap = 1.0 - max(0, eigvals_sorted[0]) / total_eig
    else:
        sos_gap = 1.0

    # Try to extract factors by rounding moments
    recovered_p = None
    recovered_q = None
    recovery_success = False

    x_rounded = [max(0, min(base - 1, int(round(v)))) for v in x_moments]
    y_rounded = [max(0, min(base - 1, int(round(v)))) for v in y_moments]
    p_candidate = _from_digits(x_rounded, base)
    q_candidate = _from_digits(y_rounded, base)

    if p_candidate > 1 and q_candidate > 1 and p_candidate * q_candidate == n:
        recovered_p = min(p_candidate, q_candidate)
        recovered_q = max(p_candidate, q_candidate)
        recovery_success = True
    else:
        # Try swapping and neighbor search
        for dp in range(-2, 3):
            for dq in range(-2, 3):
                pc = p_candidate + dp
                qc = q_candidate + dq
                if pc > 1 and qc > 1 and pc * qc == n:
                    recovered_p = min(pc, qc)
                    recovered_q = max(pc, qc)
                    recovery_success = True
                    break
            if recovery_success:
                break

    return SOSResult(
        n=n,
        base=base,
        degree=2,
        num_digit_vars=num_vars,
        moment_matrix_size=mm_size,
        solve_time_seconds=solve_time,
        solver_status=status,
        sos_gap=sos_gap,
        recovered_p=recovered_p,
        recovered_q=recovered_q,
        recovery_success=recovery_success,
        objective_value=float(prob.value) if prob.value is not None else float("inf"),
        eigenvalues_top5=[float(e) for e in eigvals_sorted[:5]],
        notes="",
        extra={
            "x_moments": x_moments.tolist(),
            "y_moments": y_moments.tolist(),
            "x_rounded": x_rounded,
            "y_rounded": y_rounded,
            "p_candidate": p_candidate,
            "q_candidate": q_candidate,
        },
    )


# ---------------------------------------------------------------------------
# Degree-4 SOS relaxation (Lasserre level 2)
# ---------------------------------------------------------------------------


def solve_sos_degree4(
    n: int,
    base: int,
    known_p: int | None = None,
    known_q: int | None = None,
) -> SOSResult:
    """Solve the degree-4 SOS (Lasserre level 2) relaxation.

    The moment matrix is indexed by monomials up to degree 2:
    {1, x_0, ..., x_{dx-1}, y_0, ..., y_{dy-1}, x_0*x_0, x_0*x_1, ..., y_{dy-1}*y_{dy-1}}

    This is MUCH larger: for d digit variables, the matrix is
    1 + d + d*(d+1)/2 = O(d^2) in each dimension.

    Additional constraints beyond degree-2:
    - Localizing matrices for x_i*(b-1-x_i) >= 0
    - Consistency: M[x_i*x_j, x_k] = M[x_i, x_j*x_k] (moment matrix structure)
    - Products of carry propagation constraints

    For small instances only (dx+dy <= 8 or so).
    """
    if not HAS_CVXPY:
        return SOSResult(
            n=n, base=base, degree=4, num_digit_vars=0,
            moment_matrix_size=0, notes="cvxpy not available",
        )

    c_digits = _to_digits(n, base)
    d = len(c_digits)
    dx = (d + 1) // 2
    dy = (d + 1) // 2
    num_vars = dx + dy

    # Check feasibility: moment matrix size
    monomials = _monomial_indices_deg2(dx, dy)
    mm_size = len(monomials)

    # Warn if too large
    if mm_size > 120:
        return SOSResult(
            n=n, base=base, degree=4, num_digit_vars=num_vars,
            moment_matrix_size=mm_size,
            notes=f"Moment matrix {mm_size}x{mm_size} too large, skipping",
        )

    t0 = time.perf_counter()

    # Build monomial index lookup
    mono_to_idx: dict[tuple[int, ...], int] = {}
    for idx, m in enumerate(monomials):
        mono_to_idx[m] = idx

    def _canonical(mono: tuple[int, ...]) -> tuple[int, ...]:
        """Canonicalize a monomial (sort indices)."""
        return tuple(sorted(mono))

    def _product_mono(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        """Product of two monomials."""
        return _canonical(a + b)

    # Moment matrix variable
    M = cp.Variable((mm_size, mm_size), symmetric=True)
    constraints = []

    # M >> 0
    constraints.append(M >> 0)

    # M[0,0] = 1
    constraints.append(M[0, 0] == 1)

    # Moment matrix consistency: M[alpha, beta] depends only on alpha + beta
    # For monomials alpha, beta with deg(alpha) + deg(beta) <= 4,
    # M[alpha, beta] = E[x^{alpha+beta}]
    # This means if alpha+beta = gamma+delta, then M[alpha,beta] = M[gamma,delta]
    #
    # We enforce this by identifying all pairs that should be equal.
    consistency_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i, mi in enumerate(monomials):
        for j, mj in enumerate(monomials):
            if j < i:
                continue
            product = _product_mono(mi, mj)
            if len(product) <= 4:  # degree <= 4
                if product not in consistency_groups:
                    consistency_groups[product] = []
                consistency_groups[product].append((i, j))

    # For each group, constrain all entries to be equal
    for product, pairs in consistency_groups.items():
        if len(pairs) > 1:
            ref_i, ref_j = pairs[0]
            for pi, pj in pairs[1:]:
                constraints.append(M[pi, pj] == M[ref_i, ref_j])

    # Digit bound constraints on first moments
    for i in range(dx):
        idx = mono_to_idx[(i,)]
        constraints.append(M[0, idx] >= 0)
        constraints.append(M[0, idx] <= base - 1)

    for j in range(dy):
        idx = mono_to_idx[(dx + j,)]
        constraints.append(M[0, idx] >= 0)
        constraints.append(M[0, idx] <= base - 1)

    # Bound constraints from x_i*(b-1-x_i) >= 0:
    # E[x_i^2] <= (b-1)*E[x_i]
    for i in range(num_vars):
        mono_sq = _canonical((i, i))
        if mono_sq in mono_to_idx:
            idx_sq = mono_to_idx[mono_sq]
            idx_1 = mono_to_idx[(i,)]
            constraints.append(M[0, idx_sq] <= (base - 1) * M[0, idx_1])
            constraints.append(M[0, idx_sq] >= 0)

    # Stronger degree-4 bounds: x_i^2 * (b-1-x_i) >= 0 -> E[x_i^3] <= (b-1)*E[x_i^2]
    # and x_i * (b-1-x_i)^2 >= 0
    for i in range(num_vars):
        mono_cube = _canonical((i, i, i))
        mono_sq = _canonical((i, i))
        mono_1 = (i,)
        if mono_cube in mono_to_idx and mono_sq in mono_to_idx:
            # Find where E[x^3] lives in the moment matrix
            # E[x^3] = M[(i,), (i,i)] by consistency
            if (i,) in mono_to_idx and mono_sq in mono_to_idx:
                idx_1 = mono_to_idx[(i,)]
                idx_sq = mono_to_idx[mono_sq]
                # E[x^3] = M[idx_1, idx_sq]
                constraints.append(
                    M[idx_1, idx_sq] <= (base - 1) * M[0, idx_sq]
                )

    # Degree-4 constraint: x_i^2*(b-1-x_i)^2 >= 0
    # expands to E[x^4] - 2(b-1)*E[x^3] + (b-1)^2*E[x^2] >= 0
    for i in range(num_vars):
        mono_4 = _canonical((i, i, i, i))
        mono_3 = _canonical((i, i, i))
        mono_2 = _canonical((i, i))
        # E[x^4] = M[(i,i), (i,i)]
        if mono_2 in mono_to_idx:
            idx_sq = mono_to_idx[mono_2]
            # M[idx_sq, idx_sq] = E[x^4]
            # M[mono_to_idx[(i,)], idx_sq] = E[x^3] (from consistency)
            if (i,) in mono_to_idx:
                idx_1 = mono_to_idx[(i,)]
                constraints.append(
                    M[idx_sq, idx_sq]
                    - 2 * (base - 1) * M[idx_1, idx_sq]
                    + (base - 1) ** 2 * M[0, idx_sq]
                    >= 0
                )

    # Cross-moment bounds: 0 <= E[x_i * y_j] <= (b-1)^2
    for i in range(dx):
        for j in range(dy):
            mono_xy = _canonical((i, dx + j))
            if mono_xy in mono_to_idx:
                idx_xy = mono_to_idx[mono_xy]
                constraints.append(M[0, idx_xy] >= 0)
                constraints.append(M[0, idx_xy] <= (base - 1) ** 2)

    # Carry propagation constraints (same as degree-2 but using moment variables)
    t = cp.Variable(d, nonneg=True)
    max_carry = dx * (base - 1) ** 2

    for k in range(d):
        conv_sum = 0
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy:
                mono_xy = _canonical((i, dx + j))
                if mono_xy in mono_to_idx:
                    idx_xy = mono_to_idx[mono_xy]
                    conv_sum = conv_sum + M[0, idx_xy]

        carry_in = t[k - 1] if k > 0 else 0
        carry_out = base * t[k]
        constraints.append(conv_sum + carry_in - carry_out == c_digits[k])

    for k in range(d):
        constraints.append(t[k] <= max_carry)

    # Degree-4 rank-1 constraints (the key addition):
    # z_{ij} * z_{kl} = z_{il} * z_{kj}  (2x2 minors of Z matrix)
    # In moment terms: E[x_i*y_j*x_k*y_l] = E[x_i*y_l*x_k*y_j]
    # Which is: M[mono(x_i,y_j), mono(x_k,y_l)] = M[mono(x_i,y_l), mono(x_k,y_j)]
    # These are degree-4 moments that appear in the moment matrix.
    for i1 in range(dx):
        for j1 in range(dy):
            for i2 in range(i1, dx):
                for j2 in range(dy):
                    if i1 == i2 and j1 >= j2:
                        continue
                    # E[x_{i1}*y_{j1} * x_{i2}*y_{j2}] = E[x_{i1}*y_{j2} * x_{i2}*y_{j1}]
                    mono_left_a = _canonical((i1, dx + j1))
                    mono_left_b = _canonical((i2, dx + j2))
                    mono_right_a = _canonical((i1, dx + j2))
                    mono_right_b = _canonical((i2, dx + j1))

                    if (mono_left_a in mono_to_idx and mono_left_b in mono_to_idx
                            and mono_right_a in mono_to_idx and mono_right_b in mono_to_idx):
                        la = mono_to_idx[mono_left_a]
                        lb = mono_to_idx[mono_left_b]
                        ra = mono_to_idx[mono_right_a]
                        rb = mono_to_idx[mono_right_b]
                        # M[la, lb] = E[product of all 4]
                        # M[ra, rb] = E[same product rearranged]
                        # By moment consistency these should already be equal
                        # if the moment matrix is correct, but we add explicit
                        # constraints for the 2x2 minor conditions.
                        constraints.append(M[la, lb] == M[ra, rb])

    # Localizing matrix for x_i*(b-1-x_i) >= 0
    # For degree-4 SOS, this means the matrix
    # L_g indexed by degree-1 monomials, where
    # L_g[alpha, beta] = E[g * x^alpha * x^beta] = (b-1)*E[x^{alpha+beta+e_i}] - E[x^{alpha+beta+2*e_i}]
    # must be PSD.
    for var_idx in range(num_vars):
        loc_size = 1 + num_vars  # indexed by {1, x_0, ..., x_{d-1}}
        L = cp.Variable((loc_size, loc_size), symmetric=True)
        constraints.append(L >> 0)

        for a_idx, a_mono in enumerate([(), *[(v,) for v in range(num_vars)]]):
            for b_idx, b_mono in enumerate([(), *[(v,) for v in range(num_vars)]]):
                if b_idx < a_idx:
                    continue
                # L[a,b] = (b-1)*E[x^{a+b} * x_var] - E[x^{a+b} * x_var^2]
                ab = _product_mono(a_mono, b_mono)
                ab_v = _product_mono(ab, (var_idx,))
                ab_v2 = _product_mono(ab, (var_idx, var_idx))

                if len(ab_v) > 4 or len(ab_v2) > 4:
                    continue  # Beyond our moment order

                # Find these moments in the moment matrix
                # We need to find entries of M that equal E[ab_v] and E[ab_v2]
                # E[monomial] = M[alpha, beta] where alpha+beta = monomial
                # Choose alpha=(), beta=monomial if monomial has degree <= 2
                ab_v_canon = _canonical(ab_v)
                ab_v2_canon = _canonical(ab_v2)

                e_ab_v = _find_moment_entry(M, ab_v_canon, mono_to_idx, monomials)
                e_ab_v2 = _find_moment_entry(M, ab_v2_canon, mono_to_idx, monomials)

                if e_ab_v is not None and e_ab_v2 is not None:
                    constraints.append(
                        L[a_idx, b_idx] == (base - 1) * e_ab_v - e_ab_v2
                    )
                    if a_idx != b_idx:
                        constraints.append(
                            L[b_idx, a_idx] == (base - 1) * e_ab_v - e_ab_v2
                        )

    # Objective: minimize trace (proxy for rank minimization)
    objective = cp.Minimize(cp.trace(M))

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-5)
    except cp.SolverError as e:
        return SOSResult(
            n=n, base=base, degree=4, num_digit_vars=num_vars,
            moment_matrix_size=mm_size,
            solve_time_seconds=time.perf_counter() - t0,
            solver_status=f"SolverError: {e}",
            notes="SDP solver failed",
        )

    solve_time = time.perf_counter() - t0
    status = prob.status

    if M.value is None:
        return SOSResult(
            n=n, base=base, degree=4, num_digit_vars=num_vars,
            moment_matrix_size=mm_size,
            solve_time_seconds=solve_time,
            solver_status=status,
            notes="No solution returned by solver",
        )

    M_val = np.array(M.value)

    # Extract first moments (digit values)
    x_moments = np.array([M_val[0, mono_to_idx[(i,)]] for i in range(dx)])
    y_moments = np.array([M_val[0, mono_to_idx[(dx + j,)]] for j in range(dy)])

    # Compute SOS gap
    eigvals = np.linalg.eigvalsh(M_val)
    eigvals_sorted = sorted(eigvals, reverse=True)
    total_eig = sum(max(0, e) for e in eigvals_sorted)
    if total_eig > 1e-12:
        sos_gap = 1.0 - max(0, eigvals_sorted[0]) / total_eig
    else:
        sos_gap = 1.0

    # Try to recover factors
    recovered_p = None
    recovered_q = None
    recovery_success = False

    x_rounded = [max(0, min(base - 1, int(round(v)))) for v in x_moments]
    y_rounded = [max(0, min(base - 1, int(round(v)))) for v in y_moments]
    p_candidate = _from_digits(x_rounded, base)
    q_candidate = _from_digits(y_rounded, base)

    if p_candidate > 1 and q_candidate > 1 and p_candidate * q_candidate == n:
        recovered_p = min(p_candidate, q_candidate)
        recovered_q = max(p_candidate, q_candidate)
        recovery_success = True
    else:
        for dp in range(-2, 3):
            for dq in range(-2, 3):
                pc = p_candidate + dp
                qc = q_candidate + dq
                if pc > 1 and qc > 1 and pc * qc == n:
                    recovered_p = min(pc, qc)
                    recovered_q = max(pc, qc)
                    recovery_success = True
                    break
            if recovery_success:
                break

    return SOSResult(
        n=n,
        base=base,
        degree=4,
        num_digit_vars=num_vars,
        moment_matrix_size=mm_size,
        solve_time_seconds=solve_time,
        solver_status=status,
        sos_gap=sos_gap,
        recovered_p=recovered_p,
        recovered_q=recovered_q,
        recovery_success=recovery_success,
        objective_value=float(prob.value) if prob.value is not None else float("inf"),
        eigenvalues_top5=[float(e) for e in eigvals_sorted[:5]],
        notes="",
        extra={
            "x_moments": x_moments.tolist(),
            "y_moments": y_moments.tolist(),
            "x_rounded": x_rounded,
            "y_rounded": y_rounded,
            "p_candidate": p_candidate,
            "q_candidate": q_candidate,
        },
    )


def _find_moment_entry(
    M: cp.Variable,
    monomial: tuple[int, ...],
    mono_to_idx: dict[tuple[int, ...], int],
    monomials: list[tuple[int, ...]],
) -> Any | None:
    """Find the moment matrix entry corresponding to E[x^monomial].

    Decomposes monomial = alpha + beta where alpha, beta are in the monomial basis,
    then returns M[alpha_idx, beta_idx].
    """
    deg = len(monomial)
    if deg == 0:
        return M[0, 0]

    # Try splitting monomial into two parts, each of degree <= max_deg_in_basis
    max_deg = max(len(m) for m in monomials)

    for split in range(deg + 1):
        alpha = tuple(sorted(monomial[:split]))
        beta = tuple(sorted(monomial[split:]))
        if alpha in mono_to_idx and beta in mono_to_idx:
            return M[mono_to_idx[alpha], mono_to_idx[beta]]

    # Try all possible partitions
    indices = list(range(deg))
    for r in range(deg + 1):
        for subset in itertools.combinations(indices, r):
            complement = [i for i in indices if i not in subset]
            alpha = tuple(sorted(monomial[i] for i in subset))
            beta = tuple(sorted(monomial[i] for i in complement))
            if alpha in mono_to_idx and beta in mono_to_idx:
                return M[mono_to_idx[alpha], mono_to_idx[beta]]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sos_relaxation(
    n: int,
    base: int,
    degree: int = 2,
    known_p: int | None = None,
    known_q: int | None = None,
) -> SOSResult:
    """Run SOS relaxation at the specified degree.

    Args:
        n: integer to factor
        base: digit base
        degree: SOS degree (2 or 4)
        known_p, known_q: known factors for validation

    Returns:
        SOSResult with gap measurements and recovery status
    """
    if degree == 2:
        return solve_sos_degree2(n, base, known_p, known_q)
    elif degree == 4:
        return solve_sos_degree4(n, base, known_p, known_q)
    else:
        return SOSResult(
            n=n, base=base, degree=degree, num_digit_vars=0,
            moment_matrix_size=0, notes=f"Degree {degree} not implemented (only 2 and 4)",
        )
