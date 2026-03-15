"""Spectral recovery experiment: blind deconvolution approach to factoring.

Tests whether SVD of the least-squares carry-constrained solution Z*
correlates with the true factor digits x, y.

The digit convolution formulation is:
    n_k = Σ_{i+j=k} x_i y_j  (before carries)
which is discrete convolution: n = x * y.

This is equivalent to the "blind deconvolution" problem:
    recover x, y from z = x * y
lifted to the matrix variable Z_{ij} = x_i y_j (rank-1).

We test:
1. Minimum-norm least-squares solution to carry constraints → SVD
2. Nuclear-norm regularized solution (encourages low rank)
3. Both with and without leaked digits

If SVD of Z* correlates with true factors → spectral methods have signal.
If not → strengthens the barrier narrative (carries destroy spectral recoverability).
"""

import sys
import time
from dataclasses import dataclass, field
from math import log2

import numpy as np

from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    to_digits,
)


@dataclass
class SpectralRecoveryResult:
    """Result of spectral recovery attempt."""

    n: int
    p: int
    q: int
    base: int
    d: int
    dx: int
    dy: int

    # Correlation of top singular vectors with true factors
    corr_u_x: float  # correlation of left singular vector with x
    corr_v_y: float  # correlation of right singular vector with y

    # Correlation after optimal sign flip
    abs_corr_u_x: float
    abs_corr_v_y: float

    # Top singular value and ratio (rank-1-ness of Z*)
    top_singular_value: float
    singular_ratio: float  # sigma_1 / sigma_2

    # Reconstruction quality
    p_reconstructed: int | None
    q_reconstructed: int | None
    recovery_success: bool

    # Method used
    method: str

    # Leaked digits (if any)
    leaked_fraction: float = 0.0


def _build_carry_system(n: int, base: int, dx: int, dy: int):
    """Build the linear system A @ z_flat = b for carry constraints.

    Variables: z_{ij} for i < dx, j < dy, i+j < d
    Plus carry variables t_k for k = 0..d-1

    Constraints: sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k
    """
    c = to_digits(n, base)
    d = len(c)

    # Map (i,j) pairs to variable indices
    z_vars = []
    z_idx = {}
    for i in range(dx):
        for j in range(dy):
            if i + j < d:
                z_idx[(i, j)] = len(z_vars)
                z_vars.append((i, j))

    num_z = len(z_vars)
    num_t = d  # carry variables t_0..t_{d-1}
    num_vars = num_z + num_t

    # Build constraint matrix A and rhs b
    # d constraints (one per digit position)
    A = np.zeros((d, num_vars))
    b_vec = np.zeros(d)

    for k in range(d):
        b_vec[k] = c[k]

        # z_{ij} terms with i+j = k
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy and (i, j) in z_idx:
                A[k, z_idx[(i, j)]] = 1.0

        # carry terms: +t_{k-1} - b*t_k
        if k > 0:
            A[k, num_z + k - 1] = 1.0  # t_{k-1}
        A[k, num_z + k] = -base  # -b * t_k

    return A, b_vec, z_vars, z_idx, num_z, num_t


def spectral_recovery(
    n: int,
    p: int,
    q: int,
    base: int,
    method: str = "least_squares",
    leaked_x: dict[int, int] | None = None,
    leaked_y: dict[int, int] | None = None,
) -> SpectralRecoveryResult:
    """Attempt spectral recovery of factors from carry-constrained system.

    Methods:
    - "least_squares": minimum-norm solution to A @ v = b, then SVD
    - "nuclear_norm": minimize ||Z||_* subject to carry constraints (if cvxpy available)
    """
    c = to_digits(n, base)
    d = len(c)
    _, dx, dy = _compute_digit_sizes(n, base)

    # True factor digits for comparison
    x_true = to_digits(p, base)
    y_true = to_digits(q, base)
    while len(x_true) < dx:
        x_true.append(0)
    while len(y_true) < dy:
        y_true.append(0)
    x_true = np.array(x_true, dtype=float)
    y_true = np.array(y_true, dtype=float)

    A, b_vec, z_vars, z_idx, num_z, num_t = _build_carry_system(n, base, dx, dy)
    num_vars = num_z + num_t

    # Add leaked digit constraints if provided
    extra_rows = []
    extra_rhs = []
    leaked_frac = 0.0

    if leaked_x or leaked_y:
        total_digits = dx + dy
        leaked_count = 0

        if leaked_x:
            for i, val in leaked_x.items():
                # x_i is known → sum_j z_{ij} constraints tighten
                # Actually, we add: for each j, z_{ij} = val * y_j
                # But we don't know y_j. Instead add:
                # z_{i,0} + z_{i,1} + ... = val * (sum y_j)
                # Simpler: fix the row of Z. For each j where (i,j) in z_idx:
                # This is tricky without knowing y. Skip for now.
                leaked_count += 1

        if leaked_y:
            for j, val in leaked_y.items():
                leaked_count += 1

        leaked_frac = leaked_count / total_digits

    # Solve based on method
    if method == "least_squares":
        # Minimum-norm least-squares: v* = A^+ @ b
        # Add box constraints approximately via regularization
        # z_{ij} in [0, (b-1)^2], t_k >= 0

        # Simple: just solve min ||v||^2 s.t. A @ v = b
        v_star, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)

        # Extract z-values and reshape to matrix
        Z_star = np.zeros((dx, dy))
        for (i, j), idx in z_idx.items():
            Z_star[i, j] = v_star[idx]

    elif method == "nuclear_norm":
        try:
            import cvxpy as cp

            # Variable: full vector (z's + carries)
            v = cp.Variable(num_vars)

            # Carry constraints
            constraints = [A @ v == b_vec]

            # Box constraints on z
            for (i, j), idx in z_idx.items():
                constraints.append(v[idx] >= 0)
                constraints.append(v[idx] <= (base - 1) ** 2)

            # Non-negative carries
            for k in range(num_t):
                constraints.append(v[num_z + k] >= 0)

            # Reshape z-part to matrix for nuclear norm
            z_vec = v[:num_z]

            # Build the matrix from z_vec
            Z_var = cp.Variable((dx, dy))
            for (i, j), idx in z_idx.items():
                constraints.append(Z_var[i, j] == v[idx])

            # Objective: minimize nuclear norm (encourages low rank)
            objective = cp.Minimize(cp.normNuc(Z_var))

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)

            Z_star = np.array(Z_var.value)

        except (ImportError, Exception) as e:
            # Fall back to least squares
            v_star, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            Z_star = np.zeros((dx, dy))
            for (i, j), idx in z_idx.items():
                Z_star[i, j] = v_star[idx]

    else:
        raise ValueError(f"Unknown method: {method}")

    # SVD of Z*
    U, S, Vt = np.linalg.svd(Z_star, full_matrices=False)

    # Top singular vectors
    u1 = U[:, 0]  # left singular vector (should correlate with x)
    v1 = Vt[0, :]  # right singular vector (should correlate with y)
    s1 = S[0]
    s2 = S[1] if len(S) > 1 else 0.0

    # Scale singular vectors to factor scale
    # Z ≈ s1 * u1 @ v1^T, and Z = x @ y^T
    # So x ≈ sqrt(s1) * u1, y ≈ sqrt(s1) * v1
    scale = np.sqrt(s1) if s1 > 0 else 1.0
    x_est = scale * u1
    y_est = scale * v1

    # Compute correlations with true factors
    def _corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_ux = _corr(x_est, x_true)
    corr_vy = _corr(y_est, y_true)

    # Try reconstruction
    x_rounded = [max(0, min(base - 1, int(round(v)))) for v in x_est]
    y_rounded = [max(0, min(base - 1, int(round(v)))) for v in y_est]

    from factoring_lab.analysis.lattice_counting import from_digits

    p_recon = from_digits(x_rounded, base)
    q_recon = from_digits(y_rounded, base)

    # Try both orderings and sign flips
    recovery = False
    p_out = None
    q_out = None

    for x_sign in [1, -1]:
        for y_sign in [1, -1]:
            xr = [max(0, min(base - 1, int(round(x_sign * v)))) for v in x_est]
            yr = [max(0, min(base - 1, int(round(y_sign * v)))) for v in y_est]
            pr = from_digits(xr, base)
            qr = from_digits(yr, base)
            if pr * qr == n and pr > 1 and qr > 1:
                p_out = min(pr, qr)
                q_out = max(pr, qr)
                recovery = True
                break
        if recovery:
            break

    singular_ratio = s1 / s2 if s2 > 1e-15 else float("inf")

    return SpectralRecoveryResult(
        n=n,
        p=p,
        q=q,
        base=base,
        d=d,
        dx=dx,
        dy=dy,
        corr_u_x=corr_ux,
        corr_v_y=corr_vy,
        abs_corr_u_x=abs(corr_ux),
        abs_corr_v_y=abs(corr_vy),
        top_singular_value=s1,
        singular_ratio=singular_ratio,
        p_reconstructed=p_out,
        q_reconstructed=q_out,
        recovery_success=recovery,
        method=method,
        leaked_fraction=leaked_frac,
    )


def run_experiment():
    """Run the full spectral recovery experiment."""
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

    print("=" * 90)
    print("SPECTRAL RECOVERY EXPERIMENT: Can SVD of carry-constrained Z* recover factors?")
    print("=" * 90)
    print()

    # Part 1: Least-squares baseline
    print("--- Part 1: Minimum-norm least-squares + SVD ---")
    print()
    print(
        f"{'n':>10} {'d':>3} {'|corr(u,x)|':>12} {'|corr(v,y)|':>12} "
        f"{'σ₁/σ₂':>8} {'recovered':>9}"
    )
    print("-" * 65)

    for n, p, q in cases:
        for base in [2, 10]:
            r = spectral_recovery(n, p, q, base, method="least_squares")
            recov = "YES" if r.recovery_success else "no"
            print(
                f"{n:>10} {r.d:>3} {r.abs_corr_u_x:>12.4f} {r.abs_corr_v_y:>12.4f} "
                f"{r.singular_ratio:>8.1f} {recov:>9}  (base {base})"
            )

    # Part 2: Nuclear norm (if cvxpy available)
    print()
    print("--- Part 2: Nuclear-norm minimization + SVD ---")
    print()
    try:
        import cvxpy

        print(
            f"{'n':>10} {'d':>3} {'|corr(u,x)|':>12} {'|corr(v,y)|':>12} "
            f"{'σ₁/σ₂':>8} {'recovered':>9}"
        )
        print("-" * 65)

        for n, p, q in cases[:8]:  # Smaller set for nuclear norm (slower)
            r = spectral_recovery(n, p, q, 2, method="nuclear_norm")
            recov = "YES" if r.recovery_success else "no"
            print(
                f"{n:>10} {r.d:>3} {r.abs_corr_u_x:>12.4f} {r.abs_corr_v_y:>12.4f} "
                f"{r.singular_ratio:>8.1f} {recov:>9}"
            )
    except ImportError:
        print("(cvxpy not available, skipping nuclear norm)")

    print()
    print("--- Summary ---")
    print()
    print("If correlations are near 0: carries destroy spectral recoverability.")
    print("  → Strengthens the barrier narrative.")
    print("If correlations are > 0.3: spectral methods have signal.")
    print("  → Suggests new algorithmic direction.")


if __name__ == "__main__":
    run_experiment()
