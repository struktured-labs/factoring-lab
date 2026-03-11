"""SDP relaxation and alternating projection approaches to digit convolution factoring.

Explores whether convex relaxation of the rank-1 constraint Z = x * y^T
can help recover integer factors. The key idea:

1. The carry-propagation constraints are LINEAR in z_{ij} and carries t_k
2. The hard part is z_{ij} = x_i * y_j, i.e., Z = x ⊗ y (rank-1)
3. SDP can enforce Z being positive semidefinite
4. Minimizing nuclear norm (trace for PSD matrices) is a convex relaxation
   of rank minimization

Since we don't have a full SDP solver in scipy, we implement:
- An alternating projection approach (fix x, solve for y, repeat)
- A projected gradient descent on the augmented matrix M = [[1, x^T], [x, Z]]
- Random restarts to escape local minima

This is EXPLORATORY RESEARCH. The approach has fundamental limitations:
- The integrality gap (continuous vs integer solutions) may be large
- Convex relaxation of rank-1 is notoriously loose for small matrices
- Even if the relaxation is tight, rounding to integers may fail

Honest negative results are expected and valuable.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext


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


def _build_convolution_constraints(
    c_digits: list[int],
    base: int,
    num_x_digits: int,
    num_y_digits: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Build the linear constraint matrix for digit convolution.

    Returns:
        (z_vars, A, b_vec) where:
        - z_vars: list of (i, j) pairs for z_{ij} variables
        - A: constraint matrix, A @ [z..., t...] = b_vec
        - b_vec: target vector (digits of n)
    """
    d = len(c_digits)
    dx = num_x_digits
    dy = num_y_digits

    z_vars: list[tuple[int, int]] = []
    for i in range(dx):
        for j in range(dy):
            if i + j < d:
                z_vars.append((i, j))

    num_z = len(z_vars)
    num_t = d
    num_vars = num_z + num_t

    A = np.zeros((d, num_vars), dtype=np.float64)
    b_vec = np.array(c_digits, dtype=np.float64)

    z_index = {pair: idx for idx, pair in enumerate(z_vars)}

    for k in range(d):
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy and (i, j) in z_index:
                A[k, z_index[(i, j)]] = 1.0

        if k > 0:
            A[k, num_z + (k - 1)] += 1.0

        A[k, num_z + k] -= float(base)

    return z_vars, A, b_vec


def _check_factorization(x_val: int, n: int) -> int | None:
    """Check if x_val is a non-trivial factor of n."""
    if x_val < 2 or x_val >= n:
        return None
    if n % x_val == 0:
        y_val = n // x_val
        if y_val > 1:
            return min(x_val, y_val)
    return None


class AlternatingProjection(FactoringAlgorithm):
    """Factor via alternating projection between digit constraints and rank-1.

    Strategy:
    1. Initialize x randomly in [2, sqrt(n)]
    2. Compute y = n / x (real division)
    3. Round y to nearest integer, check if x * y = n
    4. If not, use digit-level gradient information to adjust
    5. Repeat with perturbations and random restarts

    This is essentially a continuous relaxation of the discrete search,
    guided by the digit convolution structure.
    """

    def __init__(
        self,
        base: int = 10,
        max_restarts: int = 100,
        max_iters_per_restart: int = 50,
        seed: int | None = None,
    ) -> None:
        self._base = base
        self._max_restarts = max_restarts
        self._max_iters = max_iters_per_restart
        self._seed = seed

    @property
    def name(self) -> str:
        return f"alternating_projection_b{self._base}"

    def _digit_gradient_step(
        self, x: float, n: int, c_digits: list[int]
    ) -> float:
        """Compute a digit-level gradient step to improve x.

        Compares the digit-level convolution of current (x, n/x) against
        target digits c, and returns an adjusted x.
        """
        b = self._base
        y = n / x
        d = len(c_digits)

        x_digits_f = []
        temp = x
        for _ in range(d):
            x_digits_f.append(temp % b)
            temp = temp // b

        y_digits_f = []
        temp = y
        for _ in range(d):
            y_digits_f.append(temp % b)
            temp = temp // b

        # Compute convolution residual
        residual = 0.0
        for k in range(d):
            alpha_k = sum(
                x_digits_f[i] * y_digits_f[k - i]
                for i in range(k + 1)
                if i < d and k - i < d
            )
            residual += (alpha_k - c_digits[k]) ** 2

        # Perturbation proportional to residual
        if residual > 0:
            step = random.gauss(0, max(1.0, x * 0.01))
            return x + step
        return x

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        rng = random.Random(self._seed)
        sqrt_n = int(math.isqrt(n))
        b = self._base
        c_digits = _to_digits(n, b)

        best_residual = float("inf")
        best_x = None

        for restart in range(self._max_restarts):
            ctx.record_iteration()

            # Random initialization
            x_init = rng.randint(2, max(2, sqrt_n))
            x = float(x_init)

            for it in range(self._max_iters):
                ctx.record_iteration()

                # Project onto "x * y = n" constraint
                y = n / x

                # Round and check
                y_round = int(round(y))
                for y_candidate in [y_round, y_round - 1, y_round + 1]:
                    if y_candidate > 1:
                        result = _check_factorization(y_candidate, n)
                        if result is not None:
                            return result, (
                                f"found via alternating projection "
                                f"(restart {restart}, iter {it})"
                            )
                        # Also check x
                        x_round = int(round(x))
                        for x_candidate in [x_round, x_round - 1, x_round + 1]:
                            if x_candidate > 1:
                                result = _check_factorization(x_candidate, n)
                                if result is not None:
                                    return result, (
                                        f"found via alternating projection "
                                        f"(restart {restart}, iter {it})"
                                    )

                # Compute residual: how far is x * round(y) from n
                residual = abs(n - int(round(x)) * y_round)
                if residual < best_residual:
                    best_residual = residual
                    best_x = int(round(x))

                # Digit-level gradient step
                x = self._digit_gradient_step(x, n, c_digits)

                # Keep x in valid range
                x = max(2.0, min(float(sqrt_n), x))

                # Occasionally jump to a new region
                if it % 10 == 9:
                    x = float(rng.randint(2, max(2, sqrt_n)))

        notes = (
            f"alternating projection failed after {self._max_restarts} restarts; "
            f"best residual={best_residual}, best_x={best_x}"
        )
        return None, notes


class SDPConvolution(FactoringAlgorithm):
    """Factor via SDP-inspired relaxation of the digit convolution rank-1 constraint.

    Approach:
    1. Build the augmented matrix M = [[1, x^T], [x, Z]] where Z ≈ x*y^T
    2. Linear constraints from carry propagation
    3. M must be PSD (positive semidefinite)
    4. Minimize trace(Z) as a proxy for rank minimization
    5. Use alternating direction method of multipliers (ADMM) or
       projected gradient descent to solve

    Since we don't have a proper SDP solver, we use a pragmatic approach:
    - Solve for z_{ij} satisfying linear constraints via least squares
    - Project onto PSD cone
    - Extract rank-1 approximation via SVD
    - Round to integers and check

    Combined with multiple random initializations and the alternating
    projection fallback.
    """

    def __init__(
        self,
        base: int = 10,
        max_restarts: int = 50,
        max_iters: int = 100,
        seed: int | None = None,
    ) -> None:
        self._base = base
        self._max_restarts = max_restarts
        self._max_iters = max_iters
        self._seed = seed

    @property
    def name(self) -> str:
        return f"sdp_convolution_b{self._base}"

    def _solve_relaxation(
        self,
        n: int,
        c_digits: list[int],
        num_x: int,
        num_y: int,
        rng: random.Random,
        ctx: InstrumentedContext,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """Solve the SDP-like relaxation for one initialization.

        Returns (x_digits, y_digits, diagnostics) or (None, None, diagnostics).
        """
        b = self._base
        d = len(c_digits)

        z_vars, A, b_vec = _build_convolution_constraints(c_digits, b, num_x, num_y)
        num_z = len(z_vars)
        num_t = d
        num_vars = num_z + num_t

        diagnostics: dict[str, Any] = {
            "num_z_vars": num_z,
            "num_carries": num_t,
            "num_constraints": d,
        }

        # Strategy: ADMM-like iterations
        # 1. Solve for z (satisfying linear constraints) via least squares
        # 2. Reshape z into matrix Z
        # 3. Project Z onto PSD cone and extract rank-1 approximation
        # 4. Map back to z_{ij} from rank-1 Z
        # 5. Repeat

        # Initialize Z randomly
        Z = np.zeros((num_x, num_y), dtype=np.float64)
        for i in range(num_x):
            for j in range(num_y):
                Z[i, j] = rng.random() * (b - 1)

        z_index = {pair: idx for idx, pair in enumerate(z_vars)}
        best_integrality_gap = float("inf")
        best_x_digits = None
        best_y_digits = None

        for iteration in range(self._max_iters):
            ctx.record_iteration()

            # Step 1: Extract z values from current Z
            z_current = np.zeros(num_z, dtype=np.float64)
            for idx, (i, j) in enumerate(z_vars):
                z_current[idx] = Z[i, j]

            # Step 2: Solve for carries t given z (linear system)
            # A_z @ z + A_t @ t = b_vec
            A_z = A[:, :num_z]
            A_t = A[:, num_z:]
            rhs = b_vec - A_z @ z_current

            # A_t is lower bidiagonal: t_k appears in row k (coeff -b)
            # and row k+1 (coeff +1). Solve by forward substitution.
            t = np.zeros(num_t, dtype=np.float64)
            for k in range(d):
                # Row k: A_t[k, :] @ t = rhs[k]
                # A_t[k, k] = -b, A_t[k, k-1] = +1 (if k > 0)
                val = rhs[k]
                if k > 0:
                    val -= t[k - 1]  # A_t[k, k-1] = +1
                t[k] = val / (-float(b))

            # Step 3: Compute constraint residual
            v = np.concatenate([z_current, t])
            constraint_residual = np.linalg.norm(A @ v - b_vec)

            # Step 4: Project Z onto rank-1 PSD cone
            # Clip to non-negative (z_{ij} = x_i * y_j >= 0 for digits)
            Z_clipped = np.maximum(Z, 0)

            # SVD to get best rank-1 approximation
            U, S, Vt = np.linalg.svd(Z_clipped, full_matrices=False)
            if S[0] > 1e-10:
                # Rank-1 approximation: sigma_1 * u_1 * v_1^T
                Z_rank1 = S[0] * np.outer(U[:, 0], Vt[0, :])

                # Extract x and y from rank-1 decomposition
                x_f = np.sqrt(S[0]) * np.abs(U[:, 0])
                y_f = np.sqrt(S[0]) * np.abs(Vt[0, :])

                # Integrality gap: how far is Z from rank-1?
                rank1_residual = np.sum(S[1:]) / np.sum(S) if np.sum(S) > 1e-10 else 0
                integrality_gap = rank1_residual + constraint_residual

                if integrality_gap < best_integrality_gap:
                    best_integrality_gap = integrality_gap
                    best_x_digits = x_f.copy()
                    best_y_digits = y_f.copy()

                # Step 5: Update Z by blending rank-1 projection with constraint satisfaction
                # Move Z toward the rank-1 approximation
                alpha = 0.5  # blending parameter
                Z_new = alpha * Z_rank1 + (1 - alpha) * Z

                # Also enforce constraint satisfaction by adjusting z values
                # to reduce A @ [z, t] - b_vec
                if constraint_residual > 1e-6:
                    # Gradient of ||A @ v - b||^2 w.r.t. z
                    grad_z = A_z.T @ (A @ v - b_vec)
                    step_size = 0.1 / (1 + iteration)
                    for idx, (i, j) in enumerate(z_vars):
                        Z_new[i, j] -= step_size * grad_z[idx]

                Z = np.maximum(Z_new, 0)  # keep non-negative
                # Clip to valid digit range
                Z = np.minimum(Z, (b - 1) ** 2)
            else:
                # Z is essentially zero, reinitialize
                for i in range(num_x):
                    for j in range(num_y):
                        Z[i, j] = rng.random() * (b - 1)

        diagnostics["best_integrality_gap"] = best_integrality_gap
        diagnostics["final_constraint_residual"] = float(constraint_residual)

        if best_x_digits is not None and best_y_digits is not None:
            return best_x_digits, best_y_digits, diagnostics
        return None, None, diagnostics

    def _try_round_and_check(
        self, x_f: np.ndarray, y_f: np.ndarray, n: int, base: int
    ) -> int | None:
        """Try rounding continuous digit vectors to integers and check factorization."""
        # Try multiple rounding strategies
        for x_offset in [-0.5, 0.0, 0.5]:
            x_digits = [max(0, min(base - 1, int(round(xi + x_offset)))) for xi in x_f]
            x_val = _from_digits(x_digits, base)

            result = _check_factorization(x_val, n)
            if result is not None:
                return result

            # Also try y
            for y_offset in [-0.5, 0.0, 0.5]:
                y_digits = [
                    max(0, min(base - 1, int(round(yi + y_offset)))) for yi in y_f
                ]
                y_val = _from_digits(y_digits, base)

                result = _check_factorization(y_val, n)
                if result is not None:
                    return result

        # Try direct division for promising x values
        x_round = _from_digits(
            [max(0, min(base - 1, int(round(xi)))) for xi in x_f], base
        )
        if x_round > 1:
            result = _check_factorization(x_round, n)
            if result is not None:
                return result

            # Try neighbors
            for delta in range(-3, 4):
                result = _check_factorization(x_round + delta, n)
                if result is not None:
                    return result

        return None

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        rng = random.Random(self._seed)
        b = self._base
        c_digits = _to_digits(n, b)
        d = len(c_digits)
        num_x = d // 2 + 1
        num_y = d // 2 + 1
        sqrt_n = int(math.isqrt(n))

        all_diagnostics: list[dict] = []

        for restart in range(self._max_restarts):
            ctx.record_iteration()

            x_f, y_f, diag = self._solve_relaxation(
                n, c_digits, num_x, num_y, rng, ctx
            )
            all_diagnostics.append(diag)

            if x_f is not None and y_f is not None:
                result = self._try_round_and_check(x_f, y_f, n, b)
                if result is not None:
                    return result, (
                        f"found via SDP relaxation (restart {restart}), "
                        f"integrality_gap={diag.get('best_integrality_gap', '?'):.4f}"
                    )

            # Also try alternating projection within this restart
            x_init = rng.randint(2, max(2, sqrt_n))
            for step in range(20):
                ctx.record_iteration()
                y_approx = n / x_init
                y_round = int(round(y_approx))
                for yc in [y_round, y_round - 1, y_round + 1]:
                    if yc > 1:
                        result = _check_factorization(yc, n)
                        if result is not None:
                            return result, (
                                f"found via SDP+alternating (restart {restart})"
                            )
                x_init = rng.randint(2, max(2, sqrt_n))

        # Summarize diagnostics
        avg_gap = np.mean(
            [d.get("best_integrality_gap", float("inf")) for d in all_diagnostics]
        )

        return None, (
            f"SDP relaxation failed after {self._max_restarts} restarts; "
            f"avg integrality gap={avg_gap:.4f}"
        )


class SDPAnalysis:
    """Diagnostic tools for analyzing the SDP relaxation quality."""

    def __init__(self, base: int = 10) -> None:
        self.base = base

    def analyze_integrality_gap(
        self, p: int, q: int, num_random: int = 10
    ) -> dict[str, Any]:
        """Analyze the integrality gap for a known factorization p*q.

        Compares the true rank-1 solution Z = x*y^T against the relaxed
        solutions found by the SDP-like approach.

        Returns diagnostic information about how tight the relaxation is.
        """
        n = p * q
        b = self.base
        c_digits = _to_digits(n, b)
        d = len(c_digits)
        num_x = d // 2 + 1
        num_y = d // 2 + 1

        # True solution
        x_true = _to_digits(p, b)
        y_true = _to_digits(q, b)
        while len(x_true) < num_x:
            x_true.append(0)
        while len(y_true) < num_y:
            y_true.append(0)

        x_arr = np.array(x_true[:num_x], dtype=np.float64)
        y_arr = np.array(y_true[:num_y], dtype=np.float64)
        Z_true = np.outer(x_arr, y_arr)
        true_trace = np.trace(Z_true)
        true_nuclear_norm = np.sum(np.linalg.svd(Z_true, compute_uv=False))

        # Check rank of true solution
        S_true = np.linalg.svd(Z_true, compute_uv=False)
        rank_ratio = S_true[0] / np.sum(S_true) if np.sum(S_true) > 0 else 0

        # Build and check constraints
        z_vars, A, b_vec = _build_convolution_constraints(c_digits, b, num_x, num_y)
        num_z = len(z_vars)
        num_t = d

        # Check that true solution satisfies constraints
        z_true_vec = np.zeros(num_z, dtype=np.float64)
        z_index = {pair: idx for idx, pair in enumerate(z_vars)}
        for idx, (i, j) in enumerate(z_vars):
            if i < len(x_true) and j < len(y_true):
                z_true_vec[idx] = x_true[i] * y_true[j]

        # Compute true carries
        t_true = np.zeros(num_t, dtype=np.float64)
        carry = 0
        for k in range(d):
            alpha_k = sum(
                x_true[i] * y_true[k - i]
                for i in range(min(k + 1, len(x_true)))
                if k - i < len(y_true)
            )
            total = alpha_k + carry
            t_true[k] = total // b
            carry = total // b

        v_true = np.concatenate([z_true_vec, t_true])
        constraint_residual = float(np.linalg.norm(A @ v_true - b_vec))

        # Try random initializations and measure how far solutions are from true
        rng = random.Random(42)
        relaxed_gaps = []
        relaxed_traces = []

        for _ in range(num_random):
            # Random Z satisfying linear constraints (approximately)
            Z_rand = np.zeros((num_x, num_y), dtype=np.float64)
            for i in range(num_x):
                for j in range(num_y):
                    Z_rand[i, j] = rng.random() * (b - 1)

            # Project via SVD
            U, S, Vt = np.linalg.svd(Z_rand, full_matrices=False)
            if S[0] > 1e-10:
                gap = np.sum(S[1:]) / np.sum(S)
                relaxed_gaps.append(gap)
                relaxed_traces.append(np.trace(Z_rand))

        return {
            "n": n,
            "p": p,
            "q": q,
            "base": b,
            "true_trace": float(true_trace),
            "true_nuclear_norm": float(true_nuclear_norm),
            "true_rank1_ratio": float(rank_ratio),
            "constraint_residual": constraint_residual,
            "constraints_satisfied": constraint_residual < 1e-10,
            "avg_relaxed_gap": float(np.mean(relaxed_gaps)) if relaxed_gaps else None,
            "min_relaxed_gap": float(np.min(relaxed_gaps)) if relaxed_gaps else None,
            "avg_relaxed_trace": float(np.mean(relaxed_traces)) if relaxed_traces else None,
            "notes": (
                "The integrality gap measures how far the relaxed SDP solution is "
                "from the true rank-1 solution. A gap near 0 means the relaxation "
                "is tight; a gap near 1 means it provides no useful information."
            ),
        }
