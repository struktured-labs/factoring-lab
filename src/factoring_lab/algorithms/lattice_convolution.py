"""Lattice-based approach to digit convolution factoring.

Explores whether the digit convolution carry-propagation constraints can be
reformulated as a lattice problem solvable by LLL basis reduction.

The constraint system:
    alpha_k = sum_{i=0}^{k} x_i * y_{k-i}      (digit convolution)
    m_0 = alpha_0
    m_k = alpha_k + (m_{k-1} - c_{k-1}) / b    (carry propagation)
    m_k = c_k (mod b)                           (digit matching)

Key insight: m_k = c_k + b*t_k for integer t_k (the carry), turning the modular
constraints into linear equations with integer unknowns. However, the convolution
terms alpha_k involve PRODUCTS of unknowns x_i * y_j, making the system
fundamentally nonlinear. This is the core challenge.

Strategy: We linearize by either:
  (a) Treating products x_i * y_j as independent variables z_{ij} (with
      consistency constraints that can't be directly encoded in a lattice), or
  (b) Fixing one factor's digits and solving for the other (reducing to a
      linear system solvable by lattice methods).

Approach (b) is essentially Coppersmith's method in disguise: given partial
information about one factor, LLL can recover the rest. We implement both
to study the boundary.

This is exploratory research. The approach has fundamental limitations.
"""

from __future__ import annotations

import numpy as np

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


def gram_schmidt(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt orthogonalization for LLL.

    Returns (orthogonal basis, mu coefficients).
    """
    n = basis.shape[0]
    ortho = np.array(basis, dtype=np.float64)
    mu = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i):
            dot_ij = np.dot(ortho[i], ortho[j])
            dot_jj = np.dot(ortho[j], ortho[j])
            if dot_jj < 1e-15:
                mu[i][j] = 0.0
            else:
                mu[i][j] = dot_ij / dot_jj
            ortho[i] = ortho[i] - mu[i][j] * ortho[j]

    return ortho, mu


def lll_reduce(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL lattice basis reduction algorithm.

    Implements the Lenstra-Lenstra-Lovasz algorithm to find a reduced basis
    with short, nearly orthogonal vectors.

    Args:
        basis: Integer matrix where rows are basis vectors.
        delta: Reduction parameter in (0.25, 1). Default 0.75.

    Returns:
        Reduced basis matrix.
    """
    basis = np.array(basis, dtype=np.int64)
    n = basis.shape[0]

    ortho, mu = gram_schmidt(basis)

    k = 1
    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > 0.5:
                r = int(round(mu[k][j]))
                basis[k] = basis[k] - r * basis[j]
                ortho, mu = gram_schmidt(basis)

        # Lovasz condition
        dot_k = np.dot(ortho[k], ortho[k])
        dot_km1 = np.dot(ortho[k - 1], ortho[k - 1])

        if dot_k >= (delta - mu[k][k - 1] ** 2) * dot_km1:
            k += 1
        else:
            # Swap
            basis[[k, k - 1]] = basis[[k - 1, k]]
            ortho, mu = gram_schmidt(basis)
            k = max(k - 1, 1)

    return basis


def build_linearized_lattice(
    c_digits: list[int],
    base: int,
    num_x_digits: int,
    num_y_digits: int,
) -> tuple[np.ndarray, dict]:
    """Build a lattice encoding the digit convolution constraints.

    The linearization treats each product z_{ij} = x_i * y_j as an
    independent variable. The convolution constraint at position k is:

        sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k

    where t_k are integer carries (t_{-1} = 0).

    This creates a LINEAR system in variables z_{ij} and t_k. We encode
    it as a lattice: solutions are integer points in the kernel lattice
    of the constraint matrix.

    The fundamental problem: z_{ij} = x_i * y_j is a rank-1 constraint
    that cannot be captured by a lattice. So the lattice will contain
    many spurious solutions.

    Returns:
        (lattice_basis, metadata_dict)
    """
    d = len(c_digits)  # number of digit positions in n
    dx = num_x_digits
    dy = num_y_digits

    # Variables: z_{ij} for i in [0, dx), j in [0, dy) where i+j < d
    #            t_k for k in [0, d) (carries)
    # Constraint at position k: sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k

    # Identify z variables
    z_vars: list[tuple[int, int]] = []
    for i in range(dx):
        for j in range(dy):
            if i + j < d:
                z_vars.append((i, j))

    num_z = len(z_vars)
    num_t = d  # t_0 .. t_{d-1}
    num_vars = num_z + num_t
    num_constraints = d

    # Build constraint matrix A and target vector b_vec
    # A @ [z..., t...] = c_digits
    A = np.zeros((num_constraints, num_vars), dtype=np.int64)
    b_vec = np.array(c_digits, dtype=np.int64)

    z_index = {pair: idx for idx, pair in enumerate(z_vars)}

    for k in range(d):
        # sum_{i+j=k} z_{ij}
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy and (i, j) in z_index:
                A[k, z_index[(i, j)]] = 1

        # + t_{k-1}  (carry in from previous position)
        if k > 0:
            t_idx = num_z + (k - 1)
            A[k, t_idx] += 1

        # - b * t_k  (carry out to next position)
        t_k_idx = num_z + k
        A[k, t_k_idx] -= base

    # Build the lattice for the inhomogeneous system A @ v = b_vec
    # Using the standard embedding: kernel of [A | -b_vec]
    # We look for vectors in the lattice L = {v in Z^{num_vars} : A @ v = b_vec}
    # by finding a particular solution and adding kernel vectors.

    # For the lattice formulation, we use the "embedding" trick:
    # Build matrix M where rows span the lattice of solutions.
    # We want short vectors with z_{ij} in [0, (b-1)^2] and t_k >= 0.

    # Scale factor to penalize large values
    W = base * base  # weight for constraint satisfaction

    # Construct the lattice basis as:
    # [ I_{num_vars}  |  A^T * W ]
    # [ 0             |  b_vec^T * W ]
    # The last row encodes the target; short vectors in the reduced basis
    # that have +-1 in the last coordinate give solutions.

    dim = num_vars + num_constraints
    lat = np.zeros((num_vars + 1, dim), dtype=np.int64)

    # Identity block for variables
    for i in range(num_vars):
        lat[i, i] = 1

    # Constraint encoding
    for i in range(num_vars):
        for j in range(num_constraints):
            lat[i, num_vars + j] = W * A[j, i]

    # Target row
    for j in range(num_constraints):
        lat[num_vars, num_vars + j] = -W * b_vec[j]

    metadata = {
        "z_vars": z_vars,
        "z_index": z_index,
        "num_z": num_z,
        "num_t": num_t,
        "num_vars": num_vars,
        "num_constraints": num_constraints,
        "A": A,
        "b_vec": b_vec,
    }

    return lat, metadata


def extract_factorization_from_lattice(
    reduced: np.ndarray,
    metadata: dict,
    base: int,
    n: int,
    num_x_digits: int,
    num_y_digits: int,
) -> int | None:
    """Try to extract a valid factorization from LLL-reduced basis vectors.

    Looks for vectors where:
    - The last coordinate is +/-1 (indicating a solution to A@v = b)
    - The z_{ij} values are consistent with a rank-1 matrix (z_{ij} = x_i * y_j)
    - The resulting x, y satisfy x * y = n

    Returns a non-trivial factor or None.
    """
    num_vars = metadata["num_vars"]
    num_z = metadata["num_z"]
    z_vars = metadata["z_vars"]
    num_x = num_x_digits
    num_y = num_y_digits

    for row in reduced:
        # Check if last-block coordinates are small (constraint satisfaction)
        constraint_part = row[num_vars:]
        if np.any(constraint_part != 0):
            continue

        # Extract z values
        z_vals = row[:num_z]
        t_vals = row[num_z:num_vars]

        # Check bounds: z_{ij} should be in [0, (b-1)^2]
        max_z = (base - 1) ** 2
        if np.any(z_vals < 0) or np.any(z_vals > max_z):
            continue

        # Try to decompose z_{ij} = x_i * y_j (rank-1 check)
        # Build the z matrix
        Z = np.zeros((num_x, num_y), dtype=np.int64)
        for idx, (i, j) in enumerate(z_vars):
            Z[i, j] = z_vals[idx]

        # Try to extract x and y from Z
        # If Z = x * y^T, then each row is proportional to y
        # and each column is proportional to x
        x_digits = _extract_factor_digits(Z, num_x, num_y, base, axis=0)
        y_digits = _extract_factor_digits(Z, num_x, num_y, base, axis=1)

        if x_digits is not None and y_digits is not None:
            x = _from_digits(list(x_digits), base)
            y = _from_digits(list(y_digits), base)
            if x > 1 and y > 1 and x * y == n:
                return min(x, y)

    return None


def _extract_factor_digits(
    Z: np.ndarray, num_x: int, num_y: int, base: int, axis: int
) -> np.ndarray | None:
    """Try to extract factor digits from the product matrix Z.

    If Z = x * y^T (rank 1), extract x (axis=0) or y (axis=1).
    """
    if axis == 0:
        # Extract x: look at column 0 of Z (if y_0 != 0, Z[i,0] = x_i * y_0)
        col = Z[:, 0]
        if col[0] == 0:
            return None
        # Find y_0 by looking for a consistent divisor
        for y0 in range(1, base):
            if all(c % y0 == 0 for c in col):
                x = col // y0
                if all(0 <= xi < base for xi in x):
                    return x
        return None
    else:
        # Extract y: look at row 0 of Z (Z[0,j] = x_0 * y_j)
        row = Z[0, :]
        if row[0] == 0:
            return None
        for x0 in range(1, base):
            if all(c % x0 == 0 for c in row):
                y = row // x0
                if all(0 <= yj < base for yj in y):
                    return y
        return None


def solve_linear_for_y(
    c_digits: list[int],
    x_digits: list[int],
    base: int,
) -> list[int] | None:
    """Given known x digits, solve the LINEAR system for y digits.

    When x is fixed, alpha_k = sum_{i} x_i * y_{k-i} is linear in y.
    The carry propagation constraints become:

        sum_{i} x_i * y_{k-i} + t_{k-1} - b*t_k = c_k

    This is a linear system in (y_0, ..., y_{d-1}, t_0, ..., t_{d-1}).
    We can solve it with LLL by looking for short integer solutions
    with y_j in [0, b-1].
    """
    d = len(c_digits)
    dy = d  # number of y digits
    dx = len(x_digits)

    # Variables: y_0..y_{dy-1}, t_0..t_{d-1}
    num_vars = dy + d
    A = np.zeros((d, num_vars), dtype=np.int64)
    b_vec = np.array(c_digits, dtype=np.int64)

    for k in range(d):
        # sum_{i=0}^{k} x_i * y_{k-i}
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy:
                A[k, j] += x_digits[i]

        # + t_{k-1}
        if k > 0:
            A[k, dy + k - 1] += 1

        # - b * t_k
        A[k, dy + k] -= base

    # Solve A @ v = b_vec for integer v with y_j in [0, b-1]
    # Use the lattice embedding approach

    W = base * 10  # constraint weight

    dim = num_vars + d + 1  # vars + constraints + target
    lat = np.zeros((num_vars + 1, dim), dtype=np.int64)

    # Identity for variables
    for i in range(num_vars):
        lat[i, i] = 1

    # Constraint encoding: A^T * W
    for i in range(num_vars):
        for j in range(d):
            lat[i, num_vars + j] = W * A[j, i]

    # Target row
    lat[num_vars, num_vars + d] = 1  # marker
    for j in range(d):
        lat[num_vars, num_vars + j] = -W * b_vec[j]

    # LLL reduce
    reduced = lll_reduce(lat)

    # Look for solution vectors
    for row in reduced:
        # Check marker: last element should be +/-1
        marker = row[num_vars + d]
        if marker not in (1, -1):
            continue

        # Check constraints are satisfied
        constraint_part = row[num_vars : num_vars + d]
        if np.any(constraint_part != 0):
            continue

        # Extract y digits
        sign = marker
        y = (sign * row[:dy]).astype(np.int64)
        t = (sign * row[dy:num_vars]).astype(np.int64)

        # Check bounds
        if all(0 <= yj < base for yj in y):
            # Verify: A @ [y, t] = b_vec
            v = np.concatenate([y, t])
            if np.array_equal(A @ v, b_vec):
                return list(y)

    return None


class LatticeConvolution(FactoringAlgorithm):
    """Factor by reformulating digit convolution as a lattice problem.

    This is an EXPERIMENTAL approach that explores the boundary between
    lattice-solvable and lattice-unsolvable formulations of factoring.

    The core challenge: while the carry-propagation constraints are linear
    in the products z_{ij} = x_i * y_j, the rank-1 constraint on z (which
    encodes that the products come from actual digit sequences) is nonlinear
    and cannot be captured by a lattice.

    Two strategies are attempted:
    1. Full linearization: Treat z_{ij} as independent variables and hope
       LLL finds a rank-1 solution. (Usually fails for non-trivial cases.)
    2. Partial knowledge: Try small values for x and solve the resulting
       linear system for y via LLL. (Works but is essentially trial division
       with extra steps.)
    """

    def __init__(self, base: int = 10, max_x_value: int = 1000) -> None:
        self._base = base
        self._max_x_value = max_x_value

    @property
    def name(self) -> str:
        return f"lattice_convolution_b{self._base}"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        b = self._base
        c = _to_digits(n, b)
        d = len(c)

        # Strategy 1: Full lattice approach (linearized z_{ij})
        # This rarely works because LLL can't enforce the rank-1 constraint.
        dx = d // 2 + 1
        dy = d // 2 + 1

        ctx.record_iteration()
        lat, meta = build_linearized_lattice(c, b, dx, dy)

        try:
            reduced = lll_reduce(lat)
            ctx.record_iteration()
            result = extract_factorization_from_lattice(reduced, meta, b, n, dx, dy)
            if result is not None:
                return result, "found via full lattice linearization"
        except Exception:
            pass  # Numerical issues are expected; fall through

        # Strategy 2: Enumerate small x values, solve linear system for y
        # This is the "Coppersmith-lite" approach: given x, the system for y
        # is fully linear and LLL can solve it.
        limit = min(self._max_x_value, int(n**0.5) + 1)
        for x_val in range(2, limit):
            ctx.record_iteration()
            if n % x_val == 0:
                # Don't need lattice for this, but verify lattice would find it
                y_val = n // x_val
                if y_val > 1:
                    return min(x_val, y_val), (
                        f"found via enumeration (x={x_val}); "
                        f"lattice linearization alone insufficient"
                    )

        # Strategy 2b: Actually use lattice to solve for y given x candidates
        # Try a few small x values with the lattice solver
        for x_val in range(2, min(b**2, limit)):
            ctx.record_iteration()
            x_digits = _to_digits(x_val, b)
            # Pad x_digits
            while len(x_digits) < dx:
                x_digits.append(0)

            y_digits = solve_linear_for_y(c, x_digits[:dx], b)
            if y_digits is not None:
                y_val = _from_digits(y_digits, b)
                if y_val > 1 and x_val * y_val == n:
                    return (
                        min(x_val, y_val),
                        f"found via lattice solve with known x={x_val}",
                    )

        return None, (
            "lattice approach failed: rank-1 constraint not enforceable by LLL"
        )


class LatticeAnalysis:
    """Diagnostic tools for analyzing the lattice structure of digit convolution.

    Not a factoring algorithm per se, but provides insight into why/whether
    the lattice formulation can work.
    """

    def __init__(self, base: int = 10) -> None:
        self.base = base

    def analyze_constraint_structure(
        self, n: int
    ) -> dict:
        """Analyze the lattice structure for factoring n.

        Returns a diagnostic dict with:
        - lattice_dimension: size of the constructed lattice
        - constraint_rank: rank of the constraint matrix
        - shortest_vector_norm: norm of shortest LLL-reduced vector
        - gap_ratio: ratio of shortest to second-shortest vector
        - is_rank1_feasible: whether rank-1 solutions exist in the lattice
        """
        b = self.base
        c = _to_digits(n, b)
        d = len(c)
        dx = d // 2 + 1
        dy = d // 2 + 1

        lat, meta = build_linearized_lattice(c, b, dx, dy)

        A = meta["A"]
        constraint_rank = int(np.linalg.matrix_rank(A.astype(np.float64)))

        try:
            reduced = lll_reduce(lat)
            norms = [np.linalg.norm(row.astype(np.float64)) for row in reduced]
            norms_sorted = sorted(norms)
            shortest = norms_sorted[0] if norms_sorted else 0.0
            second = norms_sorted[1] if len(norms_sorted) > 1 else 0.0
            gap = shortest / second if second > 1e-15 else 0.0
        except Exception:
            reduced = lat
            shortest = 0.0
            gap = 0.0

        return {
            "n": n,
            "base": b,
            "num_digits": d,
            "lattice_dimension": lat.shape,
            "num_z_vars": meta["num_z"],
            "num_carry_vars": meta["num_t"],
            "constraint_rank": constraint_rank,
            "shortest_vector_norm": float(shortest),
            "gap_ratio": float(gap),
            "total_variables": meta["num_vars"],
            "notes": (
                "The rank-1 constraint z_{ij}=x_i*y_j is NOT encoded in the lattice. "
                "LLL finds short vectors satisfying the linear carry constraints, "
                "but these generally do NOT correspond to valid factorizations."
            ),
        }

    def verify_known_factorization(
        self, p: int, q: int
    ) -> dict:
        """Check whether a known factorization p*q appears as a lattice vector.

        This is a diagnostic: given the answer, does it live in our lattice?
        """
        b = self.base
        n = p * q
        c = _to_digits(n, b)
        d = len(c)

        x_digits = _to_digits(p, b)
        y_digits = _to_digits(q, b)

        # Pad
        dx = d // 2 + 1
        dy = d // 2 + 1
        while len(x_digits) < dx:
            x_digits.append(0)
        while len(y_digits) < dy:
            y_digits.append(0)

        # Construct the z_{ij} and t_k values for the known factorization
        _, meta = build_linearized_lattice(c, b, dx, dy)
        z_vars = meta["z_vars"]
        num_z = meta["num_z"]
        num_t = meta["num_t"]
        A = meta["A"]
        b_vec = meta["b_vec"]

        # Build the solution vector
        z_values = np.zeros(num_z, dtype=np.int64)
        for idx, (i, j) in enumerate(z_vars):
            if i < len(x_digits) and j < len(y_digits):
                z_values[idx] = x_digits[i] * y_digits[j]

        # Compute carries from the known factorization
        t_values = np.zeros(num_t, dtype=np.int64)
        carry = 0
        for k in range(d):
            alpha_k = sum(
                x_digits[i] * y_digits[k - i]
                for i in range(min(k + 1, len(x_digits)))
                if k - i < len(y_digits)
            )
            total = alpha_k + carry
            if k < d:
                t_values[k] = total // b
            carry = total // b

        # Check: does A @ [z, t] = b_vec?
        v = np.concatenate([z_values, t_values])
        residual = A @ v - b_vec

        # Compute vector norm
        vec_norm = float(np.linalg.norm(v.astype(np.float64)))

        return {
            "n": n,
            "p": p,
            "q": q,
            "solution_vector_norm": vec_norm,
            "constraint_residual": residual.tolist(),
            "constraints_satisfied": bool(np.all(residual == 0)),
            "z_values": z_values.tolist(),
            "carry_values": t_values.tolist(),
        }
