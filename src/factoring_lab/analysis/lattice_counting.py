"""Exact lattice point counting for the carry-propagation lattice.

Given a semiprime n in base b, the carry-propagation system is:

    sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k,   k = 0..d-1

with z_{ij} in [0, (b-1)^2], t_k in [0, max_carry].

This module counts ALL integer solutions (z, t) satisfying the linear
constraints and box bounds, then identifies which are rank-1 (valid
factorizations).

The heuristic from Lemma 4 of restricted_model_lower_bound.md says:
    |Lambda_n cap B| ~ ((b-1)^2 + 1)^{dx*dy} / b^d

We verify this by exact enumeration on small cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as cartesian_product
from math import comb, log2

import numpy as np


def to_digits(n: int, base: int) -> list[int]:
    """Convert n to base-b digits, least significant first."""
    if n == 0:
        return [0]
    digits: list[int] = []
    while n > 0:
        digits.append(n % base)
        n //= base
    return digits


def from_digits(digits: list[int], base: int) -> int:
    """Convert base-b digits back to integer."""
    result = 0
    for i in reversed(range(len(digits))):
        result = result * base + digits[i]
    return result


@dataclass
class LatticeCountResult:
    """Result of exact lattice point counting."""

    n: int
    base: int
    d: int  # number of digit positions
    dx: int  # number of x digit positions
    dy: int  # number of y digit positions
    num_z_vars: int
    total_lattice_points: int
    rank1_points: int
    heuristic_estimate: float
    ratio_exact_over_heuristic: float
    log2_exact: float
    log2_heuristic: float
    rank1_factorizations: list[tuple[int, int]]


def _compute_digit_sizes(n: int, base: int) -> tuple[int, int, int]:
    """Compute d, dx, dy for a given n and base.

    d is the number of digits of n. dx, dy are chosen so that
    dx + dy - 1 = d (products of dx-digit and dy-digit numbers have
    at most dx+dy-1 digits). We use dx = dy = ceil(d/2) + 1 to allow
    some slack, but cap them so dx + dy - 1 <= d.
    """
    c = to_digits(n, base)
    d = len(c)
    # For a balanced semiprime, each factor has about d/2 digits.
    # dx + dy - 1 <= d means products can have at most d digits.
    dx = (d + 1) // 2 + 1
    dy = (d + 1) // 2 + 1
    # Ensure dx + dy - 1 <= d (otherwise some z_{ij} with i+j >= d exist
    # but don't appear in constraints, which is fine - they're unconstrained
    # but still must satisfy bounds)
    # Actually, we only include z_{ij} with i+j < d in the constraint system.
    return d, dx, dy


def _build_constraint_system(
    c_digits: list[int], base: int, dx: int, dy: int
) -> tuple[list[tuple[int, int]], int]:
    """Build the z-variable list and count constraints.

    Returns (z_vars, d) where z_vars lists (i, j) pairs with i+j < d.
    """
    d = len(c_digits)
    z_vars: list[tuple[int, int]] = []
    for i in range(dx):
        for j in range(dy):
            if i + j < d:
                z_vars.append((i, j))
    return z_vars, d


def _max_carry(base: int, dx: int, dy: int, d: int) -> int:
    """Compute upper bound on carry at any position.

    At position k, the maximum value of sum_{i+j=k} z_{ij} is
    min(k+1, dx, dy) * (b-1)^2. Adding the carry from position k-1
    gives a recurrence. We compute the max carry iteratively.
    """
    max_z = (base - 1) ** 2
    max_t = 0
    for k in range(d):
        # Number of z_{ij} terms contributing to position k
        num_terms = 0
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy:
                num_terms += 1
        # Maximum sum at position k: num_terms * max_z + carry_in
        max_sum = num_terms * max_z + max_t
        # t_k = (max_sum - c_k) / b, but c_k >= 0 so upper bound is max_sum / b
        max_t = max_sum // base
    return max_t


def count_lattice_points_exact(
    n: int, base: int, dx: int | None = None, dy: int | None = None
) -> LatticeCountResult:
    """Count all integer points in Lambda_n cap B by direct enumeration.

    This uses a digit-by-digit recursive approach: process positions
    k = 0, 1, ..., d-1. At each position k, iterate over all valid
    assignments of z_{ij} (with i+j=k) and deduce the carry t_k.
    Prune branches where t_k is negative or exceeds the maximum.

    For small d (up to ~4-5 in base 10, ~8 in base 2), this is feasible.
    """
    c = to_digits(n, base)
    d = len(c)

    if dx is None or dy is None:
        _, dx, dy = _compute_digit_sizes(n, base)

    max_z = (base - 1) ** 2

    # Build list of z-variables per position k
    z_per_k: list[list[tuple[int, int]]] = [[] for _ in range(d)]
    for i in range(dx):
        for j in range(dy):
            k = i + j
            if k < d:
                z_per_k[k].append((i, j))

    # Compute max possible carry at each position
    max_carry_at: list[int] = [0] * d
    max_t = 0
    for k in range(d):
        max_sum = len(z_per_k[k]) * max_z + max_t
        max_t = max_sum // base
        max_carry_at[k] = max_t

    total_count = 0
    rank1_count = 0
    rank1_factorizations: list[tuple[int, int]] = []

    # We enumerate recursively: at position k, given carry_in t_{k-1},
    # enumerate all valid z-value tuples for position k, compute t_k.
    # Store all z_{ij} values to check rank-1 at the end.

    z_values: dict[tuple[int, int], int] = {}

    def recurse(k: int, carry_in: int) -> None:
        nonlocal total_count, rank1_count

        if k == d:
            # All positions processed. carry_in is the final carry.
            # For the product to equal n exactly, the final carry must be 0.
            if carry_in != 0:
                return

            total_count += 1

            # Check rank-1: does there exist x, y such that z_{ij} = x_i * y_j?
            Z = np.zeros((dx, dy), dtype=np.int64)
            for (i, j), v in z_values.items():
                Z[i, j] = v

            # Check all 2x2 minors
            is_rank1 = True
            for i1 in range(dx):
                if not is_rank1:
                    break
                for i2 in range(i1 + 1, dx):
                    if not is_rank1:
                        break
                    for j1 in range(dy):
                        if not is_rank1:
                            break
                        for j2 in range(j1 + 1, dy):
                            if Z[i1, j1] * Z[i2, j2] != Z[i1, j2] * Z[i2, j1]:
                                is_rank1 = False
                                break

            if is_rank1:
                # Extract all valid (x, y) decompositions
                decomps = _all_rank1_decompositions(Z, dx, dy, base)
                for x_digits, y_digits in decomps:
                    p_val = from_digits(x_digits, base)
                    q_val = from_digits(y_digits, base)
                    if p_val * q_val == n and p_val > 1 and q_val > 1:
                        rank1_count += 1
                        rank1_factorizations.append(
                            (min(p_val, q_val), max(p_val, q_val))
                        )
            return

        vars_k = z_per_k[k]
        if not vars_k:
            # No z variables at this position (shouldn't happen for k < d normally)
            # Constraint: carry_in - b * t_k = c_k => t_k = (carry_in - c_k) / b
            remainder = carry_in - c[k]
            if remainder < 0 or remainder % base != 0:
                return
            t_k = remainder // base
            if t_k < 0 or (k < d - 1 and t_k > max_carry_at[k]):
                return
            recurse(k + 1, t_k)
            return

        # Enumerate all valid z-value tuples for this position
        num_vars = len(vars_k)
        ranges = [range(0, max_z + 1) for _ in vars_k]

        for vals in cartesian_product(*ranges):
            z_sum = sum(vals)
            total_at_k = z_sum + carry_in

            # Constraint: z_sum + carry_in - b * t_k = c_k
            # => t_k = (z_sum + carry_in - c_k) / b
            remainder = total_at_k - c[k]
            if remainder < 0 or remainder % base != 0:
                continue
            t_k = remainder // base
            if t_k < 0:
                continue
            if k < d - 1 and t_k > max_carry_at[k]:
                continue

            # Set z values and recurse
            for var, val in zip(vars_k, vals):
                z_values[var] = val
            recurse(k + 1, t_k)

        # Clean up z_values for backtracking
        for var in vars_k:
            if var in z_values:
                del z_values[var]

    recurse(0, 0)

    # Compute heuristic estimate
    num_z = sum(len(z_per_k[k]) for k in range(d))
    heuristic = ((base - 1) ** 2 + 1) ** num_z / base**d

    log2_exact = log2(total_count) if total_count > 0 else 0.0
    log2_heur = log2(heuristic) if heuristic > 0 else 0.0
    ratio = total_count / heuristic if heuristic > 0 else float("inf")

    return LatticeCountResult(
        n=n,
        base=base,
        d=d,
        dx=dx,
        dy=dy,
        num_z_vars=num_z,
        total_lattice_points=total_count,
        rank1_points=rank1_count,
        heuristic_estimate=heuristic,
        ratio_exact_over_heuristic=ratio,
        log2_exact=log2_exact,
        log2_heuristic=log2_heur,
        rank1_factorizations=rank1_factorizations,
    )


def count_lattice_points_pruned(
    n: int, base: int, dx: int | None = None, dy: int | None = None
) -> LatticeCountResult:
    """Count lattice points with more aggressive pruning for larger cases.

    Uses tighter carry bounds and early termination when the remaining
    positions cannot possibly satisfy constraints.
    """
    # For now, delegate to exact counting. The pruning in the exact
    # version (carry bounds, divisibility checks) is already quite good.
    # This entry point exists for future optimization.
    return count_lattice_points_exact(n, base, dx, dy)


def _all_rank1_decompositions(
    Z: np.ndarray, dx: int, dy: int, base: int
) -> list[tuple[list[int], list[int]]]:
    """Find all valid (x, y) decompositions of a rank-1 matrix Z = x * y^T.

    Each x_i, y_j must be in [0, base-1]. Returns all valid (x, y) pairs.
    For the zero matrix, returns empty list (trivial case).
    """
    if np.all(Z == 0):
        return []

    max_digit = base - 1
    results: list[tuple[list[int], list[int]]] = []

    # Find a nonzero column to determine x (up to scaling)
    for j0 in range(dy):
        col = Z[:, j0]
        if np.any(col != 0):
            # Try all possible y_{j0} values as the scaling factor
            for yj0 in range(1, max_digit + 1):
                if not all(c % yj0 == 0 for c in col):
                    continue
                x = [int(c // yj0) for c in col]
                if not all(0 <= xi <= max_digit for xi in x):
                    continue
                # Now determine y from x: for each j, find y_j from a nonzero x_i
                y = [0] * dy
                y[j0] = yj0
                valid = True
                for j in range(dy):
                    if j == j0:
                        continue
                    col_j = Z[:, j]
                    # y_j = Z[i,j] / x_i for any i where x_i != 0
                    found = False
                    for i in range(dx):
                        if x[i] != 0:
                            if col_j[i] % x[i] != 0:
                                valid = False
                                break
                            y[j] = int(col_j[i] // x[i])
                            found = True
                            break
                    if not valid:
                        break
                    if not found:
                        # x is all zeros but Z isn't -> contradiction
                        # Actually if all x_i with valid (i,j) are 0
                        # then col_j should be all zeros too
                        if np.any(col_j != 0):
                            valid = False
                            break
                        y[j] = 0

                if not valid:
                    continue
                if not all(0 <= yj <= max_digit for yj in y):
                    continue

                # Verify: Z[i,j] = x[i] * y[j] for all constrained (i,j)
                ok = True
                for i in range(dx):
                    for j in range(dy):
                        if i + j < len(Z[0]) and Z[i, j] != x[i] * y[j]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    results.append((x, y))
            break  # Only need to process one nonzero column

    return results


def _is_zero_matrix(Z: np.ndarray) -> bool:
    """Check if Z is the zero matrix."""
    return bool(np.all(Z == 0))


def heuristic_estimate(n: int, base: int) -> tuple[float, int, int, int]:
    """Compute the Lemma 4 heuristic estimate.

    Returns (estimate, d, dx, dy, num_z).
    """
    c = to_digits(n, base)
    d = len(c)
    _, dx, dy = _compute_digit_sizes(n, base)

    z_per_k: list[list[tuple[int, int]]] = [[] for _ in range(d)]
    for i in range(dx):
        for j in range(dy):
            k = i + j
            if k < d:
                z_per_k[k].append((i, j))

    num_z = sum(len(z_per_k[k]) for k in range(d))
    est = ((base - 1) ** 2 + 1) ** num_z / base**d
    return est, d, dx, dy
