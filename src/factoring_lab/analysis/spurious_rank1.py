"""Spurious near-rank-1 lattice point analysis.

When minimizing the nuclear norm of Z subject to carry constraints, the solver
finds nearly rank-1 solutions (sigma_1/sigma_2 ~ 10^5) that do NOT correspond
to the true factorization. This module investigates that phenomenon by:

1. Enumerating ALL lattice points (integer solutions to carry-propagation
   constraints) for small semiprimes.
2. Computing the SVD of each lattice point's Z matrix to obtain a
   "rank deficiency" score: sigma_2 / sigma_1 (0 = exactly rank-1,
   1 = maximally far from rank-1).
3. Classifying lattice points by their rank deficiency at multiple thresholds,
   producing histograms that reveal the distribution of near-rank-1 points.

Key question: are approximately-rank-1 points sparse (2^{-Omega(d)}) or common?
- If sparse: stronger barrier result (even approximate rank-1 is hard to find).
- If common: explains why nuclear norm / SDP relaxations find them easily,
  and the true barrier is distinguishing genuine from spurious rank-1 points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

import numpy as np

from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    to_digits,
    from_digits,
)


# Default thresholds for classifying near-rank-1 points
DEFAULT_THRESHOLDS = [0.1, 0.01, 0.001, 0.0001]


@dataclass
class RankProfile:
    """Rank profile for a single lattice point's Z matrix."""

    z_matrix: np.ndarray
    sigma_1: float
    sigma_2: float
    rank_deficiency: float  # sigma_2 / sigma_1 (0 = exactly rank-1)
    is_exact_rank1: bool    # True if all 2x2 minors vanish (integer test)
    is_valid_factorization: bool  # True if rank-1 and recovers factors of n
    factors: tuple[int, int] | None  # (p, q) if valid factorization


@dataclass
class NearRank1Summary:
    """Summary statistics for near-rank-1 lattice point analysis."""

    n: int
    base: int
    d: int
    dx: int
    dy: int

    total_lattice_points: int
    exact_rank1_points: int  # sigma_2 = 0 exactly (integer minor test)
    valid_factorizations: int  # rank-1 AND recovers n = p * q

    # Counts at each threshold: how many points have rank_deficiency < threshold
    thresholds: list[float]
    counts_below_threshold: list[int]

    # Fraction of lattice points below each threshold
    fractions_below_threshold: list[float]

    # Log2 counts for scaling analysis
    log2_total: float
    log2_near_rank1: dict[float, float]  # threshold -> log2(count)

    # The spurious near-rank-1 points (near rank-1 but NOT valid factorizations)
    spurious_count: dict[float, int]  # threshold -> count of spurious points

    # Full rank profiles (for detailed analysis)
    rank_profiles: list[RankProfile] = field(repr=False, default_factory=list)

    # Distribution summary: min, max, mean, median of rank_deficiency
    rank_deficiency_min: float = 0.0
    rank_deficiency_max: float = 0.0
    rank_deficiency_mean: float = 0.0
    rank_deficiency_median: float = 0.0


def _svd_rank_deficiency(Z: np.ndarray) -> tuple[float, float, float]:
    """Compute rank deficiency score for a Z matrix via SVD.

    Returns (sigma_1, sigma_2, rank_deficiency) where:
    - sigma_1 = largest singular value
    - sigma_2 = second largest singular value
    - rank_deficiency = sigma_2 / sigma_1 (0 means exactly rank-1)

    For the zero matrix, returns (0, 0, 0).
    """
    if np.all(Z == 0):
        return 0.0, 0.0, 0.0

    Z_float = Z.astype(np.float64)
    sv = np.linalg.svd(Z_float, compute_uv=False)

    sigma_1 = float(sv[0]) if len(sv) > 0 else 0.0
    sigma_2 = float(sv[1]) if len(sv) > 1 else 0.0

    if sigma_1 == 0.0:
        return 0.0, 0.0, 0.0

    return sigma_1, sigma_2, sigma_2 / sigma_1


def _is_integer_rank1(Z: np.ndarray, dx: int, dy: int) -> bool:
    """Check if Z is exactly rank-1 by testing all 2x2 minors (integer arithmetic)."""
    for i1 in range(dx):
        for i2 in range(i1 + 1, dx):
            for j1 in range(dy):
                for j2 in range(j1 + 1, dy):
                    if Z[i1, j1] * Z[i2, j2] != Z[i1, j2] * Z[i2, j1]:
                        return False
    return True


def _recover_factorization(
    Z: np.ndarray, dx: int, dy: int, base: int, n: int
) -> tuple[int, int] | None:
    """Try to recover (p, q) from a rank-1 Z matrix. Returns None if not a valid factorization."""
    if np.all(Z == 0):
        return None

    max_digit = base - 1

    # Find a nonzero column to anchor the decomposition
    for j0 in range(dy):
        col = Z[:, j0]
        if np.any(col != 0):
            for yj0 in range(1, max_digit + 1):
                if not all(int(c) % yj0 == 0 for c in col):
                    continue
                x = [int(c) // yj0 for c in col]
                if not all(0 <= xi <= max_digit for xi in x):
                    continue

                y = [0] * dy
                y[j0] = yj0
                valid = True
                for j in range(dy):
                    if j == j0:
                        continue
                    col_j = Z[:, j]
                    found = False
                    for i in range(dx):
                        if x[i] != 0:
                            if int(col_j[i]) % x[i] != 0:
                                valid = False
                                break
                            y[j] = int(col_j[i]) // x[i]
                            found = True
                            break
                    if not valid:
                        break
                    if not found:
                        if np.any(col_j != 0):
                            valid = False
                            break
                        y[j] = 0

                if not valid:
                    continue
                if not all(0 <= yj <= max_digit for yj in y):
                    continue

                # Verify full decomposition
                ok = True
                for i in range(dx):
                    for j in range(dy):
                        if Z[i, j] != x[i] * y[j]:
                            ok = False
                            break
                    if not ok:
                        break

                if ok:
                    p_val = from_digits(x, base)
                    q_val = from_digits(y, base)
                    if p_val * q_val == n and p_val > 1 and q_val > 1:
                        return (min(p_val, q_val), max(p_val, q_val))
            break

    return None


def enumerate_rank_profiles(
    n: int,
    base: int,
    dx: int | None = None,
    dy: int | None = None,
) -> list[RankProfile]:
    """Enumerate ALL lattice points and compute rank profile for each.

    Uses the same digit-by-digit recursive enumeration as
    count_lattice_points_exact, but collects the SVD rank profile
    for each valid lattice point instead of just counting.

    WARNING: This is exponential in the number of digit positions.
    Only feasible for small semiprimes (d <= ~5 in base 10, ~8 in base 2).
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

    profiles: list[RankProfile] = []
    z_values: dict[tuple[int, int], int] = {}

    def recurse(k: int, carry_in: int) -> None:
        if k == d:
            if carry_in != 0:
                return

            # Build Z matrix
            Z = np.zeros((dx, dy), dtype=np.int64)
            for (i, j), v in z_values.items():
                Z[i, j] = v

            # Compute SVD rank deficiency
            sigma_1, sigma_2, rank_def = _svd_rank_deficiency(Z)

            # Check integer rank-1
            is_r1 = _is_integer_rank1(Z, dx, dy)

            # Try to recover factorization
            factors = None
            is_factorization = False
            if is_r1:
                factors = _recover_factorization(Z, dx, dy, base, n)
                is_factorization = factors is not None

            profiles.append(RankProfile(
                z_matrix=Z.copy(),
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                rank_deficiency=rank_def,
                is_exact_rank1=is_r1,
                is_valid_factorization=is_factorization,
                factors=factors,
            ))
            return

        vars_k = z_per_k[k]
        if not vars_k:
            remainder = carry_in - c[k]
            if remainder < 0 or remainder % base != 0:
                return
            t_k = remainder // base
            if t_k < 0 or (k < d - 1 and t_k > max_carry_at[k]):
                return
            recurse(k + 1, t_k)
            return

        from itertools import product as cartesian_product

        ranges = [range(0, max_z + 1) for _ in vars_k]
        for vals in cartesian_product(*ranges):
            z_sum = sum(vals)
            total_at_k = z_sum + carry_in

            remainder = total_at_k - c[k]
            if remainder < 0 or remainder % base != 0:
                continue
            t_k = remainder // base
            if t_k < 0:
                continue
            if k < d - 1 and t_k > max_carry_at[k]:
                continue

            for var, val in zip(vars_k, vals):
                z_values[var] = val
            recurse(k + 1, t_k)

        for var in vars_k:
            if var in z_values:
                del z_values[var]

    recurse(0, 0)
    return profiles


def count_near_rank1_points(
    n: int,
    base: int,
    threshold: float = 0.1,
    dx: int | None = None,
    dy: int | None = None,
) -> int:
    """Count lattice points with rank_deficiency (sigma_2/sigma_1) < threshold.

    This is the main entry point for the spurious rank-1 analysis.
    Returns the number of lattice points whose Z matrix has
    sigma_2/sigma_1 < threshold.

    Note: exact rank-1 points (the two valid factorizations) are included
    in this count. Use analyze_near_rank1 to separate genuine from spurious.
    """
    profiles = enumerate_rank_profiles(n, base, dx, dy)
    return sum(1 for p in profiles if p.rank_deficiency < threshold)


def analyze_near_rank1(
    n: int,
    base: int,
    thresholds: list[float] | None = None,
    dx: int | None = None,
    dy: int | None = None,
    keep_profiles: bool = True,
) -> NearRank1Summary:
    """Full analysis of near-rank-1 lattice points for a semiprime n.

    Enumerates all lattice points, computes SVD rank profiles, and
    produces a summary with counts at multiple thresholds.

    Args:
        n: The semiprime to analyze.
        base: The number base for digit representation.
        thresholds: List of rank_deficiency thresholds to test.
            Defaults to [0.1, 0.01, 0.001, 0.0001].
        dx, dy: Digit sizes (auto-computed if None).
        keep_profiles: If True, include full rank profiles in result.

    Returns:
        NearRank1Summary with complete analysis.
    """
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)

    profiles = enumerate_rank_profiles(n, base, dx, dy)
    total = len(profiles)

    if total == 0:
        _, ddx, ddy = _compute_digit_sizes(n, base)
        c = to_digits(n, base)
        return NearRank1Summary(
            n=n,
            base=base,
            d=len(c),
            dx=dx or ddx,
            dy=dy or ddy,
            total_lattice_points=0,
            exact_rank1_points=0,
            valid_factorizations=0,
            thresholds=thresholds,
            counts_below_threshold=[0] * len(thresholds),
            fractions_below_threshold=[0.0] * len(thresholds),
            log2_total=0.0,
            log2_near_rank1={t: 0.0 for t in thresholds},
            spurious_count={t: 0 for t in thresholds},
            rank_profiles=profiles if keep_profiles else [],
        )

    # Count exact rank-1 and valid factorizations
    exact_r1 = sum(1 for p in profiles if p.is_exact_rank1)
    valid_facts = sum(1 for p in profiles if p.is_valid_factorization)

    # Count at each threshold
    counts = []
    fractions = []
    log2_near: dict[float, float] = {}
    spurious: dict[float, int] = {}

    for t in thresholds:
        below = sum(1 for p in profiles if p.rank_deficiency < t)
        counts.append(below)
        fractions.append(below / total)
        log2_near[t] = log2(below) if below > 0 else 0.0
        # Spurious = below threshold but NOT a valid factorization
        spur = sum(
            1 for p in profiles
            if p.rank_deficiency < t and not p.is_valid_factorization
        )
        spurious[t] = spur

    # Rank deficiency distribution stats
    deficiencies = [p.rank_deficiency for p in profiles]
    sorted_def = sorted(deficiencies)

    c = to_digits(n, base)
    d_actual = len(c)
    _, dx_actual, dy_actual = _compute_digit_sizes(n, base)

    summary = NearRank1Summary(
        n=n,
        base=base,
        d=d_actual,
        dx=dx if dx is not None else dx_actual,
        dy=dy if dy is not None else dy_actual,
        total_lattice_points=total,
        exact_rank1_points=exact_r1,
        valid_factorizations=valid_facts,
        thresholds=thresholds,
        counts_below_threshold=counts,
        fractions_below_threshold=fractions,
        log2_total=log2(total) if total > 0 else 0.0,
        log2_near_rank1=log2_near,
        spurious_count=spurious,
        rank_profiles=profiles if keep_profiles else [],
        rank_deficiency_min=sorted_def[0],
        rank_deficiency_max=sorted_def[-1],
        rank_deficiency_mean=float(np.mean(deficiencies)),
        rank_deficiency_median=float(np.median(deficiencies)),
    )

    return summary


def print_summary_table(
    semiprimes: list[int] | None = None,
    base: int = 2,
    thresholds: list[float] | None = None,
) -> None:
    """Print a formatted summary table of near-rank-1 analysis.

    Args:
        semiprimes: List of semiprimes to analyze. Defaults to
            [15, 21, 35, 77, 143, 323].
        base: Number base for digit representation.
        thresholds: Rank deficiency thresholds. Defaults to [0.1, 0.01, 0.001].
    """
    if semiprimes is None:
        semiprimes = [15, 21, 35, 77, 143, 323]
    if thresholds is None:
        thresholds = [0.1, 0.01, 0.001]

    # Header
    thresh_cols = "  ".join(f"<{t:8.4f}" for t in thresholds)
    header = (
        f"{'n':>5}  {'base':>4}  {'d':>2}  {'total':>10}  {'rank1':>5}  "
        f"{'facts':>5}  {thresh_cols}  {'spurious(<0.1)':>14}  "
        f"{'median_rd':>10}  {'log2(tot)':>10}"
    )
    print("=" * len(header))
    print("Spurious Near-Rank-1 Lattice Point Analysis")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for n in semiprimes:
        try:
            result = analyze_near_rank1(n, base, thresholds, keep_profiles=False)
        except Exception as e:
            print(f"{n:>5}  base={base}: ERROR - {e}")
            continue

        counts_str = "  ".join(
            f"{c:>8d}" for c in result.counts_below_threshold
        )
        spur_01 = result.spurious_count.get(thresholds[0], 0) if thresholds else 0
        print(
            f"{result.n:>5}  {result.base:>4}  {result.d:>2}  "
            f"{result.total_lattice_points:>10}  {result.exact_rank1_points:>5}  "
            f"{result.valid_factorizations:>5}  {counts_str}  "
            f"{spur_01:>14}  "
            f"{result.rank_deficiency_median:>10.6f}  "
            f"{result.log2_total:>10.2f}"
        )

    print("-" * len(header))
    print()
    print("Legend:")
    print("  total     = total lattice points (carry-constraint solutions)")
    print("  rank1     = exactly rank-1 (all 2x2 minors vanish)")
    print("  facts     = valid factorizations (rank-1 AND p*q = n)")
    print("  <threshold = points with sigma_2/sigma_1 < threshold")
    print("  spurious  = near-rank-1 but NOT a valid factorization")
    print("  median_rd = median rank deficiency across all points")
    print("  log2(tot) = log_2 of total lattice points")


if __name__ == "__main__":
    print_summary_table(base=2)
    print()
    print_summary_table(base=3)
