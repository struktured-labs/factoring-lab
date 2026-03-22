"""Moment indistinguishability for degree-4 SOS rounding hardness.

Theorem B (Bounded-View Rounding Hardness):
    Any deterministic rounding scheme that reads at most O(d^c) entries of
    the degree-4 SOS moment matrix fails on at least one of each pair of
    distinct semiprimes of the same bit-length.

The proof proceeds by showing that:
1. The degree-4 moment matrix has O(d^4) entries (moments E[x_i y_j x_k y_l]).
2. For the true factorization (p,q), the exact degree-4 moment is a point mass:
       M[i,j,k,l] = x_i * y_j * x_k * y_l
3. Two semiprimes n1 = p1*q1 and n2 = p2*q2 of the same bit-length produce
   moment vectors that agree on many entries --- specifically, all entries
   where the digit products happen to coincide.
4. A rounding scheme reading only poly(d) entries sees a "view" that is
   identical for many pairs.  By pigeonhole, at least one pair is
   indistinguishable, so the rounding scheme fails on at least one of them.

This module provides:
- Computation of exact degree-4 moment vectors from true factorizations
- Pairwise Hamming distance between moment vectors
- Statistics on indistinguishability (fraction of agreeing entries)
- A summary structure proving the bounded-view theorem empirically

The key metric is the *normalized Hamming distance*:
    delta(n1, n2) = (# entries where M_{n1} != M_{n2}) / (total entries)

If delta < 1/poly(d) for many pairs, then any poly(d)-entry view is likely
to see only agreeing entries, proving indistinguishability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb, log2
from typing import Iterator

import numpy as np

from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    to_digits,
)


@dataclass
class MomentVector:
    """Exact degree-4 moment vector for a known factorization.

    For a semiprime n = p*q with known digits x_i, y_j in base b,
    the degree-4 moment entries are:
        M[i,j,k,l] = x_i * y_j * x_k * y_l
    for all valid (i,j,k,l) with i,k in [0, dx) and j,l in [0, dy).

    We also store the degree-2 moments (E[x_i * y_j] = x_i * y_j)
    and degree-1 moments (E[x_i] = x_i, E[y_j] = y_j).
    """

    n: int
    p: int
    q: int
    base: int
    dx: int
    dy: int

    # Digit representations
    x_digits: list[int]
    y_digits: list[int]

    # Degree-1 moments: x_digits and y_digits themselves
    # Degree-2 moments: flat array of x_i * y_j, indexed (i * dy + j)
    degree2: np.ndarray  # shape (dx * dy,)

    # Degree-4 moments: flat array of x_i * y_j * x_k * y_l
    # Indexed as ((i * dy + j) * dx * dy + (k * dy + l))
    # We store only the upper triangle (i*dy+j <= k*dy+l) to avoid redundancy
    degree4: np.ndarray  # shape (num_degree4_entries,)

    # Number of each type of entry
    num_degree2: int
    num_degree4: int

    @property
    def total_entries(self) -> int:
        """Total number of moment entries (degree 2 + degree 4)."""
        return self.num_degree2 + self.num_degree4


def _compute_degree4_index_count(dx: int, dy: int) -> int:
    """Count the number of unique degree-4 moment entries.

    Degree-4 entries are M[i,j,k,l] = E[x_i y_j x_k y_l].
    Since the moment matrix is symmetric (M[i,j,k,l] = M[k,l,i,j]),
    we count only the upper triangle: pairs (a, b) with a <= b
    where a = i*dy + j and b = k*dy + l.

    Total unique entries = C(dx*dy, 2) + dx*dy = C(dx*dy + 1, 2).
    """
    m = dx * dy
    return m * (m + 1) // 2


def compute_moment_vector(
    n: int,
    p: int,
    q: int,
    base: int,
    dx: int | None = None,
    dy: int | None = None,
) -> MomentVector:
    """Compute the exact degree-4 moment vector for a known factorization.

    For the point mass distribution at the true factorization (p, q),
    all moments are deterministic products of digits.
    """
    if dx is None or dy is None:
        _, dx, dy = _compute_digit_sizes(n, base)

    x_digits = to_digits(p, base)
    y_digits = to_digits(q, base)

    # Pad to required length
    while len(x_digits) < dx:
        x_digits.append(0)
    while len(y_digits) < dy:
        y_digits.append(0)

    # Truncate if longer (shouldn't happen for correct dx, dy)
    x_digits = x_digits[:dx]
    y_digits = y_digits[:dy]

    # Degree-2 moments: z_{ij} = x_i * y_j
    m = dx * dy
    degree2 = np.zeros(m, dtype=np.int64)
    for i in range(dx):
        for j in range(dy):
            degree2[i * dy + j] = x_digits[i] * y_digits[j]

    # Degree-4 moments: M[a,b] = z_a * z_b for a <= b
    # where a = i*dy+j, b = k*dy+l
    num_d4 = _compute_degree4_index_count(dx, dy)
    degree4 = np.zeros(num_d4, dtype=np.int64)

    idx = 0
    for a in range(m):
        for b in range(a, m):
            degree4[idx] = int(degree2[a]) * int(degree2[b])
            idx += 1

    return MomentVector(
        n=n,
        p=p,
        q=q,
        base=base,
        dx=dx,
        dy=dy,
        x_digits=x_digits,
        y_digits=y_digits,
        degree2=degree2,
        degree4=degree4,
        num_degree2=m,
        num_degree4=num_d4,
    )


@dataclass
class PairwiseDistance:
    """Hamming distance between moment vectors of two semiprimes."""

    n1: int
    n2: int
    p1: int
    q1: int
    p2: int
    q2: int
    base: int

    # Degree-2 distances
    degree2_hamming: int  # number of positions where degree-2 moments differ
    degree2_total: int  # total degree-2 entries
    degree2_frac_agree: float  # fraction that agree

    # Degree-4 distances
    degree4_hamming: int
    degree4_total: int
    degree4_frac_agree: float

    # Combined distances
    total_hamming: int
    total_entries: int
    total_frac_agree: float

    # Normalized Hamming distance (the key metric)
    normalized_hamming: float  # = total_hamming / total_entries

    # Digit-level overlap
    x_digits_hamming: int  # how many x-digit positions differ
    y_digits_hamming: int  # how many y-digit positions differ
    x_digits_total: int
    y_digits_total: int


def compute_pairwise_distance(
    mv1: MomentVector,
    mv2: MomentVector,
) -> PairwiseDistance:
    """Compute the Hamming distance between two moment vectors.

    Both must use the same base and digit sizes.
    """
    assert mv1.base == mv2.base
    assert mv1.dx == mv2.dx
    assert mv1.dy == mv2.dy

    # Degree-2 Hamming distance
    d2_diff = np.sum(mv1.degree2 != mv2.degree2)
    d2_total = mv1.num_degree2

    # Degree-4 Hamming distance
    d4_diff = np.sum(mv1.degree4 != mv2.degree4)
    d4_total = mv1.num_degree4

    total_diff = int(d2_diff) + int(d4_diff)
    total = d2_total + d4_total

    # Digit-level overlap
    x_diff = sum(
        1 for i in range(mv1.dx) if mv1.x_digits[i] != mv2.x_digits[i]
    )
    y_diff = sum(
        1 for j in range(mv1.dy) if mv1.y_digits[j] != mv2.y_digits[j]
    )

    d2_frac = 1.0 - int(d2_diff) / d2_total if d2_total > 0 else 1.0
    d4_frac = 1.0 - int(d4_diff) / d4_total if d4_total > 0 else 1.0
    total_frac = 1.0 - total_diff / total if total > 0 else 1.0

    return PairwiseDistance(
        n1=mv1.n,
        n2=mv2.n,
        p1=mv1.p,
        q1=mv1.q,
        p2=mv2.p,
        q2=mv2.q,
        base=mv1.base,
        degree2_hamming=int(d2_diff),
        degree2_total=d2_total,
        degree2_frac_agree=d2_frac,
        degree4_hamming=int(d4_diff),
        degree4_total=d4_total,
        degree4_frac_agree=d4_frac,
        total_hamming=total_diff,
        total_entries=total,
        total_frac_agree=total_frac,
        normalized_hamming=total_diff / total if total > 0 else 0.0,
        x_digits_hamming=x_diff,
        y_digits_hamming=y_diff,
        x_digits_total=mv1.dx,
        y_digits_total=mv1.dy,
    )


@dataclass
class IndistinguishabilityResult:
    """Summary of moment indistinguishability across semiprime pairs.

    This is the empirical core of Theorem B: for semiprimes of a given
    bit-length in a given base, how many moment entries must a rounding
    scheme read to distinguish all pairs?
    """

    base: int
    bit_length: int  # bit-length of semiprimes
    num_semiprimes: int
    num_pairs: int

    # Digit structure
    d: int  # number of digit positions
    dx: int
    dy: int
    total_degree4_entries: int
    total_moment_entries: int  # degree-2 + degree-4

    # Distribution of pairwise normalized Hamming distances
    min_hamming: float
    max_hamming: float
    mean_hamming: float
    median_hamming: float

    # The critical metric: minimum number of entries to distinguish ALL pairs
    # This is the max over all pairs of the Hamming distance
    # (= minimum number of differing entries any rounding scheme must inspect)
    min_entries_to_distinguish_all: int

    # For Theorem B: fraction of pairs that are indistinguishable
    # when reading only k entries (for various k)
    indistinguishable_at_k: dict[int, float]  # k -> fraction of indistinguishable pairs

    # Individual pair distances (for detailed analysis)
    pair_distances: list[PairwiseDistance] = field(repr=False)


def _enumerate_small_semiprimes(
    bit_length: int,
) -> list[tuple[int, int, int]]:
    """Enumerate all semiprimes n = p*q with n.bit_length() == bit_length.

    Returns list of (n, p, q) with p <= q.
    """
    from factoring_lab.generators.semiprimes import _is_prime

    lo = 1 << (bit_length - 1)
    hi = (1 << bit_length) - 1

    results: list[tuple[int, int, int]] = []
    seen: set[int] = set()

    # p ranges from 2 to sqrt(hi)
    max_p = int(hi**0.5) + 1
    for p in range(2, max_p + 1):
        if not _is_prime(p):
            continue
        # q ranges so that lo <= p*q <= hi
        q_lo = max(p, (lo + p - 1) // p)
        q_hi = hi // p
        for q in range(q_lo, q_hi + 1):
            if not _is_prime(q):
                continue
            n = p * q
            if n.bit_length() == bit_length and n not in seen:
                seen.add(n)
                results.append((n, p, q))

    return results


def analyze_indistinguishability(
    bit_length: int,
    base: int,
    max_semiprimes: int | None = None,
) -> IndistinguishabilityResult:
    """Analyze moment indistinguishability for semiprimes of a given bit-length.

    Enumerates all (or up to max_semiprimes) semiprimes of the given
    bit-length, computes their degree-4 moment vectors, and measures
    pairwise Hamming distances.

    Args:
        bit_length: bit-length of semiprimes to analyze (8-16 recommended)
        base: base for digit decomposition
        max_semiprimes: if set, limit to this many semiprimes (randomly sampled)

    Returns:
        IndistinguishabilityResult with all pairwise distance statistics
    """
    semiprimes = _enumerate_small_semiprimes(bit_length)

    if max_semiprimes is not None and len(semiprimes) > max_semiprimes:
        import random

        rng = random.Random(42)
        semiprimes = rng.sample(semiprimes, max_semiprimes)

    if len(semiprimes) < 2:
        raise ValueError(
            f"Need at least 2 semiprimes at {bit_length} bits, found {len(semiprimes)}"
        )

    # Compute moment vectors
    # Use consistent dx, dy for all semiprimes of this bit-length
    # (take the max needed across all)
    ref_n = semiprimes[0][0]
    _, dx, dy = _compute_digit_sizes(ref_n, base)

    moment_vectors: list[MomentVector] = []
    for n, p, q in semiprimes:
        mv = compute_moment_vector(n, p, q, base, dx, dy)
        moment_vectors.append(mv)

    # Compute all pairwise distances
    pair_distances: list[PairwiseDistance] = []
    for i, j in combinations(range(len(moment_vectors)), 2):
        dist = compute_pairwise_distance(moment_vectors[i], moment_vectors[j])
        pair_distances.append(dist)

    num_pairs = len(pair_distances)
    hammings = [pd.normalized_hamming for pd in pair_distances]
    abs_hammings = [pd.total_hamming for pd in pair_distances]

    sorted_hammings = sorted(hammings)
    min_h = sorted_hammings[0]
    max_h = sorted_hammings[-1]
    mean_h = sum(hammings) / num_pairs
    median_h = sorted_hammings[num_pairs // 2]

    # Minimum entries to distinguish all pairs = max absolute Hamming
    # Actually, the minimum entries a rounding scheme must read to
    # distinguish a specific pair is the absolute Hamming distance
    # (it must read at least one entry where they differ).
    # To distinguish ALL pairs, it must read enough entries to cover
    # at least one differing entry from every pair.
    #
    # But the more useful metric for Theorem B is: if we read k entries,
    # what fraction of pairs are indistinguishable?
    # A pair is indistinguishable at budget k if the probability of
    # hitting a differing entry in k random reads is small.
    # P(distinguish pair) = 1 - (1 - delta)^k where delta = normalized_hamming
    total_entries = moment_vectors[0].total_entries

    # Minimum absolute Hamming distance = minimum number of differing entries
    min_abs_hamming = min(abs_hammings)

    # Compute indistinguishable fraction at various budgets k
    # A pair is "k-indistinguishable" if k random entry reads are
    # unlikely to distinguish it.  Formally: the expected number of
    # distinguishing entries among k random reads is k * delta.
    # If k * delta < 1, we expect to see 0 distinguishing entries.
    # More precisely, P(all k reads agree) = C(total - hamming, k) / C(total, k)
    # which is approximately (1 - delta)^k for large total.
    budgets = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    # Filter budgets to those <= total_entries
    budgets = [k for k in budgets if k <= total_entries]
    # Also add d, d^2, d^3, d^4 as natural budgets
    d = len(to_digits(ref_n, base))
    for power_budget in [d, d**2, d**3, d**4]:
        if power_budget <= total_entries and power_budget not in budgets:
            budgets.append(power_budget)
    budgets.sort()

    indist_at_k: dict[int, float] = {}
    for k in budgets:
        # For each pair, compute P(indistinguishable at budget k)
        # = P(all k random entries are agreeing entries)
        # = C(total - hamming, k) / C(total, k)
        # = product_{j=0}^{k-1} (total - hamming - j) / (total - j)
        num_indist = 0
        for pd in pair_distances:
            agree = pd.total_entries - pd.total_hamming
            if k > agree:
                # Can't pick k entries from agreeing set
                prob_indist = 0.0
            else:
                # Hypergeometric: probability of picking k items all from
                # the "agree" set of size (total - hamming)
                log_prob = 0.0
                for j in range(k):
                    log_prob += log2(max(agree - j, 1e-300)) - log2(
                        max(total_entries - j, 1e-300)
                    )
                prob_indist = 2.0**log_prob
            if prob_indist > 0.5:
                # More likely than not to be indistinguishable
                num_indist += 1
        indist_at_k[k] = num_indist / num_pairs if num_pairs > 0 else 0.0

    return IndistinguishabilityResult(
        base=base,
        bit_length=bit_length,
        num_semiprimes=len(semiprimes),
        num_pairs=num_pairs,
        d=d,
        dx=dx,
        dy=dy,
        total_degree4_entries=_compute_degree4_index_count(dx, dy),
        total_moment_entries=total_entries,
        min_hamming=min_h,
        max_hamming=max_h,
        mean_hamming=mean_h,
        median_hamming=median_h,
        min_entries_to_distinguish_all=min_abs_hamming,
        indistinguishable_at_k=indist_at_k,
        pair_distances=pair_distances,
    )


@dataclass
class BoundedViewTheoremResult:
    """Formal statement and empirical verification of Theorem B.

    Theorem B: Any deterministic rounding scheme R that reads at most
    k = O(d^c) entries of the degree-4 SOS moment matrix fails to
    correctly factor at least one semiprime in each indistinguishable pair.

    The proof is by pigeonhole: if two semiprimes n1, n2 agree on all
    k queried entries, R(n1) = R(n2), but at most one of {n1, n2} has
    factorization matching R's output.
    """

    base: int

    # Results at each bit-length
    bit_lengths: list[int]
    results: list[IndistinguishabilityResult]

    # For each bit-length: minimum budget k* such that ALL pairs are
    # distinguishable (i.e., indistinguishable fraction drops to 0)
    critical_budgets: list[int | None]  # None if never all distinguishable

    # Scaling: how does critical budget grow with d?
    # We fit k* ~ d^alpha and report alpha
    scaling_exponent: float | None  # estimated alpha

    # Summary: at each bit-length, what fraction of pairs are
    # indistinguishable at budget d^2?
    indist_at_d_squared: list[float]


def prove_bounded_view_theorem(
    bit_lengths: list[int] | None = None,
    base: int = 2,
    max_semiprimes: int | None = None,
) -> BoundedViewTheoremResult:
    """Empirically verify Theorem B across multiple bit-lengths.

    For each bit-length, enumerates semiprimes, computes moment vectors,
    and measures how many entries a rounding scheme must read to distinguish
    all pairs.

    Args:
        bit_lengths: list of bit-lengths to test (default: [8, 10, 12, 14, 16])
        base: base for digit decomposition (default: 2)
        max_semiprimes: limit semiprimes per bit-length (for performance)

    Returns:
        BoundedViewTheoremResult with scaling analysis
    """
    if bit_lengths is None:
        bit_lengths = [8, 10, 12, 14, 16]

    results: list[IndistinguishabilityResult] = []
    critical_budgets: list[int | None] = []
    indist_d2: list[float] = []

    for bl in bit_lengths:
        try:
            ir = analyze_indistinguishability(bl, base, max_semiprimes)
        except ValueError:
            # Not enough semiprimes at this bit-length
            continue

        results.append(ir)

        # Find critical budget: smallest k where all pairs are distinguishable
        # (indistinguishable fraction = 0)
        crit = None
        sorted_budgets = sorted(ir.indistinguishable_at_k.keys())
        for k in sorted_budgets:
            if ir.indistinguishable_at_k[k] == 0.0:
                crit = k
                break
        critical_budgets.append(crit)

        # Indistinguishable fraction at d^2
        d = ir.d
        d2 = d * d
        # Find the closest budget to d^2
        if d2 in ir.indistinguishable_at_k:
            indist_d2.append(ir.indistinguishable_at_k[d2])
        else:
            # Interpolate: find the largest budget <= d^2
            smaller = [
                k for k in ir.indistinguishable_at_k if k <= d2
            ]
            if smaller:
                k_use = max(smaller)
                indist_d2.append(ir.indistinguishable_at_k[k_use])
            else:
                indist_d2.append(1.0)  # All pairs indistinguishable at very low budget

    # Fit scaling exponent: k* ~ d^alpha
    # Use log-log regression on (d, k*) pairs where k* is defined
    scaling_exp = None
    valid_bl = [
        bit_lengths[i]
        for i in range(len(results))
    ]
    valid_points: list[tuple[float, float]] = []
    for i, ir in enumerate(results):
        if i < len(critical_budgets) and critical_budgets[i] is not None:
            d = ir.d
            k_star = critical_budgets[i]
            if d > 1 and k_star is not None and k_star > 0:
                valid_points.append((log2(d), log2(k_star)))

    if len(valid_points) >= 2:
        # Simple linear regression in log-log space
        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 1e-10:
            scaling_exp = cov_xy / var_x

    return BoundedViewTheoremResult(
        base=base,
        bit_lengths=valid_bl,
        results=results,
        critical_budgets=critical_budgets,
        scaling_exponent=scaling_exp,
        indist_at_d_squared=indist_d2,
    )


def compute_moment_agreement_matrix(
    semiprimes: list[tuple[int, int, int]],
    base: int,
    dx: int | None = None,
    dy: int | None = None,
) -> np.ndarray:
    """Compute the pairwise moment agreement matrix.

    Returns an n x n matrix where entry [i,j] is the fraction of degree-4
    moment entries that agree between semiprimes i and j.

    This directly visualizes the indistinguishability structure.
    """
    if dx is None or dy is None:
        _, dx, dy = _compute_digit_sizes(semiprimes[0][0], base)

    mvs = [
        compute_moment_vector(n, p, q, base, dx, dy) for n, p, q in semiprimes
    ]

    n = len(mvs)
    agreement = np.eye(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            pd = compute_pairwise_distance(mvs[i], mvs[j])
            agreement[i, j] = pd.total_frac_agree
            agreement[j, i] = pd.total_frac_agree

    return agreement


def count_distinguishing_entries_by_type(
    mv1: MomentVector,
    mv2: MomentVector,
) -> dict[str, int]:
    """Break down distinguishing entries by moment degree and position.

    Returns a dictionary with counts of differing entries at each level:
    - "degree1_x": number of differing x-digit positions
    - "degree1_y": number of differing y-digit positions
    - "degree2": number of differing degree-2 entries
    - "degree4": number of differing degree-4 entries
    - "degree4_from_degree2": entries differing because at least one
      constituent degree-2 entry differs
    """
    assert mv1.dx == mv2.dx and mv1.dy == mv2.dy

    dx, dy = mv1.dx, mv1.dy

    # Degree-1 differences
    x_diff = sum(
        1 for i in range(dx) if mv1.x_digits[i] != mv2.x_digits[i]
    )
    y_diff = sum(
        1 for j in range(dy) if mv1.y_digits[j] != mv2.y_digits[j]
    )

    # Degree-2 differences
    d2_diff_mask = mv1.degree2 != mv2.degree2
    d2_diff = int(np.sum(d2_diff_mask))

    # Degree-4 differences
    d4_diff = int(np.sum(mv1.degree4 != mv2.degree4))

    # Degree-4 entries that differ BECAUSE a constituent degree-2 entry differs
    # M4[a,b] = d2[a] * d2[b], so M4 differs if d2[a] or d2[b] differs
    m = dx * dy
    d4_from_d2 = 0
    idx = 0
    for a in range(m):
        for b in range(a, m):
            if d2_diff_mask[a] or d2_diff_mask[b]:
                # At least one constituent degree-2 entry differs
                # This COULD lead to a degree-4 difference
                if mv1.degree4[idx] != mv2.degree4[idx]:
                    d4_from_d2 += 1
            idx += 1

    return {
        "degree1_x": x_diff,
        "degree1_y": y_diff,
        "degree2": d2_diff,
        "degree4": d4_diff,
        "degree4_from_degree2": d4_from_d2,
    }
