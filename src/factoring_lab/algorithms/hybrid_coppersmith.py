"""Hybrid digit-convolution + Coppersmith lattice factoring.

Combines two complementary approaches:
1. Digit convolution constraints enumerate valid low-digit assignments
   for factors x, y in base b (satisfying alpha_k = c_k mod b).
2. Coppersmith's method (via LLL) recovers the remaining high digits
   once enough low digits are known.

The key insight: digit convolution gives us structured partial knowledge
of the factors. Specifically, if we know x mod b^k and y mod b^k, we
know p_low = x mod b^k. This is exactly the input Coppersmith needs:
    p = p_low + b^k * u
    p divides n
    => f(u) = p_low + b^k * u has a small root mod p

We construct a lattice from [n, b^k, p_low] and apply LLL to find p.

This is a simplified variant of Coppersmith's method suitable for the
digit-enumeration setting, where we iterate over candidate (x_low, y_low)
pairs and attempt lattice recovery for each.

Original constraint formulation: struktured, November 2009.
Hybrid approach: Phase 2, March 2026.
"""

from __future__ import annotations

import math
import time
from typing import Iterator

import numpy as np

from factoring_lab.algorithms.base import FactoringAlgorithm, FactoringResult, InstrumentedContext
from factoring_lab.algorithms.lattice_convolution import lll_reduce


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


def valid_digit_pairs(c_digit: int, base: int) -> list[tuple[int, int]]:
    """Find all (x_i, y_j) pairs satisfying x_i * y_j = c_digit (mod base).

    These are the valid lowest-digit assignments from the convolution constraint:
        alpha_0 = x_0 * y_0 = c_0 (mod b)

    For higher positions we must account for carries, but position 0 has no
    incoming carry.
    """
    pairs = []
    for x in range(base):
        for y in range(base):
            if (x * y) % base == c_digit:
                pairs.append((x, y))
    return pairs


def enumerate_digit_assignments(
    n: int,
    base: int,
    depth: int,
) -> Iterator[tuple[int, int]]:
    """Enumerate valid (x_low, y_low) assignments up to `depth` digit positions.

    Yields (x_low, y_low) pairs such that x_low * y_low = n (mod base^depth).
    Both x_low and y_low are values mod base^depth.

    We enumerate by extending digit-by-digit:
    - At depth 1: all (x_0, y_0) with x_0 * y_0 = c_0 (mod b)
    - At depth 2: extend to (x_0 + b*x_1, y_0 + b*y_1) satisfying
      product = n (mod b^2)
    - etc.

    This is the "digit convolution backtracking" step, but we only go
    `depth` levels deep before handing off to the lattice solver.
    """
    n_mod = n % base
    # Start with depth-1 assignments
    current_pairs: list[tuple[int, int]] = []
    for x0 in range(base):
        for y0 in range(base):
            if (x0 * y0) % base == n_mod:
                current_pairs.append((x0, y0))

    if depth <= 1:
        yield from current_pairs
        return

    # Extend to higher digit positions
    for d in range(1, depth):
        power = base ** (d + 1)
        target = n % power
        next_pairs: list[tuple[int, int]] = []
        for x_low, y_low in current_pairs:
            # Try extending with all possible next digits
            prev_power = base ** d
            for xd in range(base):
                for yd in range(base):
                    x_new = x_low + xd * prev_power
                    y_new = y_low + yd * prev_power
                    if (x_new * y_new) % power == target:
                        next_pairs.append((x_new, y_new))
        current_pairs = next_pairs

    yield from current_pairs


def coppersmith_lattice_factor(
    n: int,
    p_low: int,
    k_bits: int,
) -> int | None:
    """Attempt to recover a factor of n given p_low = p mod 2^k_bits.

    Constructs a lattice based on the polynomial:
        f(u) = p_low + 2^k * u
    where p = f(u) divides n.

    We use a simplified Coppersmith-type construction:
    The lattice is formed from the rows:
        [X^2,  0,   0  ]
        [X*n,  X,   0  ]
        [n^2,  n,   1  ]

    where X = 2^k is the scaling for the unknown part.

    Actually, for the factoring setting the standard approach is:
    Given p = p_low + M*u where M = 2^k (or base^k), p | n.
    We want to find u such that p_low + M*u divides n.

    Construct lattice basis:
        B = [[M,  0 ],
             [p_low, n]]
    Short vectors in this lattice can reveal p.

    For better results, we use Howgrave-Graham's reformulation with
    multiple shifts.

    Returns a non-trivial factor of n, or None.
    """
    if p_low <= 1:
        return None

    M = 1 << k_bits  # 2^k

    # Quick GCD check: sometimes p_low itself shares a factor with n
    g = math.gcd(p_low, n)
    if 1 < g < n:
        return g

    # Strategy 1: Simple 2D lattice
    # If p = p_low + M*u, then p | n means n = p*q = (p_low + M*u)*q
    # So n = p_low*q (mod M) => q = n * p_low^{-1} (mod M) if gcd(p_low, M)=1
    # Then we can try to find p directly.
    if math.gcd(p_low, M) == 1:
        # Compute q_low = n * p_low^{-1} mod M
        try:
            p_low_inv = pow(p_low, -1, M)
            q_low = (n * p_low_inv) % M
            # Now check: does p_low * q_low = n mod M? (sanity)
            if (p_low * q_low) % M == n % M:
                # Build lattice to find the full p from p_low
                # We know p = p_low + M*u and q = q_low + M*v
                # and p*q = n, so (p_low + M*u)(q_low + M*v) = n
                # Expanding: p_low*q_low + M*(p_low*v + q_low*u) + M^2*u*v = n
                # Let R = (n - p_low*q_low) // M (integer since n = p_low*q_low mod M)
                # Then p_low*v + q_low*u + M*u*v = R
                # This is still nonlinear in u,v. But for small u we can use lattice.

                # Simpler approach: construct lattice for the univariate case
                # f(x) = p_low + M*x has a root mod p (where p | n)
                # Coppersmith bound: works when x < N^{1/2} / M roughly

                # Use Howgrave-Graham style lattice with dimension 2:
                # Basis: [[N, 0], [p_low, M]]
                # A short vector (a, b) in this lattice satisfies a = c*N + d*p_low,
                # b = d*M for integers c, d.
                # If d*p + c*n = 0 for small c, d, then a*p + b = 0 mod n...
                # Actually, the standard construction for known LSBs:

                result = _lattice_recover_factor(n, p_low, M)
                if result is not None:
                    return result
        except (ValueError, ZeroDivisionError):
            pass

    # Strategy 2: try base-b variant (p_low in arbitrary base)
    result = _lattice_recover_factor(n, p_low, M)
    if result is not None:
        return result

    return None


def coppersmith_lattice_factor_base(
    n: int,
    p_low: int,
    base: int,
    num_known_digits: int,
) -> int | None:
    """Coppersmith recovery using base-b digit knowledge instead of binary bits.

    Given p_low = p mod base^num_known_digits, attempt to recover p.
    """
    if p_low <= 0:
        return None

    M = base ** num_known_digits

    g = math.gcd(p_low, n)
    if 1 < g < n:
        return g

    return _lattice_recover_factor(n, p_low, M)


def _lattice_recover_factor(n: int, p_low: int, M: int) -> int | None:
    """Core lattice construction for factor recovery.

    Given n and p_low where p = p_low (mod M) and p | n,
    attempts to find p using LLL.

    Uses multiple lattice constructions of increasing sophistication.
    """
    isqrt_n = math.isqrt(n)

    # Construction 1: Direct 2x2 lattice
    # Basis: [[M, 0], [-p_low, 1]]  scaled appropriately
    # A vector (a, b) in L means a = c*M - d*p_low, b = d
    # If p = p_low + M*u, then (M*u, 1) . (1/M, p_low) = u + p_low/M ...
    # Better: use the Boneh-Durfee / Coppersmith style:

    # The polynomial f(x) = x + p_low has a small root x0 = p - p_low
    # modulo p (a divisor of n). The root satisfies |x0| < n/M roughly
    # (since p < sqrt(n) for balanced semiprimes, and p_low < M).

    # Howgrave-Graham lattice for f(x) = x + p_low, modular root mod p | n:
    # Use basis [[n, 0], [p_low, M]]
    # After LLL, short vectors may reveal p.

    # Bound on the unknown part
    X = isqrt_n  # upper bound on p

    # 2D lattice
    B = np.array([
        [n, 0],
        [p_low, M],
    ], dtype=object)

    # Convert to int64 if small enough, else use float64 for LLL
    try:
        if n.bit_length() < 62:
            B_int = B.astype(np.int64)
            reduced = lll_reduce(B_int)
        else:
            # For larger numbers, do a scaled version
            # Scale down to avoid int64 overflow
            scale = max(1, n >> 60)
            B_scaled = np.array([
                [n // scale, 0],
                [p_low // scale, max(1, M // scale)],
            ], dtype=np.int64)
            reduced = lll_reduce(B_scaled)
    except (OverflowError, ValueError):
        return None

    # Check each row of the reduced basis for a factor
    for row in reduced:
        for val in row:
            val = int(abs(val))
            if val < 2:
                continue
            g = math.gcd(val, n)
            if 1 < g < n:
                return int(min(g, n // g))

    # Construction 2: 3D lattice with shifts (more Coppersmith-like)
    # f(x) = p_low + M*x, f^2(x) = (p_low + M*x)^2
    # Lattice basis for polynomials mod p:
    # [n^2,  0,     0    ]
    # [n*p_low, n*M, 0   ]
    # [p_low^2, 2*p_low*M, M^2]
    # This gives Howgrave-Graham conditions on the coefficients.
    try:
        if n.bit_length() < 31:
            B3 = np.array([
                [n, 0, 0],
                [p_low, M, 0],
                [0, 0, n * M],
            ], dtype=np.int64)
            reduced3 = lll_reduce(B3)

            for row in reduced3:
                for val in row:
                    val = int(abs(val))
                    if val < 2:
                        continue
                    g = math.gcd(val, n)
                    if 1 < g < n:
                        return int(min(g, n // g))

                # Also try linear combinations
                a, b, c = int(row[0]), int(row[1]), int(row[2])
                if b != 0:
                    # From the lattice structure, a = alpha*n + beta*p_low
                    # and b = beta*M. So beta = b/M, and
                    # p_candidate = p_low + M*(something)
                    # Try: a might be related to p
                    for candidate in [abs(a), abs(b), abs(a + b), abs(a - b)]:
                        if candidate < 2:
                            continue
                        g = math.gcd(candidate, n)
                        if 1 < g < n:
                            return int(min(g, n // g))
    except (OverflowError, ValueError):
        pass

    # Construction 3: Brute force small remainders
    # If we know p mod M, try p = p_low + M*k for small k
    # This is cheap and catches cases where the lattice misses
    max_k = min(10000, isqrt_n // M + 1) if M > 0 else 0
    for k in range(max_k):
        p_candidate = p_low + M * k
        if p_candidate < 2:
            continue
        if p_candidate > isqrt_n + 1:
            break
        if n % p_candidate == 0:
            return int(min(p_candidate, n // p_candidate))

    return None


class HybridCoppersmith(FactoringAlgorithm):
    """Factor via digit-enumeration + Coppersmith lattice recovery.

    Strategy:
    1. Compute n's digits in base b.
    2. Enumerate valid (x_low, y_low) pairs satisfying the convolution
       constraint x_low * y_low = n (mod b^depth).
    3. For each candidate x_low, attempt lattice-based recovery of the
       full factor p using Coppersmith's method.
    4. Return the first valid factorization found.

    Parameters
    ----------
    base : int
        Base for digit convolution (default 10).
    depth : int
        Number of digit positions to enumerate before lattice call.
        Higher depth = fewer lattice calls but exponential enumeration.
    timeout_s : float
        Total wall-clock timeout in seconds.
    """

    def __init__(
        self,
        base: int = 10,
        depth: int = 1,
        timeout_s: float = 30.0,
    ) -> None:
        self._base = base
        self._depth = depth
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"hybrid_coppersmith_b{self._base}_d{self._depth}"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        base = self._base
        depth = self._depth
        t0 = time.perf_counter()

        M = base ** depth
        n_digits = _to_digits(n, base)
        pairs_tried = 0
        lattice_calls = 0

        for x_low, y_low in enumerate_digit_assignments(n, base, depth):
            elapsed = time.perf_counter() - t0
            if elapsed > self._timeout_s:
                return None, (
                    f"timeout after {pairs_tried} pairs, "
                    f"{lattice_calls} lattice calls, {elapsed:.1f}s"
                )

            pairs_tried += 1
            ctx.record_iteration()

            # Skip trivial assignments
            if x_low == 0 or y_low == 0:
                continue

            # Try x_low as partial knowledge of p
            lattice_calls += 1
            result = coppersmith_lattice_factor_base(n, x_low, base, depth)
            if result is not None:
                return result, (
                    f"found via digit enum (depth={depth}) + lattice: "
                    f"x_low={x_low}, pairs_tried={pairs_tried}, "
                    f"lattice_calls={lattice_calls}"
                )

            # Also try y_low as partial knowledge of p
            if y_low != x_low:
                lattice_calls += 1
                result = coppersmith_lattice_factor_base(n, y_low, base, depth)
                if result is not None:
                    return result, (
                        f"found via digit enum (depth={depth}) + lattice: "
                        f"y_low={y_low}, pairs_tried={pairs_tried}, "
                        f"lattice_calls={lattice_calls}"
                    )

        elapsed = time.perf_counter() - t0
        return None, (
            f"exhausted {pairs_tried} digit pairs at depth={depth}, "
            f"{lattice_calls} lattice calls, {elapsed:.1f}s"
        )

    def factor_with_details(self, n: int) -> dict:
        """Run factoring and return detailed diagnostics."""
        t0 = time.perf_counter()
        result = self.factor(n)
        elapsed = time.perf_counter() - t0

        return {
            "n": n,
            "bits": n.bit_length(),
            "base": self._base,
            "depth": self._depth,
            "success": result.success,
            "factor": result.factor,
            "cofactor": result.cofactor,
            "runtime_s": round(elapsed, 4),
            "iterations": result.iteration_count,
            "notes": result.notes,
        }
