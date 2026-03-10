"""Tools for studying prime gap statistics.

Research questions addressed:
1. Average gap size for primes with b to b+1 bits (PNT estimate: ~b*ln(2))
2. Empirical gap distribution vs the conjectured Exponential(1) after normalization
3. Finding primes p where no prime pair q,r satisfies p = q + r (Goldbach-related)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from factoring_lab.generators.semiprimes import _is_prime


def _next_prime(n: int) -> int:
    """Return the smallest prime strictly greater than n."""
    if n < 2:
        return 2
    candidate = n + 1 if n % 2 == 0 else n + 2
    # Ensure candidate is odd
    if candidate % 2 == 0:
        candidate += 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate


def _prev_prime(n: int) -> int | None:
    """Return the largest prime strictly less than n, or None if none exists."""
    if n <= 2:
        return None
    if n == 3:
        return 2
    candidate = n - 1 if n % 2 == 0 else n - 2
    if candidate == 2:
        return 2
    if candidate % 2 == 0:
        candidate -= 1
    while candidate >= 2 and not _is_prime(candidate):
        candidate -= 2
    if candidate < 2:
        return None
    return candidate


def _primes_in_range(low: int, high: int) -> list[int]:
    """Return all primes p with low <= p <= high."""
    if high < 2:
        return []
    primes: list[int] = []
    if low <= 2:
        primes.append(2)
    start = max(low, 3)
    if start % 2 == 0:
        start += 1
    for candidate in range(start, high + 1, 2):
        if _is_prime(candidate):
            primes.append(candidate)
    return primes


def prime_gaps_in_range(low: int, high: int) -> list[int]:
    """Return gaps between consecutive primes in [low, high].

    A gap is defined as p_{k+1} - p_k for consecutive primes p_k, p_{k+1}
    both within [low, high].

    Args:
        low: lower bound (inclusive)
        high: upper bound (inclusive)

    Returns:
        List of gap sizes. Length is len(primes_in_range) - 1.
    """
    primes = _primes_in_range(low, high)
    return [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]


def average_gap_estimate(bits: int) -> float:
    """Theoretical average gap for primes near 2^bits using the Prime Number Theorem.

    By PNT, the density of primes near x is ~1/ln(x), so the average gap
    near x = 2^bits is ln(2^bits) = bits * ln(2).

    Args:
        bits: bit size of primes to consider

    Returns:
        Estimated average gap size (float).
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    return bits * math.log(2)


def empirical_gap_stats(bits: int, count: int = 1000) -> dict[str, float]:
    """Compute statistics of actual prime gaps near 2^bits.

    Finds `count` consecutive primes starting near 2^bits and computes
    gap statistics.

    Args:
        bits: bit size region to sample (starts searching from 2^bits)
        count: number of consecutive primes to find (gaps = count - 1)

    Returns:
        Dictionary with keys: mean, median, max, min, std, pnt_estimate
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")

    # Start from 2^bits and find consecutive primes
    current = (1 << bits) - 1
    if current < 2:
        current = 2

    primes: list[int] = []
    # Find first prime >= current
    if _is_prime(current):
        primes.append(current)
    else:
        primes.append(_next_prime(current))

    while len(primes) < count:
        primes.append(_next_prime(primes[-1]))

    gaps = [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]

    pnt_est = average_gap_estimate(bits)

    return {
        "mean": statistics.mean(gaps),
        "median": statistics.median(gaps),
        "max": float(max(gaps)),
        "min": float(min(gaps)),
        "std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
        "pnt_estimate": pnt_est,
        "count": float(len(gaps)),
    }


def goldbach_check(limit: int) -> list[int]:
    """Find primes p < limit that cannot be expressed as q + r for primes q, r.

    More precisely, checks whether each odd prime p < limit can be written as
    p = q + r where q and r are both prime (with q <= r). This is related to
    Goldbach's conjecture (every even number > 2 is the sum of two primes)
    applied to odd primes: an odd prime p = 2 + (p-2), so p is expressible
    as a sum of two primes iff p-2 is prime.

    Also checks p = q + r + 1 variant: p can be written as q + r + 1 for
    primes q, r iff p - 1 = q + r, i.e., the even number p-1 is a sum of
    two primes.

    Args:
        limit: upper bound (exclusive) for primes to check

    Returns:
        List of primes p < limit where p cannot be written as q + r
        AND p cannot be written as q + r + 1 for any primes q, r.
        Expected to be empty for all tested ranges (possibly containing
        only very small primes like 2 or 3).
    """
    if limit < 2:
        return []

    primes = _primes_in_range(2, limit - 1)
    prime_set = set(primes)
    failures: list[int] = []

    for p in primes:
        # Check q + r = p: need primes q <= r with q + r = p
        # q ranges from 2 to p//2
        found_sum = False
        for q in primes:
            if q > p // 2:
                break
            r = p - q
            if r in prime_set:
                found_sum = True
                break

        if found_sum:
            continue

        # Check q + r + 1 = p, i.e., q + r = p - 1
        target = p - 1
        found_sum_plus1 = False
        for q in primes:
            if q > target // 2:
                break
            r = target - q
            if r in prime_set:
                found_sum_plus1 = True
                break

        if not found_sum_plus1:
            failures.append(p)

    return failures
