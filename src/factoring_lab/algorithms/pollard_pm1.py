"""Pollard's p-1 factoring algorithm."""

from __future__ import annotations

import math

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext


def _small_primes(limit: int) -> list[int]:
    """Simple sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, math.isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]


class PollardPM1(FactoringAlgorithm):
    """Pollard's p-1 method (stage 1 only).

    Succeeds when p-1 is B-smooth for some prime factor p of n.
    """

    def __init__(self, bound: int = 100_000) -> None:
        self._bound = bound

    @property
    def name(self) -> str:
        return "pollard_pm1"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        primes = _small_primes(self._bound)
        a = 2

        for p in primes:
            ctx.record_iteration()
            # Raise a to p^e where p^e <= bound
            pe = p
            while pe * p <= self._bound:
                pe *= p
            # a = pow(a, pe, n) -- modular exponentiation
            # Count approximate modular multiplies: ~log2(pe) squarings
            ctx.record_mod_mul(pe.bit_length())
            a = pow(a, pe, n)

            # Periodic gcd checks
            if ctx.iteration_count % 50 == 0:
                ctx.record_gcd()
                g = math.gcd(a - 1, n)
                if g == n:
                    return None, "gcd = n (try smaller bound or different base)"
                if g > 1:
                    return g, f"found via p-1 at prime {p}"

        # Final gcd
        ctx.record_gcd()
        g = math.gcd(a - 1, n)
        if 1 < g < n:
            return g, f"found via p-1 at final gcd, bound={self._bound}"
        if g == n:
            return None, "gcd = n (try smaller bound)"
        return None, f"no factor with bound={self._bound}"
