"""Lenstra's Elliptic Curve Method (ECM) factoring algorithm.

Uses Montgomery curves By^2 = x^3 + Ax^2 + x with the Montgomery ladder
for efficient scalar multiplication (avoids y-coordinate computation).
"""

from __future__ import annotations

import math
import random

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


def _montgomery_add(
    xp: int, zp: int, xq: int, zq: int, xd: int, zd: int, n: int, ctx: InstrumentedContext
) -> tuple[int, int]:
    """Montgomery differential addition: compute P + Q given P, Q, and P - Q = D.

    Uses projective coordinates (X:Z) on a Montgomery curve.
    """
    u = (xp - zp) * (xq + zq) % n
    v = (xp + zp) * (xq - zq) % n
    add = (u + v) % n
    sub = (u - v) % n
    x_out = (zd * add * add) % n
    z_out = (xd * sub * sub) % n
    ctx.record_mod_mul(6)
    return x_out, z_out


def _montgomery_double(
    xp: int, zp: int, a24: int, n: int, ctx: InstrumentedContext
) -> tuple[int, int]:
    """Montgomery point doubling using projective coordinates.

    a24 = (A + 2) / 4 for the curve By^2 = x^3 + Ax^2 + x.
    """
    s = (xp + zp) % n
    d = (xp - zp) % n
    ss = s * s % n
    dd = d * d % n
    diff = (ss - dd) % n
    x_out = ss * dd % n
    z_out = diff * (dd + a24 * diff) % n
    ctx.record_mod_mul(5)
    return x_out, z_out


def _montgomery_ladder(
    k: int, x: int, z: int, a24: int, n: int, ctx: InstrumentedContext
) -> tuple[int, int]:
    """Compute k * P using the Montgomery ladder (constant-time binary method).

    Returns projective coordinates (X:Z).
    """
    if k <= 1:
        return x, z
    x0, z0 = x, z  # R0 = P
    x1, z1 = _montgomery_double(x, z, a24, n, ctx)  # R1 = 2P

    bits = k.bit_length()
    for i in range(bits - 2, -1, -1):
        ctx.record_iteration()
        if (k >> i) & 1:
            x0, z0 = _montgomery_add(x0, z0, x1, z1, x, z, n, ctx)
            x1, z1 = _montgomery_double(x1, z1, a24, n, ctx)
        else:
            x1, z1 = _montgomery_add(x1, z1, x0, z0, x, z, n, ctx)
            x0, z0 = _montgomery_double(x0, z0, a24, n, ctx)

    return x0, z0


def _random_curve_and_point(
    n: int, rng: random.Random
) -> tuple[int, int, int]:
    """Generate a random Montgomery curve and point.

    Uses Suyama's parametrization to generate a curve with known point.
    Returns (x, z, a24) where a24 = (A+2)/4 mod n.
    If a factor is found during setup, returns (factor, 0, 0).
    """
    sigma = rng.randrange(6, n)
    u = (sigma * sigma - 5) % n
    v = (4 * sigma) % n
    u3 = u * u * u % n
    x0 = u3 % n
    z0 = v * v * v % n
    diff = (v - u) % n
    diff3 = diff * diff * diff % n
    three_u_plus_v = (3 * u + v) % n
    numerator = diff3 * three_u_plus_v % n
    denominator = (16 * u3 * v) % n
    try:
        inv_denom = pow(denominator, -1, n)
    except ValueError:
        g = math.gcd(denominator, n)
        return g, 0, 0
    a24 = numerator * inv_denom % n
    return x0, z0, a24


class ECM(FactoringAlgorithm):
    """Lenstra's Elliptic Curve Method (stage 1).

    Tries multiple random Montgomery curves. For each curve, computes
    Q = k*P where k is the product of prime powers up to B1.
    If gcd(Q_z, n) is nontrivial, a factor is found.

    When gcd == n (both factor orders are smooth), automatically retries
    with per-prime gcd checks (backtracking) to separate the factors.
    """

    def __init__(self, b1: int = 10_000, num_curves: int = 40, seed: int | None = None) -> None:
        self._b1 = b1
        self._num_curves = num_curves
        self._seed = seed

    @property
    def name(self) -> str:
        return "ecm"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        rng = random.Random(self._seed)
        primes = _small_primes(self._b1)

        for curve_idx in range(self._num_curves):
            result = self._try_curve(n, primes, rng, ctx)
            if result is not None:
                return result, f"found on curve {curve_idx + 1}, B1={self._b1}"

        return None, f"no factor after {self._num_curves} curves, B1={self._b1}"

    def _try_curve(
        self, n: int, primes: list[int], rng: random.Random, ctx: InstrumentedContext
    ) -> int | None:
        """Try a single random curve. Returns a nontrivial factor or None."""
        x0, z0, a24 = _random_curve_and_point(n, rng)

        # Check if curve generation found a factor
        if z0 == 0 and a24 == 0:
            g = x0
            ctx.record_gcd()
            if 1 < g < n:
                return g
            return None

        # First pass: batch gcd (fast, checks every 20 primes)
        x, z = x0, z0
        z_accum = 1
        count = 0

        for p in primes:
            pe = p
            while pe * p <= self._b1:
                pe *= p
            x, z = _montgomery_ladder(pe, x, z, a24, n, ctx)
            z_accum = z_accum * z % n
            ctx.record_mod_mul(1)
            count += 1

            if count % 20 == 0:
                ctx.record_gcd()
                g = math.gcd(z_accum, n)
                if 1 < g < n:
                    return g
                if g == n:
                    # Both factors are smooth -- backtrack with fine-grained checks
                    return self._try_curve_fine(x0, z0, a24, n, primes, ctx)
                z_accum = 1

        # Final gcd
        ctx.record_gcd()
        g = math.gcd(z_accum, n)
        if 1 < g < n:
            return g
        if g == n:
            return self._try_curve_fine(x0, z0, a24, n, primes, ctx)
        return None

    def _try_curve_fine(
        self, x: int, z: int, a24: int, n: int, primes: list[int],
        ctx: InstrumentedContext,
    ) -> int | None:
        """Re-run stage 1 from the saved initial point, checking gcd after each prime power.

        This handles the case where gcd(z_accum, n) == n, meaning both prime
        factors have smooth curve orders. By checking after each multiplication,
        we can catch the moment when only one factor's order has been exhausted.
        """
        for p in primes:
            pe = p
            while pe * p <= self._b1:
                pe *= p
            x, z = _montgomery_ladder(pe, x, z, a24, n, ctx)
            ctx.record_gcd()
            g = math.gcd(z, n)
            if 1 < g < n:
                return g
            if g == n:
                # Already hit n -- too late, both orders divide the partial product.
                # This can happen if both orders are identical up to this prime.
                # Continue; a later prime might split them.
                continue
        return None
