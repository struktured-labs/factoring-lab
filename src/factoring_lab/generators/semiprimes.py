"""Semiprime generators with controlled structure."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator


@dataclass
class SemiprimeSpec:
    """Describes a generated semiprime and its known factors."""

    n: int
    p: int
    q: int
    family: str
    bit_size_p: int
    bit_size_q: int
    metadata: dict[str, object] | None = None

    @property
    def bit_size_n(self) -> int:
        return self.n.bit_length()

    @property
    def balance_ratio(self) -> float:
        """Ratio of smaller to larger factor bit sizes. 1.0 = balanced."""
        small, big = sorted([self.bit_size_p, self.bit_size_q])
        return small / big if big > 0 else 0.0


def _is_prime(n: int) -> bool:
    """Miller-Rabin primality test with deterministic witnesses for < 3.3e24."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Deterministic witnesses sufficient for n < 3.3e24
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _random_prime(bits: int, rng: random.Random) -> int:
    """Generate a random prime with the given bit length."""
    while True:
        candidate = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_prime(candidate):
            return candidate


def _smooth_prime(
    bits: int, smoothness_bound: int, rng: random.Random, max_attempts: int = 10_000
) -> int:
    """Generate a prime p where p-1 is B-smooth.

    Constructs p-1 as a product of small primes, then checks if p is prime.
    """
    small_primes = [p for p in range(2, smoothness_bound + 1) if _is_prime(p)]

    for _ in range(max_attempts):
        # Build p-1 as product of random small prime powers
        product = 2  # ensure p-1 is even
        while product.bit_length() < bits - 2:
            prime = rng.choice(small_primes)
            product *= prime

        # Trim if overshot
        if product.bit_length() > bits + 1:
            continue

        p = product + 1
        if p.bit_length() == bits and _is_prime(p):
            return p

    raise ValueError(
        f"Could not generate {bits}-bit smooth prime with B={smoothness_bound} "
        f"after {max_attempts} attempts"
    )


def balanced_semiprime(bits: int, seed: int | None = None) -> SemiprimeSpec:
    """Generate a semiprime N=p*q where p and q have similar bit sizes.

    Args:
        bits: target bit size of N (each factor will be ~bits/2)
        seed: random seed for reproducibility
    """
    rng = random.Random(seed)
    half = bits // 2
    p = _random_prime(half, rng)
    q = _random_prime(half, rng)
    while p == q:
        q = _random_prime(half, rng)
    return SemiprimeSpec(
        n=p * q,
        p=min(p, q),
        q=max(p, q),
        family="balanced",
        bit_size_p=min(p, q).bit_length(),
        bit_size_q=max(p, q).bit_length(),
    )


def unbalanced_semiprime(
    bits: int, small_bits: int = 16, seed: int | None = None
) -> SemiprimeSpec:
    """Generate a semiprime with one small and one large factor.

    Args:
        bits: target bit size of N
        small_bits: bit size of the smaller factor
        seed: random seed
    """
    rng = random.Random(seed)
    large_bits = bits - small_bits
    p = _random_prime(small_bits, rng)
    q = _random_prime(large_bits, rng)
    return SemiprimeSpec(
        n=p * q,
        p=min(p, q),
        q=max(p, q),
        family="unbalanced",
        bit_size_p=min(p, q).bit_length(),
        bit_size_q=max(p, q).bit_length(),
        metadata={"small_bits": small_bits},
    )


def smooth_pm1_semiprime(
    bits: int,
    smoothness_bound: int = 1000,
    seed: int | None = None,
) -> SemiprimeSpec:
    """Generate a semiprime where one factor p has p-1 being B-smooth.

    This creates instances that Pollard p-1 should handle well.

    Args:
        bits: target bit size of N
        smoothness_bound: B-smoothness bound for p-1
        seed: random seed
    """
    rng = random.Random(seed)
    half = bits // 2
    p = _smooth_prime(half, smoothness_bound, rng)
    q = _random_prime(half, rng)
    while p == q:
        q = _random_prime(half, rng)
    return SemiprimeSpec(
        n=p * q,
        p=min(p, q),
        q=max(p, q),
        family="smooth_pm1",
        bit_size_p=min(p, q).bit_length(),
        bit_size_q=max(p, q).bit_length(),
        metadata={"smoothness_bound": smoothness_bound},
    )


def random_semiprime(bits: int, seed: int | None = None) -> SemiprimeSpec:
    """Generate a random semiprime with no special structure.

    Factor sizes are randomly distributed (not necessarily balanced).

    Args:
        bits: target bit size of N
        seed: random seed
    """
    rng = random.Random(seed)
    # Random split of bits between factors (at least 8 bits each)
    min_bits = max(8, bits // 8)
    p_bits = rng.randint(min_bits, bits - min_bits)
    q_bits = bits - p_bits
    p = _random_prime(p_bits, rng)
    q = _random_prime(q_bits, rng)
    return SemiprimeSpec(
        n=p * q,
        p=min(p, q),
        q=max(p, q),
        family="random",
        bit_size_p=min(p, q).bit_length(),
        bit_size_q=max(p, q).bit_length(),
    )


def generate_family(
    family: str,
    bits: int,
    count: int,
    seed: int = 42,
    **kwargs: object,
) -> Iterator[SemiprimeSpec]:
    """Generate a family of semiprimes.

    Args:
        family: one of "balanced", "unbalanced", "smooth_pm1", "random"
        bits: target bit size
        count: number of instances to generate
        seed: base seed (each instance uses seed + i)
        **kwargs: passed to the specific generator
    """
    generators = {
        "balanced": balanced_semiprime,
        "unbalanced": unbalanced_semiprime,
        "smooth_pm1": smooth_pm1_semiprime,
        "random": random_semiprime,
    }
    gen_fn = generators.get(family)
    if gen_fn is None:
        raise ValueError(f"Unknown family '{family}', choose from {list(generators)}")

    for i in range(count):
        yield gen_fn(bits=bits, seed=seed + i, **kwargs)  # type: ignore[arg-type]
