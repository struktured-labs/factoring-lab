"""Tests for semiprime generators."""

from factoring_lab.generators.semiprimes import (
    balanced_semiprime,
    unbalanced_semiprime,
    smooth_pm1_semiprime,
    random_semiprime,
    generate_family,
    _is_prime,
)


def test_is_prime_basic():
    assert not _is_prime(0)
    assert not _is_prime(1)
    assert _is_prime(2)
    assert _is_prime(3)
    assert not _is_prime(4)
    assert _is_prime(5)
    assert _is_prime(7919)
    assert not _is_prime(7917)


def test_balanced_semiprime():
    spec = balanced_semiprime(bits=32, seed=42)
    assert spec.n == spec.p * spec.q
    assert _is_prime(spec.p)
    assert _is_prime(spec.q)
    assert spec.p != spec.q
    assert spec.family == "balanced"
    assert spec.balance_ratio > 0.8  # should be close to 1.0


def test_balanced_reproducible():
    a = balanced_semiprime(bits=32, seed=99)
    b = balanced_semiprime(bits=32, seed=99)
    assert a.n == b.n


def test_unbalanced_semiprime():
    spec = unbalanced_semiprime(bits=64, small_bits=16, seed=42)
    assert spec.n == spec.p * spec.q
    assert _is_prime(spec.p)
    assert _is_prime(spec.q)
    assert spec.family == "unbalanced"
    assert spec.balance_ratio < 0.5


def test_smooth_pm1_semiprime():
    spec = smooth_pm1_semiprime(bits=32, smoothness_bound=100, seed=42)
    assert spec.n == spec.p * spec.q
    assert _is_prime(spec.p)
    assert _is_prime(spec.q)
    assert spec.family == "smooth_pm1"


def test_random_semiprime():
    spec = random_semiprime(bits=32, seed=42)
    assert spec.n == spec.p * spec.q
    assert _is_prime(spec.p)
    assert _is_prime(spec.q)
    assert spec.family == "random"


def test_generate_family():
    specs = list(generate_family("balanced", bits=32, count=5, seed=42))
    assert len(specs) == 5
    for spec in specs:
        assert spec.n == spec.p * spec.q
        assert spec.family == "balanced"

    # Different seeds produce different instances
    ns = [s.n for s in specs]
    assert len(set(ns)) == 5
