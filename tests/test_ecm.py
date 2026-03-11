"""Tests for the Elliptic Curve Method (ECM) factoring algorithm."""

from factoring_lab.algorithms.ecm import ECM
from factoring_lab.generators.semiprimes import (
    balanced_semiprime,
    smooth_pm1_semiprime,
)


# Known small semiprimes
SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (77, {7, 11}),
    (221, {13, 17}),
    (10403, {101, 103}),
    (1000003 * 1000033, {1000003, 1000033}),
]


class TestECMSmall:
    def test_small_semiprimes(self):
        algo = ECM(b1=2000, num_curves=50, seed=42)
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, f"wrong factor {result.factor} for {n}"

    def test_even(self):
        result = ECM(seed=1).factor(100)
        assert result.success
        assert result.factor == 2

    def test_prime_returns_failure(self):
        result = ECM(b1=500, num_curves=10, seed=42).factor(997)
        assert not result.success

    def test_large_prime_returns_failure(self):
        result = ECM(b1=500, num_curves=5, seed=42).factor(104729)
        assert not result.success


class TestECMGenerated:
    def test_balanced_32bit(self):
        for seed in range(5):
            spec = balanced_semiprime(bits=32, seed=seed)
            result = ECM(b1=2000, num_curves=30, seed=seed).factor(spec.n)
            assert result.success, f"failed on {spec.n} (seed={seed}): {result.notes}"
            assert result.factor in (spec.p, spec.q)

    def test_balanced_48bit(self):
        for seed in range(3):
            spec = balanced_semiprime(bits=48, seed=seed)
            result = ECM(b1=5000, num_curves=50, seed=seed).factor(spec.n)
            assert result.success, f"failed on {spec.n} (seed={seed}): {result.notes}"
            assert result.factor in (spec.p, spec.q)

    def test_smooth_pm1(self):
        """ECM should easily handle p-1 smooth instances."""
        for seed in range(5):
            spec = smooth_pm1_semiprime(bits=32, smoothness_bound=100, seed=seed)
            result = ECM(b1=1000, num_curves=20, seed=seed).factor(spec.n)
            assert result.success, f"failed on smooth_pm1 seed={seed}: {result.notes}"
            assert result.factor in (spec.p, spec.q)


class TestECMInstrumentation:
    def test_instrumentation_recorded(self):
        result = ECM(b1=2000, num_curves=50, seed=42).factor(10403)
        assert result.success
        assert result.iteration_count > 0
        assert result.gcd_calls > 0
        assert result.modular_multiplies > 0

    def test_notes_contain_curve_info(self):
        result = ECM(b1=2000, num_curves=50, seed=42).factor(10403)
        assert result.success
        assert "curve" in result.notes
        assert "B1" in result.notes
