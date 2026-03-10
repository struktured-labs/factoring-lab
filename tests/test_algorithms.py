"""Tests for factoring algorithms."""

from factoring_lab.algorithms import TrialDivision, PollardRho, PollardPM1
from factoring_lab.generators.semiprimes import (
    balanced_semiprime,
    smooth_pm1_semiprime,
    unbalanced_semiprime,
)


# Known semiprimes for deterministic testing
SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (77, {7, 11}),
    (221, {13, 17}),
    (10403, {101, 103}),
    (1000003 * 1000033, {1000003, 1000033}),
]


class TestTrialDivision:
    def test_small_semiprimes(self):
        algo = TrialDivision()
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}"
            assert result.factor in factors

    def test_even(self):
        result = TrialDivision().factor(100)
        assert result.success
        assert result.factor == 2

    def test_prime(self):
        result = TrialDivision().factor(997)
        assert not result.success

    def test_with_limit(self):
        # With a low limit, should fail on large factors
        result = TrialDivision(limit=10).factor(221)
        assert not result.success

    def test_instrumentation(self):
        result = TrialDivision().factor(77)
        assert result.iteration_count > 0
        assert result.modular_multiplies > 0


class TestPollardRho:
    def test_small_semiprimes(self):
        algo = PollardRho()
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors

    def test_instrumentation(self):
        result = PollardRho().factor(10403)
        assert result.success
        assert result.gcd_calls > 0
        assert result.modular_multiplies > 0

    def test_generated_balanced(self):
        spec = balanced_semiprime(bits=40, seed=42)
        result = PollardRho().factor(spec.n)
        assert result.success
        assert result.factor in (spec.p, spec.q)


class TestPollardPM1:
    def test_smooth_factor(self):
        # p-1 smooth semiprime should be easy for p-1 method
        spec = smooth_pm1_semiprime(bits=32, smoothness_bound=100, seed=42)
        result = PollardPM1(bound=1000).factor(spec.n)
        assert result.success, f"failed: {result.notes}"
        assert result.factor in (spec.p, spec.q)

    def test_small_semiprimes(self):
        algo = PollardPM1(bound=10000)
        for n, factors in SMALL_SEMIPRIMES[:3]:
            result = algo.factor(n)
            # p-1 may not succeed on all, but should on some small ones
            if result.success:
                assert result.factor in factors

    def test_instrumentation(self):
        spec = smooth_pm1_semiprime(bits=32, smoothness_bound=50, seed=7)
        result = PollardPM1(bound=1000).factor(spec.n)
        assert result.gcd_calls > 0
        assert result.modular_multiplies > 0


class TestGeneratedInstances:
    """Cross-algorithm tests on generated instances."""

    def test_unbalanced_trial_division(self):
        spec = unbalanced_semiprime(bits=48, small_bits=16, seed=42)
        result = TrialDivision().factor(spec.n)
        assert result.success
        assert result.factor == spec.p  # small factor should be found
