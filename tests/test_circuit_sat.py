"""Tests for circuit-based SAT factoring."""

import pytest
from factoring_lab.algorithms.circuit_sat import CircuitSAT
from factoring_lab.generators.semiprimes import balanced_semiprime


SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (21, {3, 7}),
    (35, {5, 7}),
    (77, {7, 11}),
    (143, {11, 13}),
    (221, {13, 17}),
    (10403, {101, 103}),
]


class TestCircuitSAT:
    """Test circuit SAT encoding on small semiprimes."""

    def test_small_semiprimes(self):
        algo = CircuitSAT(timeout_ms=10_000)
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, f"wrong factor for {n}: {result.factor}"

    def test_prime_detection(self):
        """A prime has no non-trivial factors; solver should return unsat."""
        algo = CircuitSAT(timeout_ms=10_000)
        for p in [97, 101, 127, 131]:
            result = algo.factor(p)
            assert not result.success, f"prime {p} should not be factorable"

    def test_balanced_16bit(self):
        """Circuit SAT should handle 16-bit balanced semiprimes."""
        spec = balanced_semiprime(bits=16, seed=42)
        algo = CircuitSAT(timeout_ms=15_000)
        result = algo.factor(spec.n)
        assert result.success, f"failed on {spec.n}: {result.notes}"
        assert result.factor in (spec.p, spec.q)

    def test_even_number(self):
        """Even numbers are handled by the base class shortcut."""
        algo = CircuitSAT()
        result = algo.factor(6)
        assert result.success
        assert result.factor == 2

    def test_algorithm_name(self):
        algo = CircuitSAT()
        assert algo.name == "circuit_sat"
