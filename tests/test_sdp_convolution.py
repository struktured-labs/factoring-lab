"""Tests for SDP relaxation and alternating projection factoring.

These tests are EXPLORATORY — many document expected limitations rather
than guaranteed successes. The SDP approach is a convex relaxation of a
fundamentally hard problem, so failures are informative.
"""

import pytest

from factoring_lab.algorithms.sdp_convolution import (
    AlternatingProjection,
    SDPConvolution,
    SDPAnalysis,
    _to_digits,
    _from_digits,
    _build_convolution_constraints,
    _check_factorization,
)


SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (21, {3, 7}),
    (77, {7, 11}),
    (143, {11, 13}),
    (221, {13, 17}),
]


class TestHelpers:
    """Test utility functions."""

    def test_to_digits_base10(self):
        assert _to_digits(15, 10) == [5, 1]
        assert _to_digits(143, 10) == [3, 4, 1]

    def test_from_digits_base10(self):
        assert _from_digits([5, 1], 10) == 15
        assert _from_digits([3, 4, 1], 10) == 143

    def test_roundtrip(self):
        for n in [0, 1, 15, 77, 143, 221, 9999]:
            assert _from_digits(_to_digits(n, 10), 10) == n

    def test_roundtrip_base2(self):
        for n in [1, 15, 77, 143]:
            assert _from_digits(_to_digits(n, 2), 2) == n

    def test_check_factorization_success(self):
        assert _check_factorization(3, 15) == 3
        assert _check_factorization(5, 15) == 3  # returns min factor

    def test_check_factorization_failure(self):
        assert _check_factorization(4, 15) is None
        assert _check_factorization(1, 15) is None
        assert _check_factorization(15, 15) is None

    def test_build_constraints(self):
        """Constraint matrix should have correct dimensions."""
        c = _to_digits(15, 10)
        z_vars, A, b_vec = _build_convolution_constraints(c, 10, 2, 2)
        d = len(c)
        num_z = len(z_vars)
        assert A.shape == (d, num_z + d)
        assert len(b_vec) == d

    def test_constraints_satisfied_by_true_solution(self):
        """The true factorization should satisfy the linear constraints."""
        import numpy as np

        p, q = 3, 5
        n = p * q
        base = 10
        c = _to_digits(n, base)
        d = len(c)
        num_x = d // 2 + 1
        num_y = d // 2 + 1

        z_vars, A, b_vec = _build_convolution_constraints(c, base, num_x, num_y)
        num_z = len(z_vars)

        x_digits = _to_digits(p, base)
        y_digits = _to_digits(q, base)
        while len(x_digits) < num_x:
            x_digits.append(0)
        while len(y_digits) < num_y:
            y_digits.append(0)

        # Build z values from true factorization
        z_index = {pair: idx for idx, pair in enumerate(z_vars)}
        z_vals = np.zeros(num_z)
        for idx, (i, j) in enumerate(z_vars):
            z_vals[idx] = x_digits[i] * y_digits[j]

        # Compute carries
        t_vals = np.zeros(d)
        carry = 0
        for k in range(d):
            alpha_k = sum(
                x_digits[i] * y_digits[k - i]
                for i in range(min(k + 1, len(x_digits)))
                if k - i < len(y_digits)
            )
            total = alpha_k + carry
            t_vals[k] = total // base
            carry = total // base

        v = np.concatenate([z_vals, t_vals])
        residual = np.linalg.norm(A @ v - b_vec)
        assert residual < 1e-10, f"Constraint residual {residual} too large"


class TestAlternatingProjection:
    """Test the alternating projection factoring algorithm."""

    def test_small_semiprimes(self):
        """Should factor small semiprimes via random search."""
        algo = AlternatingProjection(
            base=10, max_restarts=200, max_iters_per_restart=30, seed=42
        )
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, (
                f"wrong factor for {n}: got {result.factor}"
            )

    def test_convergence_tracking(self):
        """Algorithm should track iterations."""
        algo = AlternatingProjection(
            base=10, max_restarts=10, max_iters_per_restart=10, seed=42
        )
        result = algo.factor(15)
        assert result.iteration_count > 0

    def test_prime_returns_failure(self):
        """Primes should not be factored."""
        algo = AlternatingProjection(
            base=10, max_restarts=50, max_iters_per_restart=20, seed=42
        )
        result = algo.factor(97)
        assert not result.success

    def test_even_shortcut(self):
        """Even numbers should be caught by base class."""
        algo = AlternatingProjection(base=10, seed=42)
        result = algo.factor(14)
        assert result.success
        assert result.factor == 2

    def test_different_bases(self):
        """Should work across different bases."""
        for base in [2, 10, 16]:
            algo = AlternatingProjection(
                base=base, max_restarts=200, max_iters_per_restart=30, seed=42
            )
            result = algo.factor(15)
            assert result.success, f"failed in base {base}: {result.notes}"
            assert result.factor in (3, 5)

    def test_larger_semiprime(self):
        """Test on 437 = 19 * 23."""
        algo = AlternatingProjection(
            base=10, max_restarts=500, max_iters_per_restart=50, seed=42
        )
        result = algo.factor(437)
        assert result.success, f"failed on 437: {result.notes}"
        assert result.factor in (19, 23)


class TestSDPConvolution:
    """Test the SDP relaxation factoring algorithm."""

    def test_small_semiprimes(self):
        """Should factor small semiprimes."""
        algo = SDPConvolution(
            base=10, max_restarts=50, max_iters=50, seed=42
        )
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, (
                f"wrong factor for {n}: got {result.factor}"
            )

    def test_prime_returns_failure(self):
        """Primes should not be factored."""
        algo = SDPConvolution(
            base=10, max_restarts=20, max_iters=20, seed=42
        )
        result = algo.factor(97)
        assert not result.success

    def test_notes_contain_info(self):
        """Notes should contain diagnostic information."""
        algo = SDPConvolution(
            base=10, max_restarts=5, max_iters=10, seed=42
        )
        result = algo.factor(15)
        assert len(result.notes) > 0

    def test_instrumentation(self):
        """Should track iterations."""
        algo = SDPConvolution(
            base=10, max_restarts=5, max_iters=10, seed=42
        )
        result = algo.factor(15)
        assert result.iteration_count > 0


class TestSDPAnalysis:
    """Test the SDP analysis diagnostics."""

    def test_integrality_gap_small(self):
        """Analyze integrality gap for small semiprimes."""
        analyzer = SDPAnalysis(base=10)
        result = analyzer.analyze_integrality_gap(3, 5)
        assert result["n"] == 15
        assert result["constraints_satisfied"]
        assert result["true_rank1_ratio"] > 0

    def test_integrality_gap_larger(self):
        """Analyze integrality gap for larger semiprime."""
        analyzer = SDPAnalysis(base=10)
        result = analyzer.analyze_integrality_gap(11, 13)
        assert result["n"] == 143
        assert result["constraints_satisfied"]

    def test_different_bases(self):
        """Analysis should work across bases."""
        for base in [2, 10, 16]:
            analyzer = SDPAnalysis(base=base)
            result = analyzer.analyze_integrality_gap(3, 5)
            assert result["base"] == base
            assert result["constraints_satisfied"]
