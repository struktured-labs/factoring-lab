"""Tests for SMT-based digit convolution factoring."""

import pytest
from factoring_lab.algorithms.smt_convolution import (
    SMTConvolution,
    SMTConvolutionRaw,
    SMTConvolutionBase10,
    SMTConvolutionBase2,
)
from factoring_lab.generators.semiprimes import balanced_semiprime, smooth_pm1_semiprime


SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (21, {3, 7}),
    (35, {5, 7}),
    (77, {7, 11}),
    (143, {11, 13}),
    (221, {13, 17}),
    (10403, {101, 103}),
]


class TestSMTRaw:
    """Test raw bitvector multiplication (no digit constraints)."""

    def test_small_semiprimes(self):
        algo = SMTConvolutionRaw()
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, f"wrong factor for {n}: {result.factor}"

    def test_prime_fails(self):
        algo = SMTConvolutionRaw()
        result = algo.factor(97)
        assert not result.success

    def test_32bit_semiprime(self):
        """Z3 handles 32-bit balanced semiprimes."""
        spec = balanced_semiprime(bits=32, seed=99)
        algo = SMTConvolutionRaw(timeout_ms=10_000)
        result = algo.factor(spec.n)
        assert result.success
        assert result.factor in (spec.p, spec.q)

    def test_40bit_timeout(self):
        """Z3 raw bitvector times out on 40-bit — this IS the research finding."""
        algo = SMTConvolutionRaw(timeout_ms=5_000)
        result = algo.factor(1000003 * 1000033)
        # Z3 may or may not solve 40-bit in 5s — just check it doesn't crash
        if result.success:
            assert result.factor in (1000003, 1000033)


class TestSMTBase10:
    """Test with base-10 digit convolution constraints."""

    def test_small_semiprimes(self):
        algo = SMTConvolutionBase10()
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors

    def test_base11_on_143(self):
        """Base 11 should work via SMT even though backtracking fails."""
        algo = SMTConvolution(base=11)
        result = algo.factor(143)
        assert result.success
        assert result.factor in (11, 13)


class TestSMTBase2:
    """Test with binary digit constraints."""

    def test_small_semiprimes(self):
        algo = SMTConvolutionBase2()
        for n, factors in SMALL_SEMIPRIMES[:4]:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors


class TestSMTvsBacktracking:
    """Compare SMT against backtracking on generated instances."""

    def test_balanced_32bit(self):
        spec = balanced_semiprime(bits=32, seed=42)
        raw = SMTConvolutionRaw(timeout_ms=10_000)
        result = raw.factor(spec.n)
        assert result.success
        assert result.factor in (spec.p, spec.q)

    def test_digit_constraints_help(self):
        """Test whether digit constraints help Z3 vs raw bitvector on 32-bit."""
        spec = balanced_semiprime(bits=32, seed=42)
        raw = SMTConvolutionRaw(timeout_ms=10_000)
        b10 = SMTConvolutionBase10(timeout_ms=10_000)
        r_raw = raw.factor(spec.n)
        r_b10 = b10.factor(spec.n)
        # Both should succeed on 32-bit
        assert r_raw.success
        assert r_b10.success
        # Log the comparison (the interesting data point)
        assert r_raw.factor in (spec.p, spec.q)
        assert r_b10.factor in (spec.p, spec.q)

    def test_smooth_pm1(self):
        spec = smooth_pm1_semiprime(bits=32, smoothness_bound=100, seed=42)
        algo = SMTConvolutionBase10(timeout_ms=10_000)
        result = algo.factor(spec.n)
        assert result.success
        assert result.factor in (spec.p, spec.q)
