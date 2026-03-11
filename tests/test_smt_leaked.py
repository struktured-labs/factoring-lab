"""Tests for SMTLeakedBits factoring algorithm."""

from __future__ import annotations

import pytest

from factoring_lab.algorithms.smt_leaked import SMTLeakedBits
from factoring_lab.generators.semiprimes import balanced_semiprime


class TestSMTLeakedBitsBasic:
    """Basic functionality tests."""

    def test_full_leak_factors_small(self):
        """With 100% leaked bits, even a trivial case should solve."""
        spec = balanced_semiprime(32, seed=42)
        p = spec.p
        solver = SMTLeakedBits(leak_fraction=1.0, known_p=p, timeout_ms=10_000)
        result = solver.factor(spec.n)
        assert result.success
        assert result.factor in (spec.p, spec.q)

    def test_half_leak_small(self):
        """50% leaked bits on a 32-bit semiprime should be solvable."""
        spec = balanced_semiprime(32, seed=42)
        solver = SMTLeakedBits(leak_fraction=0.5, known_p=spec.p, timeout_ms=10_000)
        result = solver.factor(spec.n)
        assert result.success
        assert result.factor in (spec.p, spec.q)

    def test_no_leak_small(self):
        """No leaked bits - just plain SMT. Should still work for small n."""
        spec = balanced_semiprime(32, seed=42)
        solver = SMTLeakedBits(leak_fraction=0.0, known_p=spec.p, timeout_ms=10_000)
        result = solver.factor(spec.n)
        # May or may not succeed, but should not crash
        if result.success:
            assert result.factor in (spec.p, spec.q)

    def test_factor_with_leak_method(self):
        """Test the factor_with_leak convenience method."""
        spec = balanced_semiprime(32, seed=42)
        p = spec.p
        num_bits = p.bit_length() // 2
        known_bits = p & ((1 << num_bits) - 1)

        solver = SMTLeakedBits(timeout_ms=10_000)
        result = solver.factor_with_leak(spec.n, known_bits, num_bits)
        assert result.success
        assert result.factor in (spec.p, spec.q)

    def test_invalid_leak_fraction(self):
        """leak_fraction outside [0, 1] should raise."""
        with pytest.raises(ValueError):
            SMTLeakedBits(leak_fraction=1.5)
        with pytest.raises(ValueError):
            SMTLeakedBits(leak_fraction=-0.1)

    def test_name_property(self):
        solver = SMTLeakedBits(leak_fraction=0.25)
        assert "leaked" in solver.name
        assert "0.25" in solver.name

    def test_with_base(self):
        """Test with digit convolution constraints on top."""
        spec = balanced_semiprime(32, seed=42)
        solver = SMTLeakedBits(
            leak_fraction=0.5, known_p=spec.p, base=10, timeout_ms=10_000
        )
        result = solver.factor(spec.n)
        assert result.success
        assert result.factor in (spec.p, spec.q)


class TestSMTLeakedBitsScaling:
    """Test that more leaked bits makes larger problems solvable."""

    def test_more_leak_helps_48bit(self):
        """For 48-bit semiprimes, 70% leak should beat 0% leak."""
        spec = balanced_semiprime(48, seed=42)

        solver_high = SMTLeakedBits(
            leak_fraction=0.7, known_p=spec.p, timeout_ms=15_000
        )
        result_high = solver_high.factor(spec.n)
        assert result_high.success, "70% leak should solve 48-bit"
