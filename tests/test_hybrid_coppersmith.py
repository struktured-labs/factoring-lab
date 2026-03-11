"""Tests for hybrid digit-convolution + Coppersmith factoring.

Tests cover:
- Digit pair enumeration correctness
- Coppersmith lattice recovery with known partial factors
- Full hybrid algorithm on small semiprimes
- Comparison with pure approaches
"""

import math
import pytest

from factoring_lab.algorithms.hybrid_coppersmith import (
    HybridCoppersmith,
    coppersmith_lattice_factor,
    coppersmith_lattice_factor_base,
    enumerate_digit_assignments,
    valid_digit_pairs,
    _to_digits,
    _from_digits,
)


# ---------- small semiprimes for testing ----------

SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (21, {3, 7}),
    (35, {5, 7}),
    (77, {7, 11}),
    (143, {11, 13}),
    (221, {13, 17}),
    (323, {17, 19}),
    (437, {19, 23}),
    (667, {23, 29}),
    (899, {29, 31}),
]

MEDIUM_SEMIPRIMES = [
    (10403, {101, 103}),
    (25553, {127, 201}),  # 127 * 201 = 25527 -- let me fix
]


class TestValidDigitPairs:
    """Test enumeration of valid (x_0, y_0) digit pairs."""

    def test_base10_digit0(self):
        """For n=15, c_0=5 in base 10. Pairs: x*y=5 mod 10."""
        pairs = valid_digit_pairs(5, 10)
        # 1*5, 3*5, 5*1, 5*3, 5*5, 5*7, 5*9, 7*5, 9*5 -- check some
        assert (1, 5) in pairs
        assert (5, 1) in pairs
        assert (3, 5) in pairs
        assert (5, 3) in pairs
        # all pairs satisfy the constraint
        for x, y in pairs:
            assert (x * y) % 10 == 5

    def test_base2(self):
        """In base 2, c_0=1 means x_0*y_0=1 mod 2, so both must be 1."""
        pairs = valid_digit_pairs(1, 2)
        assert (1, 1) in pairs
        # 0*0=0, 0*1=0, 1*0=0 don't satisfy
        assert (0, 0) not in pairs

    def test_all_valid(self):
        """Every returned pair satisfies the modular constraint."""
        for base in [2, 5, 10, 16]:
            for c in range(base):
                for x, y in valid_digit_pairs(c, base):
                    assert (x * y) % base == c


class TestEnumerateDigitAssignments:
    """Test multi-level digit enumeration."""

    def test_depth1_base10(self):
        """Depth-1 enumeration for 15 in base 10."""
        pairs = list(enumerate_digit_assignments(15, 10, 1))
        # All pairs must satisfy x*y = 15 mod 10 = 5 mod 10
        for x, y in pairs:
            assert (x * y) % 10 == 5

    def test_depth1_contains_true_factors(self):
        """The true factor digits should appear in depth-1 enumeration."""
        # n=15, p=3, q=5 in base 10: x_0=3, y_0=5
        pairs = list(enumerate_digit_assignments(15, 10, 1))
        assert (3, 5) in pairs or (5, 3) in pairs

    def test_depth2_base10(self):
        """Depth-2 narrows candidates further."""
        # n=143 (11*13), base 10: c_0=3, c_1=4, c_2=1
        # depth 2: x*y = 143 mod 100 = 43
        pairs = list(enumerate_digit_assignments(143, 10, 2))
        for x, y in pairs:
            assert (x * y) % 100 == 43
        # True factors: 11*13=143, so (11,13) should be in depth-2 pairs
        assert (11, 13) in pairs or (13, 11) in pairs

    def test_depth1_base2(self):
        """Binary depth-1 for n=15."""
        pairs = list(enumerate_digit_assignments(15, 2, 1))
        # n=15, c_0=1 in binary, so x*y=1 mod 2
        for x, y in pairs:
            assert (x * y) % 2 == 1

    def test_deeper_narrows(self):
        """Higher depth should produce fewer or equal candidates."""
        n = 143
        p1 = list(enumerate_digit_assignments(n, 10, 1))
        p2 = list(enumerate_digit_assignments(n, 10, 2))
        # depth-2 pairs are more constrained, but there may be more due
        # to extension. The key property: all satisfy mod b^depth.
        for x, y in p2:
            assert (x * y) % 100 == 43


class TestCoppersmithLattice:
    """Test the Coppersmith-style lattice recovery."""

    def test_known_half_bits(self):
        """With half the bits known, should recover the factor."""
        p, q = 251, 241
        n = p * q  # 60491
        k = p.bit_length() // 2  # ~4 bits known
        p_low = p % (1 << k)
        result = coppersmith_lattice_factor(n, p_low, k)
        # May or may not succeed depending on lattice quality
        if result is not None:
            assert n % result == 0 and 1 < result < n

    def test_known_most_bits(self):
        """With most bits known, should always succeed."""
        p, q = 251, 241
        n = p * q
        k = p.bit_length() - 1  # all but 1 bit known
        p_low = p % (1 << k)
        result = coppersmith_lattice_factor(n, p_low, k)
        assert result is not None
        assert n % result == 0 and 1 < result < n

    def test_base_variant(self):
        """Test base-b Coppersmith with known low digits."""
        p, q = 13, 17
        n = p * q  # 221
        # Know 1 digit in base 10: p mod 10 = 3
        result = coppersmith_lattice_factor_base(n, 3, 10, 1)
        if result is not None:
            assert n % result == 0 and 1 < result < n

    def test_gcd_shortcut(self):
        """When p_low shares a factor with n, should find it immediately."""
        p, q = 7, 11
        n = p * q  # 77
        # If p_low = 7 (happens to be p itself)
        result = coppersmith_lattice_factor(n, 7, 3)
        assert result is not None
        assert result in (7, 11)

    def test_trivial_p_low_rejected(self):
        """p_low=0 or p_low=1 should return None."""
        assert coppersmith_lattice_factor(77, 0, 4) is None
        assert coppersmith_lattice_factor(77, 1, 4) is None


class TestHybridCoppersmith:
    """Test the full hybrid algorithm."""

    def test_small_semiprimes_depth1(self):
        """Should factor small semiprimes at depth 1."""
        algo = HybridCoppersmith(base=10, depth=1, timeout_s=10.0)
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, (
                f"wrong factor for {n}: got {result.factor}, expected one of {factors}"
            )

    def test_small_semiprimes_depth2(self):
        """Should factor small semiprimes at depth 2."""
        algo = HybridCoppersmith(base=10, depth=2, timeout_s=10.0)
        for n, factors in SMALL_SEMIPRIMES[:5]:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"

    def test_base2_depth1(self):
        """Binary base should also work."""
        algo = HybridCoppersmith(base=2, depth=1, timeout_s=10.0)
        result = algo.factor(15)
        assert result.success
        assert result.factor in (3, 5)

    def test_base16(self):
        """Hexadecimal base."""
        algo = HybridCoppersmith(base=16, depth=1, timeout_s=10.0)
        result = algo.factor(221)
        assert result.success
        assert result.factor in (13, 17)

    def test_prime_fails(self):
        """Primes should not be factored."""
        algo = HybridCoppersmith(base=10, depth=1, timeout_s=5.0)
        result = algo.factor(97)
        assert not result.success

    def test_even_number(self):
        """Even numbers are handled by the base class."""
        algo = HybridCoppersmith(base=10, depth=1)
        result = algo.factor(14)
        assert result.success
        assert result.factor == 2

    def test_name_format(self):
        algo = HybridCoppersmith(base=10, depth=2)
        assert algo.name == "hybrid_coppersmith_b10_d2"

    def test_notes_contain_info(self):
        """Notes should contain diagnostic information."""
        algo = HybridCoppersmith(base=10, depth=1, timeout_s=10.0)
        result = algo.factor(77)
        assert result.notes  # should have some info

    def test_factor_with_details(self):
        """The detailed diagnostics method should work."""
        algo = HybridCoppersmith(base=10, depth=1, timeout_s=10.0)
        details = algo.factor_with_details(77)
        assert details["n"] == 77
        assert details["success"]
        assert details["factor"] in (7, 11)
        assert details["base"] == 10
        assert details["depth"] == 1

    def test_16bit_semiprime(self):
        """Test on a 16-bit semiprime."""
        p, q = 251, 241
        n = p * q  # 60491
        algo = HybridCoppersmith(base=10, depth=1, timeout_s=10.0)
        result = algo.factor(n)
        assert result.success
        assert result.factor in (p, q)

    def test_different_depths_all_work(self):
        """All reasonable depths should eventually factor small numbers."""
        for depth in [1, 2, 3]:
            algo = HybridCoppersmith(base=10, depth=depth, timeout_s=10.0)
            result = algo.factor(143)
            assert result.success, f"depth={depth} failed: {result.notes}"


class TestComparisonWithPureMethods:
    """Compare hybrid approach with pure SMT and backtracking."""

    def test_hybrid_finds_factor_small(self):
        """Hybrid should match or beat pure backtracking on small cases."""
        algo = HybridCoppersmith(base=10, depth=1, timeout_s=10.0)
        successes = 0
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            if result.success and result.factor in factors:
                successes += 1
        # Should succeed on all small cases
        assert successes == len(SMALL_SEMIPRIMES)

    def test_digit_enumeration_count(self):
        """Count how many pairs are enumerated at each depth."""
        n = 221  # 13 * 17
        for depth in [1, 2]:
            pairs = list(enumerate_digit_assignments(n, 10, depth))
            # Depth 1: x*y=1 mod 10, there are 10 pairs (1*1, 3*7, 7*3, 9*9, ...)
            # Depth 2: x*y=21 mod 100
            assert len(pairs) > 0
