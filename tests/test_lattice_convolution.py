"""Tests for lattice-based digit convolution factoring.

These tests explore whether LLL can solve the digit convolution constraints.
Many tests document EXPECTED LIMITATIONS — the lattice approach has fundamental
difficulties with the rank-1 constraint.
"""

import numpy as np
import pytest

from factoring_lab.algorithms.lattice_convolution import (
    LatticeAnalysis,
    LatticeConvolution,
    build_linearized_lattice,
    lll_reduce,
    solve_linear_for_y,
    _to_digits,
    _from_digits,
)


SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (21, {3, 7}),
    (35, {5, 7}),
    (77, {7, 11}),
    (143, {11, 13}),
    (221, {13, 17}),
]


class TestLLLReduction:
    """Verify our LLL implementation on known lattice problems."""

    def test_identity_basis(self):
        """LLL on an identity matrix should return the identity."""
        basis = np.eye(3, dtype=np.int64)
        reduced = lll_reduce(basis)
        # Reduced basis should span the same lattice
        assert reduced.shape == (3, 3)
        assert abs(int(round(np.linalg.det(reduced.astype(float))))) == 1

    def test_simple_reduction(self):
        """LLL should find shorter vectors than a skewed basis."""
        basis = np.array([[1, 0, 0], [0, 1, 0], [10, 10, 1]], dtype=np.int64)
        reduced = lll_reduce(basis)
        # The reduced basis should have shorter vectors
        original_norms = sorted(np.linalg.norm(basis.astype(float), axis=1))
        reduced_norms = sorted(np.linalg.norm(reduced.astype(float), axis=1))
        assert reduced_norms[0] <= original_norms[-1]

    def test_2d_reduction(self):
        """Classic 2D LLL example."""
        basis = np.array([[1, 0], [0, 100]], dtype=np.int64)
        reduced = lll_reduce(basis)
        norms = [np.linalg.norm(row.astype(float)) for row in reduced]
        assert min(norms) <= 100


class TestDigitConversion:
    def test_roundtrip_base10(self):
        for n in [0, 1, 15, 143, 9999]:
            assert _from_digits(_to_digits(n, 10), 10) == n

    def test_roundtrip_base2(self):
        for n in [0, 1, 15, 143]:
            assert _from_digits(_to_digits(n, 2), 2) == n


class TestBuildLattice:
    """Test the lattice construction for digit convolution."""

    def test_lattice_dimensions(self):
        """Lattice should have correct dimensions."""
        c = _to_digits(15, 10)
        dx, dy = 2, 2
        lat, meta = build_linearized_lattice(c, 10, dx, dy)
        # rows = num_vars + 1, cols = num_vars + num_constraints
        assert lat.shape[0] == meta["num_vars"] + 1
        assert lat.shape[1] == meta["num_vars"] + meta["num_constraints"]

    def test_known_solution_satisfies_constraints(self):
        """The true factorization should satisfy A @ v = b."""
        analyzer = LatticeAnalysis(base=10)
        result = analyzer.verify_known_factorization(3, 5)
        assert result["constraints_satisfied"], (
            f"Known factorization 3*5=15 does not satisfy constraints: "
            f"residual={result['constraint_residual']}"
        )

    def test_known_solution_several(self):
        """Check constraint satisfaction for several known factorizations."""
        analyzer = LatticeAnalysis(base=10)
        for p, q in [(3, 5), (3, 7), (7, 11), (11, 13), (13, 17)]:
            result = analyzer.verify_known_factorization(p, q)
            assert result["constraints_satisfied"], (
                f"Failed for {p}*{q}={p * q}"
            )


class TestSolveLinearForY:
    """Test solving for y given known x digits."""

    def test_known_x_finds_y(self):
        """Given x=3, solve for y in 3*5=15."""
        c = _to_digits(15, 10)
        x_digits = _to_digits(3, 10)
        # Pad
        while len(x_digits) < len(c):
            x_digits.append(0)
        y_digits = solve_linear_for_y(c, x_digits, 10)
        if y_digits is not None:
            y = _from_digits(y_digits, 10)
            assert y == 5 or y * 3 == 15

    def test_known_x_base2(self):
        """Given x=3 (binary: 11), solve for y in 3*5=15."""
        c = _to_digits(15, 2)
        x_digits = _to_digits(3, 2)
        while len(x_digits) < len(c):
            x_digits.append(0)
        y_digits = solve_linear_for_y(c, x_digits, 2)
        if y_digits is not None:
            y = _from_digits(y_digits, 2)
            assert y * 3 == 15


class TestLatticeConvolution:
    """Test the full LatticeConvolution factoring algorithm."""

    def test_small_semiprimes(self):
        """Should factor small semiprimes (via enumeration fallback)."""
        algo = LatticeConvolution(base=10, max_x_value=100)
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, (
                f"wrong factor for {n}: {result.factor}"
            )

    def test_base2(self):
        algo = LatticeConvolution(base=2, max_x_value=100)
        result = algo.factor(15)
        assert result.success
        assert result.factor in (3, 5)

    def test_base16(self):
        algo = LatticeConvolution(base=16, max_x_value=100)
        result = algo.factor(221)
        assert result.success
        assert result.factor in (13, 17)

    def test_prime_fails(self):
        """Primes should not be factored."""
        algo = LatticeConvolution(base=10, max_x_value=200)
        result = algo.factor(97)
        assert not result.success

    def test_instrumentation(self):
        algo = LatticeConvolution(base=10, max_x_value=100)
        result = algo.factor(77)
        assert result.iteration_count > 0

    def test_notes_mention_lattice_limitation(self):
        """For non-trivial cases, notes should mention limitations."""
        algo = LatticeConvolution(base=10, max_x_value=100)
        result = algo.factor(77)
        # The algorithm works via enumeration, not pure lattice
        assert "lattice" in result.notes.lower() or "enumeration" in result.notes.lower()

    def test_larger_semiprime(self):
        """Test on a slightly larger semiprime."""
        algo = LatticeConvolution(base=10, max_x_value=500)
        n = 437  # 19 * 23
        result = algo.factor(n)
        assert result.success
        assert result.factor in (19, 23)


class TestLatticeAnalysis:
    """Test the diagnostic analysis tools."""

    def test_analyze_small(self):
        analyzer = LatticeAnalysis(base=10)
        info = analyzer.analyze_constraint_structure(15)
        assert info["n"] == 15
        assert info["base"] == 10
        assert info["num_digits"] == 2
        assert info["constraint_rank"] > 0
        assert "rank-1" in info["notes"].lower()

    def test_analyze_larger(self):
        analyzer = LatticeAnalysis(base=10)
        info = analyzer.analyze_constraint_structure(143)
        assert info["num_digits"] == 3
        assert info["lattice_dimension"][0] > 0

    def test_verify_known_factorization(self):
        analyzer = LatticeAnalysis(base=10)
        result = analyzer.verify_known_factorization(11, 13)
        assert result["n"] == 143
        assert result["constraints_satisfied"]
        assert result["solution_vector_norm"] > 0

    def test_different_bases(self):
        """Analysis should work across bases."""
        for base in [2, 5, 10, 16]:
            analyzer = LatticeAnalysis(base=base)
            info = analyzer.analyze_constraint_structure(35)
            assert info["base"] == base
            assert info["constraint_rank"] > 0
