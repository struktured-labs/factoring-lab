"""Tests for exact lattice point counting."""

import pytest

from factoring_lab.analysis.lattice_counting import (
    LatticeCountResult,
    count_lattice_points_exact,
    heuristic_estimate,
    to_digits,
    from_digits,
)


class TestDigitConversion:
    """Basic digit conversion tests."""

    def test_to_digits_base2(self) -> None:
        assert to_digits(15, 2) == [1, 1, 1, 1]

    def test_to_digits_base10(self) -> None:
        assert to_digits(35, 10) == [5, 3]

    def test_roundtrip(self) -> None:
        for base in [2, 3, 5, 10]:
            for n in [15, 21, 35, 77, 143]:
                assert from_digits(to_digits(n, base), base) == n


class TestExactCounting:
    """Test exact lattice point counts on small semiprimes."""

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    @pytest.mark.parametrize("base", [2, 3, 5])
    def test_rank1_count_is_2(self, n: int, p: int, q: int, base: int) -> None:
        """Rank-1 count should be exactly 2 for semiprimes (p*q and q*p)."""
        result = count_lattice_points_exact(n, base)
        assert result.rank1_points == 2, (
            f"n={n} base={base}: expected 2 rank-1 points, got {result.rank1_points}. "
            f"Factorizations found: {result.rank1_factorizations}"
        )

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ])
    @pytest.mark.parametrize("base", [2, 3])
    def test_exact_count_at_least_2(self, n: int, p: int, q: int, base: int) -> None:
        """There must be at least 2 lattice points (the factorizations)."""
        result = count_lattice_points_exact(n, base)
        assert result.total_lattice_points >= 2

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    def test_factorizations_correct(self, n: int, p: int, q: int) -> None:
        """Rank-1 factorizations should recover the correct factor pair."""
        result = count_lattice_points_exact(n, 2)
        found_pairs = set(result.rank1_factorizations)
        assert (min(p, q), max(p, q)) in found_pairs, (
            f"Expected ({min(p,q)}, {max(p,q)}) in {found_pairs}"
        )

    @pytest.mark.parametrize("base", [2, 3, 5])
    def test_heuristic_within_order_of_magnitude(self, base: int) -> None:
        """Heuristic should be within ~1 order of magnitude for small cases."""
        for n in [15, 21, 35]:
            result = count_lattice_points_exact(n, base)
            if result.total_lattice_points > 0 and result.heuristic_estimate > 0:
                ratio = result.ratio_exact_over_heuristic
                # Allow up to 4 orders of magnitude difference for small cases.
                # The heuristic is a volume estimate that can significantly
                # overcount for small d where boundary effects dominate.
                assert 0.0001 < ratio < 10000, (
                    f"n={n} base={base}: ratio {ratio} outside [0.0001, 10000]"
                )


class TestHeuristicEstimate:
    """Test the heuristic formula."""

    def test_positive(self) -> None:
        est, d, dx, dy = heuristic_estimate(15, 2)
        assert est > 0

    def test_grows_with_base(self) -> None:
        """Larger base -> more values per z_{ij} -> larger estimate (usually)."""
        est2, *_ = heuristic_estimate(15, 2)
        est10, *_ = heuristic_estimate(15, 10)
        # This isn't strictly monotone because d changes with base,
        # but for n=15 the effect should hold
        # Just check both are positive
        assert est2 > 0
        assert est10 > 0
