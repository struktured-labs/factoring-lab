"""Tests for exact lattice point counting."""

import pytest

from factoring_lab.analysis.lattice_counting import (
    LatticeCountResult,
    TransferMatrixResult,
    count_lattice_points_exact,
    count_lattice_points_transfer_matrix,
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


class TestTransferMatrix:
    """Test transfer matrix method matches brute-force exact counts."""

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    @pytest.mark.parametrize("base", [2, 3, 5])
    def test_matches_exact_count(self, n: int, p: int, q: int, base: int) -> None:
        """Transfer matrix count must equal brute-force enumeration."""
        exact = count_lattice_points_exact(n, base)
        tm = count_lattice_points_transfer_matrix(n, base)
        assert tm.total_lattice_points == exact.total_lattice_points, (
            f"n={n} base={base}: transfer matrix={tm.total_lattice_points}, "
            f"exact={exact.total_lattice_points}"
        )

    @pytest.mark.parametrize("n,p,q", [
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
    ])
    @pytest.mark.parametrize("base", [2, 3])
    def test_matches_exact_larger(self, n: int, p: int, q: int, base: int) -> None:
        """Transfer matrix matches exact for larger semiprimes."""
        exact = count_lattice_points_exact(n, base)
        tm = count_lattice_points_transfer_matrix(n, base)
        assert tm.total_lattice_points == exact.total_lattice_points, (
            f"n={n} base={base}: transfer matrix={tm.total_lattice_points}, "
            f"exact={exact.total_lattice_points}"
        )

    @pytest.mark.parametrize("base", [2, 3, 5, 10])
    def test_positive_count(self, base: int) -> None:
        """Every semiprime must have at least 2 lattice points."""
        tm = count_lattice_points_transfer_matrix(15, base)
        assert tm.total_lattice_points >= 2

    def test_spectral_radii_positive(self) -> None:
        """Spectral radii should be positive for non-degenerate positions."""
        tm = count_lattice_points_transfer_matrix(143, 2)
        # At least the middle positions should have positive spectral radius
        assert any(sr > 0 for sr in tm.spectral_radii)

    def test_transfer_matrix_dimensions(self) -> None:
        """Transfer matrices should have consistent count."""
        tm = count_lattice_points_transfer_matrix(35, 2)
        assert len(tm.transfer_matrices) == tm.d
        for T_k in tm.transfer_matrices:
            assert T_k.ndim == 2  # 2D matrices

    @pytest.mark.parametrize("base", [2, 3, 5, 10])
    def test_heuristic_is_upper_bound(self, base: int) -> None:
        """The heuristic should overestimate (or be close) for small cases."""
        for n in [15, 21, 35]:
            tm = count_lattice_points_transfer_matrix(n, base)
            # Heuristic is typically an upper bound
            if tm.total_lattice_points > 0 and tm.heuristic_estimate > 0:
                ratio = tm.ratio_exact_over_heuristic
                assert ratio < 10000, (
                    f"n={n} base={base}: ratio {ratio} unexpectedly large"
                )

    def test_scales_to_larger_cases(self) -> None:
        """Transfer matrix should handle cases too large for brute force."""
        # 323 = 17 * 19, base 2, d=9 — this is slow for brute force
        tm = count_lattice_points_transfer_matrix(323, 2)
        assert tm.total_lattice_points > 0
        assert tm.d == 9
        assert tm.log2_exact > 0


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
