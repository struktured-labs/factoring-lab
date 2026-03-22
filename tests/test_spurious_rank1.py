"""Tests for spurious near-rank-1 lattice point analysis."""

import pytest
import numpy as np

from factoring_lab.analysis.spurious_rank1 import (
    RankProfile,
    NearRank1Summary,
    _svd_rank_deficiency,
    _is_integer_rank1,
    _recover_factorization,
    enumerate_rank_profiles,
    count_near_rank1_points,
    analyze_near_rank1,
    DEFAULT_THRESHOLDS,
)


class TestSvdRankDeficiency:
    """Tests for the SVD-based rank deficiency computation."""

    def test_zero_matrix(self) -> None:
        Z = np.zeros((3, 3), dtype=np.int64)
        s1, s2, rd = _svd_rank_deficiency(Z)
        assert s1 == 0.0
        assert s2 == 0.0
        assert rd == 0.0

    def test_rank1_matrix(self) -> None:
        """A rank-1 matrix should have rank_deficiency = 0."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        Z = np.outer(x, y).astype(np.int64)
        s1, s2, rd = _svd_rank_deficiency(Z)
        assert s1 > 0
        assert s2 < 1e-10
        assert rd < 1e-10

    def test_full_rank_matrix(self) -> None:
        """A full-rank matrix should have rank_deficiency > 0."""
        Z = np.array([[1, 0], [0, 1]], dtype=np.int64)
        s1, s2, rd = _svd_rank_deficiency(Z)
        assert s1 > 0
        assert s2 > 0
        assert rd == pytest.approx(1.0)  # identity: both singular values = 1

    def test_near_rank1_matrix(self) -> None:
        """A nearly rank-1 matrix with small perturbation."""
        x = np.array([3, 5])
        y = np.array([7, 11])
        Z = np.outer(x, y).astype(np.int64)
        Z[0, 0] += 1  # Small perturbation
        s1, s2, rd = _svd_rank_deficiency(Z)
        assert s1 > 0
        assert s2 > 0
        assert 0 < rd < 0.1  # Should be small but nonzero


class TestIntegerRank1:
    """Tests for exact integer rank-1 check."""

    def test_rank1_true(self) -> None:
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        Z = np.outer(x, y).astype(np.int64)
        assert _is_integer_rank1(Z, 3, 2) is True

    def test_rank1_false(self) -> None:
        Z = np.array([[1, 0], [0, 1]], dtype=np.int64)
        assert _is_integer_rank1(Z, 2, 2) is False

    def test_zero_matrix_is_rank1(self) -> None:
        """The zero matrix technically has all minors = 0."""
        Z = np.zeros((2, 3), dtype=np.int64)
        assert _is_integer_rank1(Z, 2, 3) is True


class TestRecoverFactorization:
    """Tests for factorization recovery from rank-1 Z matrices."""

    def test_recover_15(self) -> None:
        """Recover 3 * 5 = 15 from a rank-1 Z in base 2."""
        # 3 = [1,1] in base 2, 5 = [1,0,1] in base 2
        x = np.array([1, 1, 0])  # pad to dx=3
        y = np.array([1, 0, 1])  # pad to dy=3
        Z = np.outer(x, y).astype(np.int64)
        result = _recover_factorization(Z, 3, 3, 2, 15)
        assert result is not None
        assert result == (3, 5)

    def test_non_rank1_returns_none(self) -> None:
        Z = np.array([[1, 0], [0, 1]], dtype=np.int64)
        result = _recover_factorization(Z, 2, 2, 2, 15)
        assert result is None


class TestEnumerateRankProfiles:
    """Tests for full lattice point enumeration with rank profiles."""

    @pytest.mark.parametrize("n", [15, 21, 35])
    def test_count_matches_existing(self, n: int) -> None:
        """Profile count should match the existing exact enumeration."""
        from factoring_lab.analysis.lattice_counting import (
            count_lattice_points_exact,
        )
        base = 2
        existing = count_lattice_points_exact(n, base)
        profiles = enumerate_rank_profiles(n, base)
        assert len(profiles) == existing.total_lattice_points, (
            f"n={n}: profile count {len(profiles)} != "
            f"existing count {existing.total_lattice_points}"
        )

    @pytest.mark.parametrize("n,p,q", [(15, 3, 5), (21, 3, 7), (35, 5, 7)])
    def test_exactly_two_valid_factorizations(self, n: int, p: int, q: int) -> None:
        """There should be exactly 2 valid factorizations (p*q and q*p)."""
        profiles = enumerate_rank_profiles(n, 2)
        fact_count = sum(1 for pr in profiles if pr.is_valid_factorization)
        assert fact_count == 2, (
            f"n={n}: expected 2 factorizations, got {fact_count}"
        )

    def test_factorization_profiles_have_zero_rank_deficiency(self) -> None:
        """Valid factorizations should have sigma_2/sigma_1 ~ 0."""
        profiles = enumerate_rank_profiles(15, 2)
        for pr in profiles:
            if pr.is_valid_factorization:
                assert pr.rank_deficiency < 1e-10, (
                    f"Factorization has rank_deficiency={pr.rank_deficiency}"
                )
                assert pr.is_exact_rank1 is True

    def test_non_rank1_profiles_exist(self) -> None:
        """There should be some lattice points that are NOT rank-1."""
        profiles = enumerate_rank_profiles(15, 2)
        non_r1 = [pr for pr in profiles if not pr.is_exact_rank1]
        assert len(non_r1) > 0, "Expected some non-rank-1 lattice points"


class TestCountNearRank1Points:
    """Tests for the count_near_rank1_points entry point."""

    def test_returns_nonneg(self) -> None:
        count = count_near_rank1_points(15, 2, threshold=0.1)
        assert count >= 0

    def test_threshold_1_includes_all(self) -> None:
        """With threshold=1.0, should include (nearly) all points."""
        # rank_deficiency = sigma_2/sigma_1 is in [0, 1] for nonneg matrices.
        # At threshold=1.0 we include everything with rd < 1.0.
        # Some points may have rd exactly 1.0 but that's rare for integer matrices.
        count = count_near_rank1_points(15, 2, threshold=1.0)
        profiles = enumerate_rank_profiles(15, 2)
        # Should include all or almost all
        assert count >= len(profiles) - 1

    def test_includes_factorizations(self) -> None:
        """The factorization points (rank_deficiency ~ 0) should always be counted."""
        count = count_near_rank1_points(15, 2, threshold=0.1)
        assert count >= 2  # At least the two factorizations

    def test_stricter_threshold_fewer_points(self) -> None:
        """Stricter threshold should give fewer or equal points."""
        c1 = count_near_rank1_points(35, 2, threshold=0.1)
        c2 = count_near_rank1_points(35, 2, threshold=0.01)
        assert c2 <= c1


class TestAnalyzeNearRank1:
    """Tests for the full analysis function."""

    def test_basic_structure(self) -> None:
        result = analyze_near_rank1(15, 2)
        assert isinstance(result, NearRank1Summary)
        assert result.n == 15
        assert result.base == 2
        assert result.total_lattice_points > 0
        assert result.valid_factorizations == 2

    def test_thresholds_monotone(self) -> None:
        """Counts at looser thresholds should be >= counts at stricter thresholds."""
        result = analyze_near_rank1(21, 2, thresholds=[0.1, 0.01, 0.001])
        for i in range(len(result.counts_below_threshold) - 1):
            assert result.counts_below_threshold[i] >= result.counts_below_threshold[i + 1], (
                f"Non-monotone: threshold {result.thresholds[i]} has count "
                f"{result.counts_below_threshold[i]} < "
                f"{result.counts_below_threshold[i + 1]} at threshold "
                f"{result.thresholds[i + 1]}"
            )

    def test_spurious_count_nonneg(self) -> None:
        """Spurious counts should be non-negative."""
        result = analyze_near_rank1(15, 2)
        for t, count in result.spurious_count.items():
            assert count >= 0

    def test_spurious_is_near_minus_factorizations(self) -> None:
        """Spurious at threshold t = (near-rank-1 at t) - (valid factorizations at t)."""
        result = analyze_near_rank1(15, 2, thresholds=[0.1])
        near = result.counts_below_threshold[0]
        # Number of valid factorizations with rank_deficiency < threshold
        facts_below = sum(
            1 for p in result.rank_profiles
            if p.is_valid_factorization and p.rank_deficiency < 0.1
        )
        assert result.spurious_count[0.1] == near - facts_below

    @pytest.mark.parametrize("n", [15, 21, 35])
    def test_exact_rank1_includes_factorizations(self, n: int) -> None:
        """Exact rank-1 count should be >= valid factorization count."""
        result = analyze_near_rank1(n, 2)
        assert result.exact_rank1_points >= result.valid_factorizations

    def test_keep_profiles_false(self) -> None:
        """With keep_profiles=False, profiles list should be empty."""
        result = analyze_near_rank1(15, 2, keep_profiles=False)
        assert result.rank_profiles == []
        # But counts should still be correct
        assert result.total_lattice_points > 0

    def test_rank_deficiency_stats(self) -> None:
        """Summary statistics should be consistent."""
        result = analyze_near_rank1(15, 2)
        assert result.rank_deficiency_min <= result.rank_deficiency_median
        assert result.rank_deficiency_median <= result.rank_deficiency_max
        assert result.rank_deficiency_min >= 0.0

    @pytest.mark.parametrize("n,p,q", [(77, 7, 11)])
    def test_larger_semiprime(self, n: int, p: int, q: int) -> None:
        """Test on a slightly larger semiprime."""
        result = analyze_near_rank1(n, 2, thresholds=[0.1, 0.01])
        assert result.total_lattice_points > 2
        assert result.valid_factorizations == 2

    def test_base3(self) -> None:
        """Test with base 3."""
        result = analyze_near_rank1(15, 3, thresholds=[0.1])
        assert result.total_lattice_points > 0
        assert result.valid_factorizations == 2
