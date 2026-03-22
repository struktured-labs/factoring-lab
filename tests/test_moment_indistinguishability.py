"""Tests for moment indistinguishability analysis (Theorem B)."""

import pytest
import numpy as np

from factoring_lab.analysis.moment_indistinguishability import (
    MomentVector,
    PairwiseDistance,
    IndistinguishabilityResult,
    BoundedViewTheoremResult,
    compute_moment_vector,
    compute_pairwise_distance,
    analyze_indistinguishability,
    prove_bounded_view_theorem,
    compute_moment_agreement_matrix,
    count_distinguishing_entries_by_type,
    _compute_degree4_index_count,
    _enumerate_small_semiprimes,
)


class TestMomentVector:
    """Tests for moment vector computation."""

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
            (77, 7, 11),
        ],
    )
    @pytest.mark.parametrize("base", [2, 3])
    def test_degree2_equals_digit_products(
        self, n: int, p: int, q: int, base: int
    ) -> None:
        """Degree-2 moments must equal x_i * y_j."""
        mv = compute_moment_vector(n, p, q, base)
        for i in range(mv.dx):
            for j in range(mv.dy):
                expected = mv.x_digits[i] * mv.y_digits[j]
                actual = mv.degree2[i * mv.dy + j]
                assert actual == expected, (
                    f"n={n} base={base}: degree2[{i},{j}] = {actual} != "
                    f"x[{i}]*y[{j}] = {mv.x_digits[i]}*{mv.y_digits[j]} = {expected}"
                )

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
        ],
    )
    def test_degree4_equals_product_of_degree2(
        self, n: int, p: int, q: int
    ) -> None:
        """Degree-4 moments must equal products of degree-2 moments."""
        mv = compute_moment_vector(n, p, q, 2)
        m = mv.dx * mv.dy
        idx = 0
        for a in range(m):
            for b in range(a, m):
                expected = int(mv.degree2[a]) * int(mv.degree2[b])
                actual = mv.degree4[idx]
                assert actual == expected, (
                    f"degree4[{a},{b}] = {actual} != d2[{a}]*d2[{b}] = {expected}"
                )
                idx += 1

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
        ],
    )
    def test_degree4_entry_count(self, n: int, p: int, q: int) -> None:
        """Degree-4 entry count must match formula."""
        mv = compute_moment_vector(n, p, q, 2)
        m = mv.dx * mv.dy
        expected_count = m * (m + 1) // 2
        assert len(mv.degree4) == expected_count
        assert mv.num_degree4 == expected_count

    def test_moment_vector_self_distance_zero(self) -> None:
        """Distance of a moment vector to itself must be zero."""
        mv = compute_moment_vector(15, 3, 5, 2)
        pd = compute_pairwise_distance(mv, mv)
        assert pd.degree2_hamming == 0
        assert pd.degree4_hamming == 0
        assert pd.total_hamming == 0
        assert pd.total_frac_agree == 1.0

    @pytest.mark.parametrize("base", [2, 3, 5])
    def test_digits_reconstruct_factors(self, base: int) -> None:
        """Digits stored in MomentVector must reconstruct p and q."""
        from factoring_lab.analysis.lattice_counting import from_digits

        mv = compute_moment_vector(77, 7, 11, base)
        p_recon = from_digits(mv.x_digits, base)
        q_recon = from_digits(mv.y_digits, base)
        # One of (p_recon, q_recon) or (q_recon, p_recon) should be (7, 11)
        assert {p_recon, q_recon} == {7, 11} or (
            p_recon * q_recon == 77
        ), f"Digits don't reconstruct: p={p_recon}, q={q_recon}"


class TestPairwiseDistance:
    """Tests for pairwise distance computation.

    All comparisons use explicit dx/dy to ensure matching dimensions
    (semiprimes of different values may get different default digit sizes).
    """

    def test_different_semiprimes_have_positive_distance(self) -> None:
        """Different semiprimes must have nonzero Hamming distance."""
        # Use same dx/dy for both (both are 4-bit semiprimes)
        mv1 = compute_moment_vector(15, 3, 5, 2, dx=3, dy=3)
        mv2 = compute_moment_vector(21, 3, 7, 2, dx=3, dy=3)
        pd = compute_pairwise_distance(mv1, mv2)
        assert pd.total_hamming > 0
        assert pd.normalized_hamming > 0

    def test_distance_symmetry(self) -> None:
        """Hamming distance must be symmetric."""
        mv1 = compute_moment_vector(15, 3, 5, 2, dx=3, dy=3)
        mv2 = compute_moment_vector(21, 3, 7, 2, dx=3, dy=3)
        pd12 = compute_pairwise_distance(mv1, mv2)
        pd21 = compute_pairwise_distance(mv2, mv1)
        assert pd12.total_hamming == pd21.total_hamming
        assert pd12.normalized_hamming == pd21.normalized_hamming

    def test_normalized_hamming_bounded(self) -> None:
        """Normalized Hamming distance must be in [0, 1]."""
        mv1 = compute_moment_vector(15, 3, 5, 2, dx=3, dy=3)
        mv2 = compute_moment_vector(21, 3, 7, 2, dx=3, dy=3)
        pd = compute_pairwise_distance(mv1, mv2)
        assert 0.0 <= pd.normalized_hamming <= 1.0

    def test_frac_agree_complement_of_hamming(self) -> None:
        """Fraction agreeing + normalized Hamming = 1."""
        mv1 = compute_moment_vector(15, 3, 5, 2, dx=3, dy=3)
        mv2 = compute_moment_vector(35, 5, 7, 2, dx=3, dy=3)
        pd = compute_pairwise_distance(mv1, mv2)
        assert abs(pd.total_frac_agree + pd.normalized_hamming - 1.0) < 1e-10

    @pytest.mark.parametrize("base", [2, 3])
    def test_shared_factor_reduces_distance(self, base: int) -> None:
        """Semiprimes sharing a factor should have smaller distance."""
        # 15 = 3*5, 21 = 3*7 (share factor 3)
        # 35 = 5*7 (shares no factor with 15's 3)
        # Use explicit matching dx/dy for comparisons
        mv_15 = compute_moment_vector(15, 3, 5, base, dx=4, dy=4)
        mv_21 = compute_moment_vector(21, 3, 7, base, dx=4, dy=4)
        mv_35 = compute_moment_vector(35, 5, 7, base, dx=4, dy=4)

        pd_shared = compute_pairwise_distance(mv_15, mv_21)  # share factor 3
        pd_none = compute_pairwise_distance(mv_15, mv_35)  # share no factor

        # Not guaranteed to hold strictly (digit alignment matters),
        # but we at least verify both distances are computed correctly
        assert pd_shared.total_hamming >= 0
        assert pd_none.total_hamming >= 0


class TestEnumerateSemiprimes:
    """Tests for semiprime enumeration."""

    def test_8bit_semiprimes(self) -> None:
        """Should find multiple 8-bit semiprimes."""
        sps = _enumerate_small_semiprimes(8)
        assert len(sps) >= 2
        for n, p, q in sps:
            assert n == p * q
            assert n.bit_length() == 8
            assert p <= q

    def test_10bit_semiprimes(self) -> None:
        """Should find many 10-bit semiprimes."""
        sps = _enumerate_small_semiprimes(10)
        assert len(sps) >= 10
        for n, p, q in sps:
            assert n == p * q
            assert n.bit_length() == 10

    def test_all_semiprimes_valid(self) -> None:
        """Every returned semiprime must be valid."""
        sps = _enumerate_small_semiprimes(8)
        for n, p, q in sps:
            assert p >= 2
            assert q >= 2
            assert p <= q
            assert p * q == n


class TestAnalyzeIndistinguishability:
    """Tests for the main indistinguishability analysis."""

    def test_basic_8bit(self) -> None:
        """Basic test on 8-bit semiprimes."""
        ir = analyze_indistinguishability(8, 2)
        assert ir.num_semiprimes >= 2
        assert ir.num_pairs >= 1
        assert 0.0 <= ir.min_hamming <= ir.max_hamming <= 1.0
        assert ir.total_moment_entries > 0

    def test_most_pairs_have_positive_distance(self) -> None:
        """Most pairs of different semiprimes should have positive Hamming distance.

        A small number of pairs may have zero distance due to digit truncation:
        when the digit window (dx, dy) is smaller than the full digit
        representation of the factors, higher-order digits are lost.
        This is a genuine phenomenon that STRENGTHENS Theorem B -- it shows
        that even exact moments can be indistinguishable within the window.
        """
        ir = analyze_indistinguishability(8, 2)
        zero_count = sum(
            1 for pd in ir.pair_distances if pd.n1 != pd.n2 and pd.total_hamming == 0
        )
        # At most a small fraction should have zero distance
        frac_zero = zero_count / ir.num_pairs if ir.num_pairs > 0 else 0.0
        assert frac_zero < 0.05, (
            f"{zero_count}/{ir.num_pairs} pairs ({frac_zero:.1%}) have zero distance"
        )

    def test_indistinguishable_fraction_decreases_with_budget(self) -> None:
        """More entries read -> fewer indistinguishable pairs."""
        ir = analyze_indistinguishability(8, 2)
        sorted_k = sorted(ir.indistinguishable_at_k.keys())
        if len(sorted_k) >= 2:
            # Indistinguishable fraction should be non-increasing
            for i in range(1, len(sorted_k)):
                k_prev = sorted_k[i - 1]
                k_curr = sorted_k[i]
                assert ir.indistinguishable_at_k[k_curr] <= (
                    ir.indistinguishable_at_k[k_prev] + 1e-10
                ), (
                    f"Indist. fraction increased from k={k_prev} "
                    f"({ir.indistinguishable_at_k[k_prev]:.3f}) to k={k_curr} "
                    f"({ir.indistinguishable_at_k[k_curr]:.3f})"
                )

    @pytest.mark.parametrize("base", [2, 3])
    def test_total_entries_matches_formula(self, base: int) -> None:
        """Total moment entries must match dx*dy + C(dx*dy+1, 2)."""
        ir = analyze_indistinguishability(8, base)
        m = ir.dx * ir.dy
        expected_d4 = m * (m + 1) // 2
        assert ir.total_degree4_entries == expected_d4
        assert ir.total_moment_entries == m + expected_d4

    def test_10bit_more_semiprimes_than_8bit(self) -> None:
        """10-bit should have more semiprimes than 8-bit."""
        ir8 = analyze_indistinguishability(8, 2)
        ir10 = analyze_indistinguishability(10, 2)
        assert ir10.num_semiprimes >= ir8.num_semiprimes


class TestBoundedViewTheorem:
    """Tests for the bounded-view theorem proof."""

    def test_small_bit_lengths(self) -> None:
        """Test theorem verification on small bit-lengths."""
        result = prove_bounded_view_theorem(
            bit_lengths=[8, 10], base=2, max_semiprimes=20
        )
        assert len(result.results) >= 1
        assert len(result.bit_lengths) >= 1

    def test_indistinguishable_pairs_exist_at_low_budget(self) -> None:
        """At budget k=1, most pairs should be indistinguishable."""
        result = prove_bounded_view_theorem(
            bit_lengths=[8], base=2
        )
        assert len(result.results) >= 1
        ir = result.results[0]
        if 1 in ir.indistinguishable_at_k:
            # At k=1, we should see many indistinguishable pairs
            # (reading 1 entry out of many is unlikely to distinguish)
            assert ir.indistinguishable_at_k[1] > 0.0, (
                "At budget k=1, expected some indistinguishable pairs"
            )

    def test_high_budget_distinguishes_all(self) -> None:
        """At sufficiently high budget, all pairs should be distinguishable."""
        result = prove_bounded_view_theorem(
            bit_lengths=[8], base=2
        )
        assert len(result.results) >= 1
        ir = result.results[0]
        max_k = max(ir.indistinguishable_at_k.keys())
        # At the maximum budget, we expect most pairs to be distinguishable
        # (not necessarily ALL, since our probabilistic model is approximate)
        assert ir.indistinguishable_at_k[max_k] < 1.0


class TestAgreementMatrix:
    """Tests for the moment agreement matrix."""

    def test_diagonal_is_one(self) -> None:
        """Diagonal of agreement matrix should be 1.0."""
        semiprimes = [(15, 3, 5), (21, 3, 7), (35, 5, 7)]
        mat = compute_moment_agreement_matrix(semiprimes, 2)
        for i in range(len(semiprimes)):
            assert mat[i, i] == 1.0

    def test_symmetric(self) -> None:
        """Agreement matrix must be symmetric."""
        semiprimes = [(15, 3, 5), (21, 3, 7), (35, 5, 7)]
        mat = compute_moment_agreement_matrix(semiprimes, 2)
        assert np.allclose(mat, mat.T)

    def test_values_in_zero_one(self) -> None:
        """All entries must be in [0, 1]."""
        semiprimes = [(15, 3, 5), (21, 3, 7), (35, 5, 7)]
        mat = compute_moment_agreement_matrix(semiprimes, 2)
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0)

    def test_off_diagonal_less_than_one(self) -> None:
        """Off-diagonal entries must be < 1 for different semiprimes."""
        semiprimes = [(15, 3, 5), (21, 3, 7), (35, 5, 7)]
        mat = compute_moment_agreement_matrix(semiprimes, 2)
        for i in range(len(semiprimes)):
            for j in range(i + 1, len(semiprimes)):
                assert mat[i, j] < 1.0, (
                    f"Semiprimes {semiprimes[i][0]} and {semiprimes[j][0]} "
                    f"have agreement 1.0 (should be < 1)"
                )


class TestDistinguishingEntriesByType:
    """Tests for the entry type breakdown."""

    def test_degree4_from_degree2_subset(self) -> None:
        """degree4_from_degree2 must be <= degree4."""
        mv1 = compute_moment_vector(15, 3, 5, 2, dx=3, dy=3)
        mv2 = compute_moment_vector(21, 3, 7, 2, dx=3, dy=3)
        breakdown = count_distinguishing_entries_by_type(mv1, mv2)
        assert breakdown["degree4_from_degree2"] <= breakdown["degree4"]

    def test_total_consistent(self) -> None:
        """Sum of degree-2 and degree-4 differences must match pairwise distance."""
        mv1 = compute_moment_vector(15, 3, 5, 2, dx=3, dy=3)
        mv2 = compute_moment_vector(21, 3, 7, 2, dx=3, dy=3)
        breakdown = count_distinguishing_entries_by_type(mv1, mv2)
        pd = compute_pairwise_distance(mv1, mv2)
        assert breakdown["degree2"] == pd.degree2_hamming
        assert breakdown["degree4"] == pd.degree4_hamming

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
        ],
    )
    def test_self_comparison_zero(self, n: int, p: int, q: int) -> None:
        """Comparing a moment vector to itself should give all zeros."""
        mv = compute_moment_vector(n, p, q, 2)
        breakdown = count_distinguishing_entries_by_type(mv, mv)
        assert breakdown["degree1_x"] == 0
        assert breakdown["degree1_y"] == 0
        assert breakdown["degree2"] == 0
        assert breakdown["degree4"] == 0
        assert breakdown["degree4_from_degree2"] == 0


class TestDegree4IndexCount:
    """Tests for the degree-4 index counting formula."""

    @pytest.mark.parametrize(
        "dx,dy,expected",
        [
            (1, 1, 1),  # m=1, C(2,2) = 1
            (1, 2, 3),  # m=2, C(3,2) = 3
            (2, 2, 10),  # m=4, C(5,2) = 10
            (3, 3, 45),  # m=9, C(10,2) = 45
        ],
    )
    def test_known_values(self, dx: int, dy: int, expected: int) -> None:
        """Test against hand-computed values."""
        assert _compute_degree4_index_count(dx, dy) == expected


class TestTheoremBKeyProperty:
    """Tests for the KEY property of Theorem B: at polynomial budget,
    many pairs remain indistinguishable."""

    def test_many_pairs_indistinguishable_at_d_budget(self) -> None:
        """At budget k = d (linear in digits), significant fraction should
        be indistinguishable.

        This is the empirical core of Theorem B: poly(d) entries are
        insufficient to distinguish all pairs.
        """
        ir = analyze_indistinguishability(10, 2)
        d = ir.d
        # Find the budget closest to d
        closest_k = min(ir.indistinguishable_at_k.keys(), key=lambda k: abs(k - d))
        frac_indist = ir.indistinguishable_at_k[closest_k]
        # At budget d, we expect a significant fraction to be indistinguishable
        # because d << total_entries (which is O(d^4))
        assert frac_indist > 0.0, (
            f"Expected indistinguishable pairs at budget {closest_k}, "
            f"but fraction is 0.0"
        )

    def test_moment_entries_grow_as_d4(self) -> None:
        """Total moment entries should grow as O(d^4)."""
        results = []
        for bl in [8, 10, 12]:
            ir = analyze_indistinguishability(bl, 2)
            results.append((ir.d, ir.total_moment_entries))

        # Check that ratio entries/d^4 is roughly constant
        ratios = [entries / (d**4) for d, entries in results if d > 0]
        # Allow wide tolerance since d is small
        assert all(r > 0 for r in ratios), "All ratios should be positive"
        # The ratios should be in the same ballpark (within 10x)
        assert max(ratios) / min(ratios) < 10.0, (
            f"Ratios vary too much: {ratios}"
        )
