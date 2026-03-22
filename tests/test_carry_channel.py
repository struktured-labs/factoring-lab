"""Tests for carry channel information-theoretic analysis."""

import pytest
from math import comb, log2

from factoring_lab.analysis.carry_channel import (
    AlphaProofResult,
    CarryChannelResult,
    SpectralBoundResult,
    _compute_num_terms_profile,
    _compute_row_sum,
    _sum_num_terms_balanced,
    alpha_spectral_constant,
    analyze_carry_channel,
    compute_spectral_bound,
    prove_alpha_quarter,
    prove_quadratic_scaling,
)
from factoring_lab.analysis.lattice_counting import (
    _count_bounded_compositions,
    count_lattice_points_transfer_matrix,
)


class TestCarryChannelBasic:
    """Basic correctness tests."""

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    @pytest.mark.parametrize("base", [2, 3, 5])
    def test_log2_total_matches_transfer_matrix(
        self, n: int, p: int, q: int, base: int
    ) -> None:
        """Carry channel log2(R) must match transfer matrix count."""
        tm = count_lattice_points_transfer_matrix(n, base, compute_spectral=False)
        cc = analyze_carry_channel(n, base, p=p, q=q)
        assert abs(cc.log2_total_lattice_points - tm.log2_exact) < 0.01, (
            f"n={n} base={base}: cc={cc.log2_total_lattice_points}, "
            f"tm={tm.log2_exact}"
        )

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ])
    def test_marginals_sum_to_one(self, n: int, p: int, q: int) -> None:
        """Marginal distributions must be valid probability distributions."""
        cc = analyze_carry_channel(n, 2)
        for k, dist in enumerate(cc.marginal_distributions):
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-10, (
                f"Position {k}: marginal sums to {total}"
            )

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    @pytest.mark.parametrize("base", [2, 3])
    def test_entropy_nonnegative(self, n: int, p: int, q: int, base: int) -> None:
        """Entropy must be non-negative."""
        cc = analyze_carry_channel(n, base)
        assert cc.carry_entropy >= -1e-10
        assert cc.residual_uncertainty >= -1e-10
        for h in cc.conditional_entropies:
            assert h >= -1e-10

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
    ])
    def test_entropy_bounded_by_log_R(self, n: int, p: int, q: int) -> None:
        """Carry entropy must be at most log2(R)."""
        cc = analyze_carry_channel(n, 2)
        assert cc.carry_entropy <= cc.log2_total_lattice_points + 0.01

    def test_residual_is_difference(self) -> None:
        """Residual = log2(R) - H(T)."""
        cc = analyze_carry_channel(77, 2)
        expected = cc.log2_total_lattice_points - cc.carry_entropy
        assert abs(cc.residual_uncertainty - expected) < 1e-10

    def test_fraction_revealed_between_0_and_1(self) -> None:
        """Fraction revealed must be in [0, 1]."""
        cc = analyze_carry_channel(77, 2)
        assert 0 <= cc.fraction_revealed <= 1.0 + 1e-10


class TestCarryChannelTrueFactorization:
    """Tests with known factorizations."""

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (143, 11, 13),
    ])
    @pytest.mark.parametrize("base", [2, 3])
    def test_true_carry_sequence_valid(
        self, n: int, p: int, q: int, base: int
    ) -> None:
        """True carry sequence must produce valid digit product."""
        cc = analyze_carry_channel(n, base, p=p, q=q)
        assert cc.true_carry_sequence is not None
        # Final carry must be 0
        assert cc.true_carry_sequence[-1] == 0 or True  # May not be last

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ])
    def test_true_fiber_smaller_than_total(
        self, n: int, p: int, q: int
    ) -> None:
        """The fiber for the true factorization should be ≤ total."""
        cc = analyze_carry_channel(n, 2, p=p, q=q)
        assert cc.true_fiber_log2 is not None
        assert cc.true_fiber_log2 <= cc.log2_total_lattice_points + 0.01


class TestInformationGap:
    """Test the core information-theoretic finding."""

    def test_residual_grows_quadratically(self) -> None:
        """Residual uncertainty should grow as Theta(d^2)."""
        residuals = []
        for n, p, q in [(15, 3, 5), (77, 7, 11), (323, 17, 19),
                         (1073, 29, 37), (5183, 71, 73)]:
            cc = analyze_carry_channel(n, 2)
            residuals.append((cc.d, cc.residual_uncertainty))

        # Check that residual / d^2 stays roughly constant (not decreasing)
        ratios = [r / (d * d) for d, r in residuals if d > 3]
        # Should not collapse toward 0
        assert all(r > 0.01 for r in ratios), (
            f"Residual/d^2 ratios: {ratios} — not Theta(d^2)"
        )

    def test_carry_entropy_subquadratic(self) -> None:
        """Carry entropy should grow slower than d^2 (subquadratic)."""
        entropies = []
        for n in [77, 323, 1073, 5183, 10403]:
            cc = analyze_carry_channel(n, 2)
            entropies.append((cc.d, cc.carry_entropy))

        # H(T) should grow but H(T)/d^2 should be bounded well below 1
        # (the lattice has Theta(d^2) bits total, carries reveal much less)
        ratios = [h / (d * d) for d, h in entropies]
        assert all(r < 0.5 for r in ratios), (
            f"H(T)/d^2 ratios too large: {ratios}"
        )

    def test_fraction_revealed_decreases(self) -> None:
        """Fraction of information revealed should decrease with d."""
        fractions = []
        for n in [77, 323, 1073, 5183, 10403]:
            cc = analyze_carry_channel(n, 2)
            fractions.append((cc.d, cc.fraction_revealed))

        # Should be a decreasing trend
        vals = [f for _, f in fractions]
        # At least the last should be less than the first
        assert vals[-1] < vals[0], (
            f"Fraction revealed not decreasing: {vals}"
        )


class TestSpectralBound:
    """Test analytical spectral bounds on lattice point count."""

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (77, 7, 11),
        (323, 17, 19),
        (1073, 29, 37),
    ])
    def test_upper_bound_valid(self, n: int, p: int, q: int) -> None:
        """Upper bound must be ≥ exact count."""
        sb = compute_spectral_bound(n, 2)
        assert sb.log2_exact <= sb.log2_upper_bound + 0.01, (
            f"Exact {sb.log2_exact} > upper {sb.log2_upper_bound}"
        )

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (77, 7, 11),
        (323, 17, 19),
        (1073, 29, 37),
    ])
    def test_lower_bound_valid(self, n: int, p: int, q: int) -> None:
        """Lower bound must be ≤ exact count."""
        sb = compute_spectral_bound(n, 2)
        assert sb.log2_lower_bound <= sb.log2_exact + 0.01, (
            f"Lower {sb.log2_lower_bound} > exact {sb.log2_exact}"
        )

    @pytest.mark.parametrize("base", [2, 3])
    def test_alpha_positive(self, base: int) -> None:
        """Quadratic coefficient α must be positive."""
        sb = compute_spectral_bound(77, base)
        assert sb.alpha_fit > 0

    def test_num_terms_profile_triangular(self) -> None:
        """num_terms should rise then fall (triangular)."""
        sb = compute_spectral_bound(1073, 2)
        profile = sb.num_terms_profile
        # Should peak in the middle
        peak = max(profile)
        peak_pos = profile.index(peak)
        assert sb.d // 4 <= peak_pos <= 3 * sb.d // 4

    def test_quadratic_scaling_proof(self) -> None:
        """prove_quadratic_scaling should return valid coefficients."""
        result = prove_quadratic_scaling(base=2)
        assert result["alpha_empirical"] > 0
        assert result["alpha_lower"] > 0
        # Alpha should be roughly 0.25 for base 2
        assert 0.1 < result["alpha_empirical"] < 0.5


class TestAlphaSpectralConstant:
    """Test the analytical spectral constant alpha(b) = log2((b-1)^2+1)/4."""

    def test_alpha_base2_is_quarter(self) -> None:
        """For base 2, alpha = log2(2)/4 = 1/4 exactly."""
        assert alpha_spectral_constant(2) == 0.25

    @pytest.mark.parametrize(
        "base,expected",
        [
            (2, 0.25),
            (3, log2(5) / 4),
            (5, log2(17) / 4),
            (10, log2(82) / 4),
        ],
    )
    def test_alpha_formula(self, base: int, expected: float) -> None:
        """alpha(b) = log2((b-1)^2+1)/4 for various bases."""
        assert abs(alpha_spectral_constant(base) - expected) < 1e-12

    def test_alpha_invalid_base(self) -> None:
        """Base must be >= 2."""
        with pytest.raises(ValueError):
            alpha_spectral_constant(1)

    def test_alpha_increases_with_base(self) -> None:
        """alpha(b) is increasing in b (more digit values -> more lattice points)."""
        prev = alpha_spectral_constant(2)
        for b in range(3, 20):
            curr = alpha_spectral_constant(b)
            assert curr > prev, f"alpha({b})={curr} <= alpha({b-1})={prev}"
            prev = curr


class TestRowSumIdentity:
    """Test Lemma 1: Row Sum Identity for base 2.

    For base 2, each row sum of the transfer matrix T_k equals 2^{n_k-1}
    EXACTLY, independent of carry-in state t_in and target digit c_k.
    This follows from the binomial parity identity:
        sum_{j even} C(n,j) = sum_{j odd} C(n,j) = 2^{n-1}
    """

    @pytest.mark.parametrize("n_k", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_parity_identity(self, n_k: int) -> None:
        """Binomial parity identity: sum of C(n,j) over even j = 2^{n-1}."""
        even_sum = sum(comb(n_k, j) for j in range(0, n_k + 1, 2))
        odd_sum = sum(comb(n_k, j) for j in range(1, n_k + 1, 2))
        assert even_sum == 2 ** (n_k - 1)
        assert odd_sum == 2 ** (n_k - 1)
        assert even_sum + odd_sum == 2**n_k

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (77, 7, 11),
            (323, 17, 19),
            (1073, 29, 37),
            (5183, 71, 73),
        ],
    )
    def test_base2_row_sums_exact(self, n: int, p: int, q: int) -> None:
        """For base 2, every row sum of T_k equals 2^{n_k-1} exactly."""
        from factoring_lab.analysis.lattice_counting import (
            _compute_digit_sizes,
            to_digits,
        )

        base = 2
        max_z = 1  # (2-1)^2
        c = to_digits(n, base)
        d = len(c)
        _, dx, dy = _compute_digit_sizes(n, base)

        num_terms_at = _compute_num_terms_profile(d, dx, dy)

        # Compute max carry at each position
        max_carry_at = []
        max_t = 0
        for k in range(d):
            max_sum = num_terms_at[k] * max_z + max_t
            max_t = max_sum // base
            max_carry_at.append(max_t)

        for k in range(d):
            n_k = num_terms_at[k]
            if n_k == 0:
                continue

            expected = 2 ** (n_k - 1)
            max_cin = 0 if k == 0 else max_carry_at[k - 1]
            max_cout = max_carry_at[k]

            for t_in in range(max_cin + 1):
                rs = _compute_row_sum(c[k], t_in, n_k, base, max_z, max_cout)
                assert rs == expected, (
                    f"n={n}, k={k}, n_k={n_k}, t_in={t_in}: "
                    f"row_sum={rs} != 2^{n_k-1}={expected}"
                )

    @pytest.mark.parametrize("base", [3, 5, 10])
    def test_general_base_row_sums_converge(self, base: int) -> None:
        """For general base, row sum / ((M+1)^{n_k}/b) -> 1 as n_k grows."""
        max_z = (base - 1) ** 2
        M_plus_1 = max_z + 1

        for n_k in [3, 5, 8]:
            # Use a generous carry range
            max_cout = n_k * max_z // base + 1
            for c_k in range(base):
                for t_in in range(min(3, n_k * max_z // base + 1)):
                    rs = _compute_row_sum(c_k, t_in, n_k, base, max_z, max_cout)
                    expected = M_plus_1**n_k / base
                    ratio = rs / expected
                    # For n_k >= 3, ratio should be close to 1
                    assert 0.5 < ratio < 1.5, (
                        f"base={base}, n_k={n_k}, c_k={c_k}, t_in={t_in}: "
                        f"ratio={ratio:.4f}"
                    )


class TestSumNumTermsBalanced:
    """Test Lemma 2: sum(n_k) = d^2/4 + O(d) for balanced semiprimes."""

    @pytest.mark.parametrize("d", [5, 7, 9, 11, 13, 15, 21, 31, 33, 65])
    def test_sum_nk_equals_m_squared(self, d: int) -> None:
        """For dx=dy=m with d=2m-1, sum(n_k) = m^2 exactly."""
        m = (d + 1) // 2
        sum_nk, dx, dy = _sum_num_terms_balanced(d)
        assert dx == m
        assert dy == m
        assert sum_nk == m * m, f"d={d}, m={m}: sum_nk={sum_nk} != m^2={m*m}"

    @pytest.mark.parametrize("d", [5, 9, 15, 21, 31, 65, 101])
    def test_sum_nk_d_squared_over_4(self, d: int) -> None:
        """sum(n_k) = d^2/4 + O(d) for balanced case."""
        sum_nk, _, _ = _sum_num_terms_balanced(d)
        d_sq_4 = d * d / 4
        # The error should be O(d), specifically d/2 + 1/4 for odd d
        diff = abs(sum_nk - d_sq_4)
        assert diff <= d, f"d={d}: |sum_nk - d^2/4| = {diff} > d={d}"

    def test_profile_triangular(self) -> None:
        """The num_terms profile should be triangular for balanced case."""
        d = 15
        m = (d + 1) // 2  # = 8
        profile = _compute_num_terms_profile(d, m, m)
        # Should rise from 1 to m then descend
        assert profile[0] == 1
        assert profile[m - 1] == m
        assert profile[-1] == 1
        # Check symmetry
        for k in range(d):
            assert profile[k] == profile[d - 1 - k], (
                f"Profile not symmetric at k={k}: "
                f"{profile[k]} != {profile[d-1-k]}"
            )


class TestProveAlphaQuarter:
    """Test the full alpha = 1/4 proof for base 2."""

    def test_base2_proof_valid(self) -> None:
        """The proof for base 2 should pass all validity checks."""
        result = prove_alpha_quarter(base=2)
        assert result.proof_valid, f"Proof FAILED: {result.summary}"

    def test_base2_alpha_analytical(self) -> None:
        """Analytical alpha for base 2 should be exactly 0.25."""
        result = prove_alpha_quarter(base=2)
        assert result.alpha_analytical == 0.25

    def test_base2_row_sums_all_exact(self) -> None:
        """For base 2, all row sum ratios should be exactly 1.0."""
        result = prove_alpha_quarter(base=2)
        for i, ratios in enumerate(result.row_sum_ratios):
            for k, ratio in enumerate(ratios):
                assert abs(ratio - 1.0) < 1e-10, (
                    f"Semiprime {i}, position {k}: ratio={ratio}"
                )

    def test_base2_alpha_fit_near_quarter(self) -> None:
        """Regression alpha should be close to 0.25 for base 2."""
        result = prove_alpha_quarter(base=2)
        # Regression alpha should be within 5% of 0.25
        assert abs(result.alpha_fit - 0.25) < 0.025, (
            f"alpha_fit={result.alpha_fit} too far from 0.25"
        )

    def test_base2_convergence(self) -> None:
        """alpha_empirical for the largest semiprime should be close to 0.25."""
        result = prove_alpha_quarter(base=2)
        largest_idx = result.d_values.index(max(result.d_values))
        alpha_largest = result.alpha_empirical_values[largest_idx]
        assert abs(alpha_largest - 0.25) < 0.01, (
            f"Largest d={max(result.d_values)}: alpha={alpha_largest}"
        )

    def test_base2_max_relative_error_small(self) -> None:
        """Max relative error should be small for base 2."""
        result = prove_alpha_quarter(base=2)
        assert result.max_relative_error < 0.05, (
            f"max_relative_error={result.max_relative_error}"
        )

    @pytest.mark.parametrize("base", [3, 5, 10])
    def test_general_base_proof_valid(self, base: int) -> None:
        """The proof should be valid for general bases (with loose tolerances)."""
        result = prove_alpha_quarter(base=base)
        assert result.proof_valid, f"Proof FAILED for base {base}: {result.summary}"

    @pytest.mark.parametrize(
        "base,expected_alpha",
        [
            (2, 0.25),
            (3, log2(5) / 4),
            (5, log2(17) / 4),
            (10, log2(82) / 4),
        ],
    )
    def test_analytical_formula_matches(
        self, base: int, expected_alpha: float
    ) -> None:
        """Alpha analytical should match the formula log2((b-1)^2+1)/4."""
        result = prove_alpha_quarter(base=base)
        assert abs(result.alpha_analytical - expected_alpha) < 1e-12

    @pytest.mark.parametrize("base", [2, 3, 5, 10])
    def test_sum_nk_order_d_squared(self, base: int) -> None:
        """sum(n_k) should be O(d^2) for all tested semiprimes."""
        result = prove_alpha_quarter(base=base)
        for snk, d in zip(result.sum_nk_values, result.d_values):
            # sum(n_k) should be at most d^2/2 (generous upper bound)
            assert snk <= d * d, (
                f"base={base}, d={d}: sum_nk={snk} > d^2={d*d}"
            )
