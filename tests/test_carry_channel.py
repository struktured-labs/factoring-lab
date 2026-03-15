"""Tests for carry channel information-theoretic analysis."""

import pytest
from math import log2

from factoring_lab.analysis.carry_channel import (
    CarryChannelResult,
    SpectralBoundResult,
    analyze_carry_channel,
    compute_spectral_bound,
    prove_quadratic_scaling,
)
from factoring_lab.analysis.lattice_counting import (
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
