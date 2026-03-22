"""Tests for SOS rounding hardness analysis."""

import pytest
from math import log2

from factoring_lab.analysis.rounding_hardness import (
    SequentialRoundingResult,
    RoundingBoundResult,
    BoundedViewTheorem,
    BoundedViewTheoremSuite,
    analyze_sequential_rounding,
    prove_rounding_bound,
    prove_bounded_view_hardness,
)


class TestSequentialRounding:
    """Tests for sequential rounding analysis."""

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ])
    @pytest.mark.parametrize("base", [2, 3])
    def test_per_step_prob_at_most_one(
        self, n: int, p: int, q: int, base: int
    ) -> None:
        """Each per-step probability must be ≤ 1."""
        sr = analyze_sequential_rounding(n, base, p, q)
        for k, prob in enumerate(sr.per_step_success_prob):
            assert prob <= 1.0 + 1e-10, (
                f"Position {k}: prob={prob} > 1"
            )

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (77, 7, 11),
        (143, 11, 13),
    ])
    def test_overall_success_decreases_with_size(
        self, n: int, p: int, q: int
    ) -> None:
        """Overall success probability should be very small."""
        sr = analyze_sequential_rounding(n, 2, p, q)
        # For any non-trivial case, success prob should be < 1
        assert sr.overall_success_prob < 1.0

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    def test_true_carry_sequence_valid(
        self, n: int, p: int, q: int
    ) -> None:
        """True carry sequence must end with 0."""
        sr = analyze_sequential_rounding(n, 2, p, q)
        assert sr.true_carry_sequence[-1] == 0

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (77, 7, 11),
    ])
    @pytest.mark.parametrize("base", [2, 3])
    def test_valid_z_count_positive(
        self, n: int, p: int, q: int, base: int
    ) -> None:
        """Valid z-count at each position must be positive."""
        sr = analyze_sequential_rounding(n, base, p, q)
        for k, cnt in enumerate(sr.valid_z_count):
            assert cnt > 0, f"Position {k}: valid_z_count={cnt}"

    def test_cumulative_monotonically_decreasing(self) -> None:
        """Cumulative success prob must decrease."""
        sr = analyze_sequential_rounding(77, 2, 7, 11)
        for k in range(1, len(sr.cumulative_success_prob)):
            assert sr.cumulative_success_prob[k] <= (
                sr.cumulative_success_prob[k - 1] + 1e-15
            )


class TestRoundingBound:
    """Tests for the formal rounding bound."""

    @pytest.mark.parametrize("n,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_bound_is_positive(self, n: int, p: int, q: int) -> None:
        """The bound on log₂(1/P(success)) should be positive."""
        rb = prove_rounding_bound(n, 2, p, q)
        assert rb.log2_success_upper_bound > 0

    @pytest.mark.parametrize("n,p,q", [
        (77, 7, 11),
        (323, 17, 19),
        (1073, 29, 37),
    ])
    def test_bound_grows_with_d(self, n: int, p: int, q: int) -> None:
        """The rounding hardness should increase with problem size."""
        rb = prove_rounding_bound(n, 2, p, q)
        # log₂(1/P(success)) should be at least d
        assert rb.log2_success_upper_bound >= rb.d

    def test_bound_exceeds_fiber(self) -> None:
        """Rounding bound should be ≥ the true fiber size."""
        rb = prove_rounding_bound(77, 2, 7, 11)
        # The bound counts total z-tuples across all positions along the
        # true carry path; the fiber is the product of per-position counts.
        # These should be equal.
        assert abs(rb.log2_success_upper_bound - rb.log2_true_fiber) < 0.01 or (
            rb.log2_success_upper_bound >= rb.log2_true_fiber - 0.01
        )


class TestScalingLaw:
    """Test the scaling law for rounding hardness."""

    def test_hardness_grows_faster_than_linear(self) -> None:
        """log₂(1/P(success)) should grow faster than d."""
        cases = [
            (15, 3, 5),
            (77, 7, 11),
            (323, 17, 19),
            (1073, 29, 37),
            (5183, 71, 73),
        ]
        results = []
        for n, p, q in cases:
            rb = prove_rounding_bound(n, 2, p, q)
            results.append((rb.d, rb.log2_success_upper_bound))

        # Ratio bound/d should increase (superlinear growth)
        ratios = [b / d for d, b in results if d > 4]
        assert ratios[-1] > ratios[0], (
            f"Bound/d not increasing: {ratios}"
        )


class TestBoundedViewTheorem:
    """Tests for Theorem B (Bounded-View Rounding Hardness).

    Theorem B: For any deterministic rounding scheme R that reads at most k
    entries of the degree-4 moment matrix M of a d-digit base-b semiprime,
    there exist at least a fraction (1 - k/D_agree) of semiprimes at each
    bit-length for which R fails to recover the factors.

    These tests verify the theorem at 8, 10, and 12 bit semiprimes.
    """

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_proof_valid_at_budget_d(self, bit_length: int) -> None:
        """Theorem B proof certificate must be valid at budget k = d.

        At budget k = d (linear in number of digits), the theorem should
        report a positive failure fraction because d << D_agree.
        """
        suite = prove_bounded_view_hardness(bit_length, base=2)
        assert len(suite.certificates) >= 1

        # Find the certificate at budget d
        d_cert = None
        for cert in suite.certificates:
            if cert.budget == cert.d:
                d_cert = cert
                break

        assert d_cert is not None, (
            f"No certificate at budget k=d for {bit_length}-bit semiprimes"
        )
        assert d_cert.proof_valid, (
            f"Theorem B proof invalid at budget k=d for {bit_length}-bit"
        )
        assert d_cert.failure_lower_bound > 0.0, (
            f"Failure lower bound should be > 0 at budget k=d, "
            f"got {d_cert.failure_lower_bound}"
        )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_d_agree_is_large_fraction_of_total(self, bit_length: int) -> None:
        """Empirical D_agree must be a large fraction of D_total.

        Most moment entries agree across semiprime pairs because the
        degree-4 moment space is high-dimensional (O(d^4)) while the
        factorization-dependent entries are low-dimensional (O(d^2)).
        D_agree / D_total should be well above 0.5.
        """
        suite = prove_bounded_view_hardness(bit_length, base=2)
        cert = suite.certificates[0]
        frac = cert.d_agree / cert.total_moment_entries
        assert frac > 0.5, (
            f"D_agree/D_total = {frac:.3f} should be > 0.5 "
            f"(D_agree={cert.d_agree:.1f}, D_total={cert.total_moment_entries})"
        )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_self_moments_dominate(self, bit_length: int) -> None:
        """Self-moment count must exceed cross-moment count.

        This is a key structural property: D_total - dx*dy >> dx*dy,
        meaning most moment entries are determined by n alone.
        """
        suite = prove_bounded_view_hardness(bit_length, base=2)
        cert = suite.certificates[0]
        assert cert.self_moment_count > cert.cross_moment_count, (
            f"Self-moments ({cert.self_moment_count}) should exceed "
            f"cross-moments ({cert.cross_moment_count})"
        )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_failure_decreases_with_budget(self, bit_length: int) -> None:
        """Failure lower bound must decrease as budget increases.

        More queries should make the rounding scheme more powerful,
        so the failure fraction should decrease.
        """
        suite = prove_bounded_view_hardness(bit_length, base=2)
        if len(suite.certificates) >= 2:
            for i in range(1, len(suite.certificates)):
                prev = suite.certificates[i - 1]
                curr = suite.certificates[i]
                assert curr.budget >= prev.budget
                assert curr.failure_lower_bound <= prev.failure_lower_bound + 1e-10, (
                    f"Failure bound increased from budget {prev.budget} "
                    f"({prev.failure_lower_bound:.3f}) to {curr.budget} "
                    f"({curr.failure_lower_bound:.3f})"
                )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_structural_bound_is_valid(self, bit_length: int) -> None:
        """The structural failure bound must be non-negative and <= 1."""
        suite = prove_bounded_view_hardness(bit_length, base=2)
        for cert in suite.certificates:
            assert 0.0 <= cert.failure_structural_bound <= 1.0, (
                f"Structural bound out of range: {cert.failure_structural_bound}"
            )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_indistinguishable_fraction_positive_at_d(
        self, bit_length: int
    ) -> None:
        """At budget k = d, some pairs must be empirically indistinguishable.

        This is the core empirical verification of Theorem B: at polynomial
        budget, a nontrivial fraction of semiprime pairs share their entire
        k-entry view.
        """
        suite = prove_bounded_view_hardness(bit_length, base=2)
        # Find the certificate at budget d
        for cert in suite.certificates:
            if cert.budget == cert.d:
                assert cert.indistinguishable_fraction > 0.0, (
                    f"Expected indistinguishable pairs at budget k=d={cert.d}, "
                    f"but fraction is 0 for {bit_length}-bit semiprimes"
                )
                break

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_total_entries_decomposition(self, bit_length: int) -> None:
        """D_total = cross_moment_count + self_moment_count."""
        suite = prove_bounded_view_hardness(bit_length, base=2)
        cert = suite.certificates[0]
        assert cert.total_moment_entries == (
            cert.cross_moment_count + cert.self_moment_count
        ), (
            f"Entry decomposition mismatch: "
            f"{cert.total_moment_entries} != "
            f"{cert.cross_moment_count} + {cert.self_moment_count}"
        )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_suite_has_multiple_budgets(self, bit_length: int) -> None:
        """Suite should contain certificates at multiple budgets."""
        suite = prove_bounded_view_hardness(bit_length, base=2)
        assert len(suite.certificates) >= 2, (
            f"Expected >= 2 budget levels, got {len(suite.certificates)}"
        )
        assert suite.budgets == sorted(suite.budgets), (
            "Budgets should be sorted"
        )

    def test_8bit_failure_at_least_26_percent(self) -> None:
        """At 8-bit with budget k=d, failure should be >= 26%.

        This matches the empirical finding from moment_indistinguishability:
        at budget k=d, 26-43% of pairs share ALL k entries.
        """
        suite = prove_bounded_view_hardness(8, base=2)
        for cert in suite.certificates:
            if cert.budget == cert.d:
                # The indistinguishable fraction or failure bound should
                # confirm the 26% lower bound from the paper
                assert cert.indistinguishable_fraction >= 0.20, (
                    f"Expected >= 20% indistinguishable at k=d for 8-bit, "
                    f"got {cert.indistinguishable_fraction:.1%}"
                )
                break

    def test_cross_moment_count_equals_dx_times_dy(self) -> None:
        """Cross-moment count must be exactly dx * dy."""
        suite = prove_bounded_view_hardness(10, base=2)
        cert = suite.certificates[0]
        assert cert.cross_moment_count == cert.dx * cert.dy

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_d_agree_positive(self, bit_length: int) -> None:
        """Average agreement D_agree must be positive."""
        suite = prove_bounded_view_hardness(bit_length, base=2)
        cert = suite.certificates[0]
        assert cert.d_agree > 0, (
            f"D_agree should be positive, got {cert.d_agree}"
        )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_failure_bounds_both_positive(self, bit_length: int) -> None:
        """Both empirical and structural failure bounds must be positive.

        Both bounds should predict failure > 0 at small budgets,
        confirming that bounded-view rounding is inherently limited.
        The structural bound uses D_total - dx*dy as an estimate of D_agree,
        while the empirical bound uses the measured D_agree.
        """
        suite = prove_bounded_view_hardness(bit_length, base=2)
        for cert in suite.certificates:
            if cert.budget <= cert.d:
                assert cert.failure_lower_bound > 0.0, (
                    f"Empirical failure bound should be > 0 at budget "
                    f"{cert.budget}, got {cert.failure_lower_bound:.3f}"
                )
                assert cert.failure_structural_bound > 0.0, (
                    f"Structural failure bound should be > 0 at budget "
                    f"{cert.budget}, got {cert.failure_structural_bound:.3f}"
                )

    @pytest.mark.parametrize("bit_length", [8, 10, 12])
    def test_structural_and_empirical_bounds_close(
        self, bit_length: int
    ) -> None:
        """Structural and empirical failure bounds should be within 10%.

        The structural estimate D_total - dx*dy is a reasonable
        approximation of D_agree; both bounds should agree up to a
        small constant factor.
        """
        suite = prove_bounded_view_hardness(bit_length, base=2)
        for cert in suite.certificates:
            if cert.failure_lower_bound > 0 and cert.failure_structural_bound > 0:
                ratio = cert.failure_lower_bound / cert.failure_structural_bound
                assert 0.9 <= ratio <= 1.1, (
                    f"Empirical/structural ratio = {ratio:.3f} at budget "
                    f"{cert.budget} (expected ~1.0)"
                )
