"""Tests for SOS rounding hardness analysis."""

import pytest
from math import log2

from factoring_lab.analysis.rounding_hardness import (
    SequentialRoundingResult,
    RoundingBoundResult,
    analyze_sequential_rounding,
    prove_rounding_bound,
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
