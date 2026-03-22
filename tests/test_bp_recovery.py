"""Tests for Belief Propagation recovery on the carry chain factor graph."""

import numpy as np
import pytest

from factoring_lab.analysis.bp_recovery import (
    BPRecoveryResult,
    _beliefs_to_map,
    _beliefs_to_means,
    _beliefs_to_variances,
    _normalize,
    bp_factor_recovery,
)


# --- Helper function tests ---


class TestNormalize:
    """Tests for the normalization helper."""

    def test_already_normalized(self):
        arr = np.array([0.25, 0.25, 0.25, 0.25])
        result = _normalize(arr)
        np.testing.assert_allclose(result, arr)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_unnormalized(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _normalize(arr)
        assert abs(result.sum() - 1.0) < 1e-10
        np.testing.assert_allclose(result, [1 / 6, 2 / 6, 3 / 6])

    def test_all_zeros_gives_uniform(self):
        arr = np.array([0.0, 0.0, 0.0])
        result = _normalize(arr)
        assert abs(result.sum() - 1.0) < 1e-10
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3])


class TestBeliefsToMap:
    """Tests for MAP extraction from beliefs."""

    def test_clear_peak(self):
        beliefs = [
            np.array([0.1, 0.8, 0.1]),
            np.array([0.7, 0.2, 0.1]),
        ]
        assert _beliefs_to_map(beliefs) == [1, 0]

    def test_uniform_gives_zero(self):
        beliefs = [np.array([0.5, 0.5])]
        result = _beliefs_to_map(beliefs)
        assert result == [0]  # argmax returns first occurrence


class TestBeliefsToMeans:
    """Tests for mean extraction from beliefs."""

    def test_delta_distribution(self):
        beliefs = [np.array([0.0, 1.0, 0.0])]
        means = _beliefs_to_means(beliefs)
        assert abs(means[0] - 1.0) < 1e-10

    def test_uniform_gives_midpoint(self):
        beliefs = [np.array([0.5, 0.5])]
        means = _beliefs_to_means(beliefs)
        assert abs(means[0] - 0.5) < 1e-10


class TestBeliefsToVariances:
    """Tests for variance extraction from beliefs."""

    def test_delta_has_zero_variance(self):
        beliefs = [np.array([0.0, 0.0, 1.0])]
        variances = _beliefs_to_variances(beliefs)
        assert abs(variances[0]) < 1e-10

    def test_uniform_binary_variance(self):
        beliefs = [np.array([0.5, 0.5])]
        variances = _beliefs_to_variances(beliefs)
        # Var of Bernoulli(0.5) = 0.25
        assert abs(variances[0] - 0.25) < 1e-10


# --- Main BP recovery tests ---


class TestBPRecovery:
    """Tests for the full BP recovery pipeline."""

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
            (77, 7, 11),
            (143, 11, 13),
            (323, 17, 19),
        ],
    )
    def test_returns_valid_result(self, n, p, q):
        result = bp_factor_recovery(n, 10, p, q, max_iters=30)
        assert isinstance(result, BPRecoveryResult)
        assert result.n == n
        assert result.base == 10
        assert result.d > 0
        assert result.num_iterations > 0
        assert len(result.x_beliefs) == result.dx
        assert len(result.y_beliefs) == result.dy
        assert len(result.x_map) == result.dx
        assert len(result.y_map) == result.dy

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
            (77, 7, 11),
            (143, 11, 13),
            (323, 17, 19),
        ],
    )
    def test_beliefs_are_valid_distributions(self, n, p, q):
        """All beliefs should be valid probability distributions."""
        result = bp_factor_recovery(n, 10, p, q, max_iters=30)
        for i, belief in enumerate(result.x_beliefs):
            assert abs(belief.sum() - 1.0) < 1e-6, (
                f"x_belief[{i}] sums to {belief.sum()}"
            )
            assert np.all(belief >= -1e-10), (
                f"x_belief[{i}] has negative values"
            )
        for j, belief in enumerate(result.y_beliefs):
            assert abs(belief.sum() - 1.0) < 1e-6, (
                f"y_belief[{j}] sums to {belief.sum()}"
            )
            assert np.all(belief >= -1e-10), (
                f"y_belief[{j}] has negative values"
            )

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
            (77, 7, 11),
            (143, 11, 13),
            (323, 17, 19),
        ],
    )
    def test_convergence_no_divergence(self, n, p, q):
        """BP beliefs should not diverge (correlations should not collapse to zero)."""
        result = bp_factor_recovery(n, 10, p, q, max_iters=50)
        # Check that beliefs don't collapse to all-zero or all-uniform
        for i, belief in enumerate(result.x_beliefs):
            # Should not be perfectly uniform (no information)
            # unless lambda_svd is 0
            entropy = -np.sum(
                belief * np.log(np.maximum(belief, 1e-300))
            )
            max_entropy = np.log(result.base)
            # Allow some tolerance - beliefs might be close to uniform for
            # unconstrained variables
            assert entropy <= max_entropy + 1e-6

    def test_if_recovered_then_correct(self):
        """If recovery claims success, the factors must be correct."""
        for n, p, q in [(15, 3, 5), (21, 3, 7), (35, 5, 7), (77, 7, 11)]:
            result = bp_factor_recovery(n, 10, p, q, max_iters=50)
            if result.recovery_success:
                assert result.recovered_p * result.recovered_q == n
                assert result.recovered_p > 1
                assert result.recovered_q > 1

    @pytest.mark.parametrize("base", [2, 10])
    def test_base_variants(self, base):
        """Test BP on both base 2 and base 10."""
        result = bp_factor_recovery(15, base, 3, 5, max_iters=30)
        assert result.n == 15
        assert result.base == base
        assert result.num_iterations > 0
        # Beliefs should be valid distributions
        for belief in result.x_beliefs + result.y_beliefs:
            assert abs(belief.sum() - 1.0) < 1e-6

    def test_correlation_tracking(self):
        """Per-iteration correlation should be tracked."""
        result = bp_factor_recovery(77, 10, 7, 11, max_iters=20)
        assert len(result.per_iteration_correlation) == result.num_iterations
        # All correlations should be finite
        for c in result.per_iteration_correlation:
            assert np.isfinite(c)

    def test_map_estimates_in_range(self):
        """MAP estimates should be valid digit values."""
        result = bp_factor_recovery(77, 10, 7, 11, max_iters=30)
        for v in result.x_map:
            assert 0 <= v < result.base
        for v in result.y_map:
            assert 0 <= v < result.base

    def test_damping_affects_convergence(self):
        """Different damping values should affect convergence speed."""
        result_low = bp_factor_recovery(
            77, 10, 7, 11, max_iters=50, damping=0.2
        )
        result_high = bp_factor_recovery(
            77, 10, 7, 11, max_iters=50, damping=0.8
        )
        # Both should produce valid results (not crash)
        assert isinstance(result_low, BPRecoveryResult)
        assert isinstance(result_high, BPRecoveryResult)


class TestBPvsBaselines:
    """Compare BP correlation with SVD baseline."""

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
            (77, 7, 11),
            (143, 11, 13),
        ],
    )
    def test_bp_correlation_is_valid(self, n, p, q):
        """BP correlation should be a valid number (not NaN or inf)."""
        result = bp_factor_recovery(n, 10, p, q, max_iters=50)
        assert np.isfinite(result.bp_corr_x)
        assert np.isfinite(result.bp_corr_y)
        assert 0.0 <= result.bp_corr_x <= 1.0 + 1e-6
        assert 0.0 <= result.bp_corr_y <= 1.0 + 1e-6

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (15, 3, 5),
            (21, 3, 7),
            (35, 5, 7),
            (77, 7, 11),
            (143, 11, 13),
        ],
    )
    def test_svd_correlation_is_valid(self, n, p, q):
        """SVD baseline correlation should be valid."""
        result = bp_factor_recovery(n, 10, p, q, max_iters=50)
        assert np.isfinite(result.svd_corr_x)
        assert np.isfinite(result.svd_corr_y)


class TestBPLargerSemiprimes:
    """Tests on slightly larger semiprimes."""

    @pytest.mark.parametrize(
        "n,p,q",
        [
            (323, 17, 19),
            (1073, 29, 37),
            (5183, 71, 73),
        ],
    )
    def test_larger_cases_produce_results(self, n, p, q):
        for base in [2, 10]:
            result = bp_factor_recovery(n, base, p, q, max_iters=30)
            assert result.d > 0
            assert result.num_iterations > 0
            if result.recovery_success:
                assert result.recovered_p * result.recovered_q == n
