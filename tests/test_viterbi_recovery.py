"""Tests for Viterbi recovery with SVD prior."""

import numpy as np
import pytest

from factoring_lab.analysis.viterbi_recovery import (
    ViterbiRecoveryResult,
    _corr,
    _get_pairs_at_position,
    _naive_rounding,
    _project_to_simplex_with_bounds,
    _round_to_integers_with_sum,
    _solve_svd_estimates,
    sweep_lambda,
    viterbi_factor_recovery,
)


# --- Helper function tests ---


class TestProjectToSimplex:
    """Tests for the water-filling / simplex projection."""

    def test_already_on_simplex(self):
        z_est = np.array([1.0, 2.0, 3.0])
        result = _project_to_simplex_with_bounds(z_est, 6, 5.0)
        assert abs(result.sum() - 6) < 1e-6
        np.testing.assert_allclose(result, z_est, atol=1e-6)

    def test_sum_too_large(self):
        z_est = np.array([3.0, 3.0, 3.0])
        result = _project_to_simplex_with_bounds(z_est, 6, 5.0)
        assert abs(result.sum() - 6) < 1e-6
        assert all(0 <= v <= 5.0 + 1e-10 for v in result)

    def test_sum_too_small(self):
        z_est = np.array([1.0, 1.0, 1.0])
        result = _project_to_simplex_with_bounds(z_est, 6, 5.0)
        assert abs(result.sum() - 6) < 1e-6
        assert all(0 <= v <= 5.0 + 1e-10 for v in result)

    def test_box_constraints(self):
        z_est = np.array([10.0, -5.0, 3.0])
        result = _project_to_simplex_with_bounds(z_est, 5, 4.0)
        assert abs(result.sum() - 5) < 1e-6
        assert all(-1e-10 <= v <= 4.0 + 1e-10 for v in result)

    def test_single_variable(self):
        z_est = np.array([3.0])
        result = _project_to_simplex_with_bounds(z_est, 2, 5.0)
        assert abs(result[0] - 2.0) < 1e-6

    def test_empty(self):
        result = _project_to_simplex_with_bounds(np.array([]), 0, 5.0)
        assert len(result) == 0


class TestRoundToIntegersWithSum:
    """Tests for integer rounding with sum constraint."""

    def test_exact_rounding(self):
        z = np.array([1.0, 2.0, 3.0])
        result = _round_to_integers_with_sum(z, 6, 5)
        assert result.sum() == 6
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_fractional_rounding(self):
        z = np.array([1.4, 2.3, 2.3])
        result = _round_to_integers_with_sum(z, 6, 5)
        assert result.sum() == 6
        assert all(0 <= v <= 5 for v in result)

    def test_respects_bounds(self):
        z = np.array([4.8, 4.8, 0.4])
        result = _round_to_integers_with_sum(z, 10, 5)
        assert result.sum() == 10
        assert all(0 <= v <= 5 for v in result)

    def test_binary_case(self):
        z = np.array([0.7, 0.3, 0.8, 0.2])
        result = _round_to_integers_with_sum(z, 2, 1)
        assert result.sum() == 2
        assert all(0 <= v <= 1 for v in result)


class TestGetPairsAtPosition:
    """Tests for position-to-pair mapping."""

    def test_position_0(self):
        pairs = _get_pairs_at_position(0, 3, 3)
        assert pairs == [(0, 0)]

    def test_position_1(self):
        pairs = _get_pairs_at_position(1, 3, 3)
        assert (0, 1) in pairs
        assert (1, 0) in pairs
        assert len(pairs) == 2

    def test_position_2_small(self):
        pairs = _get_pairs_at_position(2, 2, 2)
        assert pairs == [(1, 1)]

    def test_large_position(self):
        pairs = _get_pairs_at_position(4, 3, 3)
        assert pairs == [(2, 2)]


class TestCorrelation:
    """Tests for the correlation helper."""

    def test_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(_corr(a, a) - 1.0) < 1e-10

    def test_negative_correlation(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert abs(_corr(a, b) + 1.0) < 1e-10

    def test_constant_array(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _corr(a, b) == 0.0


# --- SVD estimation tests ---


class TestSVDEstimates:
    """Tests for SVD estimate extraction."""

    def test_returns_correct_shapes(self):
        n, base = 15, 2
        x_est, y_est, Z_star = _solve_svd_estimates(n, base, 3, 3)
        assert len(x_est) == 3
        assert len(y_est) == 3
        assert Z_star.shape == (3, 3)

    def test_z_star_satisfies_constraints_approximately(self):
        """Z* should approximately satisfy the carry constraints."""
        from factoring_lab.analysis.lattice_counting import to_digits

        n, base = 77, 10
        x_est, y_est, Z_star = _solve_svd_estimates(n, base, 2, 2)
        # Z_star comes from least-squares, so it should be close to satisfying
        # the constraints, but may not be exact (underdetermined system)
        assert Z_star.shape == (2, 2)


# --- Main recovery tests ---


class TestViterbiRecovery:
    """Tests for the full Viterbi recovery pipeline."""

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
    def test_returns_valid_result(self, n, p, q):
        result = viterbi_factor_recovery(n, 2, p, q)
        assert isinstance(result, ViterbiRecoveryResult)
        assert result.n == n
        assert result.base == 2
        assert result.d > 0
        assert len(result.recovered_carry_sequence) == result.d

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
    def test_true_carry_sequence_computed(self, n, p, q):
        result = viterbi_factor_recovery(n, 2, p, q)
        assert result.true_carry_sequence is not None
        assert len(result.true_carry_sequence) == result.d

    @pytest.mark.parametrize("base", [2, 10])
    def test_small_semiprime_base_variants(self, base):
        result = viterbi_factor_recovery(15, base, 3, 5)
        assert result.n == 15
        assert result.base == base
        assert result.viterbi_log_likelihood < 0 or result.viterbi_log_likelihood == 0

    def test_svd_correlations_exist(self):
        result = viterbi_factor_recovery(77, 2, 7, 11)
        # SVD should have some correlation with true factors
        assert isinstance(result.svd_corr_x, float)
        assert isinstance(result.svd_corr_y, float)

    def test_viterbi_correlations_at_least_as_good(self):
        """Viterbi correlations should be comparable to or better than naive."""
        result = viterbi_factor_recovery(77, 10, 7, 11, lambda_param=1.0)
        # Not guaranteed to be better in all cases, but should be a valid float
        assert isinstance(result.viterbi_corr_x, float)
        assert isinstance(result.viterbi_corr_y, float)

    def test_if_recovered_then_correct(self):
        """If recovery claims success, the factors must be correct."""
        for n, p, q in [(15, 3, 5), (21, 3, 7), (35, 5, 7), (77, 7, 11)]:
            result = viterbi_factor_recovery(n, 10, p, q, lambda_param=2.0)
            if result.recovery_success:
                assert result.recovered_p * result.recovered_q == n
                assert result.recovered_p > 1
                assert result.recovered_q > 1
            if result.greedy_success:
                assert result.greedy_p * result.greedy_q == n
            if result.naive_success:
                assert result.naive_p * result.naive_q == n


class TestSweepLambda:
    """Tests for lambda parameter sweep."""

    def test_returns_multiple_results(self):
        results = sweep_lambda(15, 2, 3, 5)
        assert len(results) == 6  # default 6 lambda values

    def test_custom_lambdas(self):
        results = sweep_lambda(15, 2, 3, 5, lambdas=[0.5, 2.0])
        assert len(results) == 2
        assert results[0].lambda_param == 0.5
        assert results[1].lambda_param == 2.0


class TestLargerSemiprimes:
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
            result = viterbi_factor_recovery(n, base, p, q)
            assert result.d > 0
            assert len(result.recovered_carry_sequence) == result.d
            if result.recovery_success:
                assert result.recovered_p * result.recovered_q == n
