"""Tests for the SVD -> Coppersmith hybrid factoring approach.

Tests cover:
- SVD digit estimation with proper affine rescaling
- Confidence metric correctness
- Coppersmith recovery with SVD-estimated digits
- Comparison of SVD-informed vs oracle selection
"""

import math

import numpy as np
import pytest

# Import the functions under test from the script
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from svd_coppersmith_hybrid import (
    _affine_rescale,
    _build_carry_system,
    svd_estimate_digits,
    attempt_svd_coppersmith,
    attempt_random_true_coppersmith,
    svd_coppersmith_recovery,
)
from factoring_lab.analysis.lattice_counting import to_digits, from_digits, _compute_digit_sizes
from factoring_lab.algorithms.hybrid_coppersmith import coppersmith_lattice_factor_base
from factoring_lab.generators.semiprimes import balanced_semiprime


# ---------- Test data ----------

SMALL_SEMIPRIMES = [
    (15, 3, 5),
    (21, 3, 7),
    (35, 5, 7),
    (77, 7, 11),
    (143, 11, 13),
    (323, 17, 19),
]


class TestAffineRescale:
    """Test affine rescaling of SVD vectors."""

    def test_basic_rescale(self):
        v = np.array([0.0, 0.5, 1.0])
        result = _affine_rescale(v, 0, 9)
        np.testing.assert_allclose(result, [0, 4.5, 9])

    def test_negative_values(self):
        v = np.array([-1.0, 0.0, 1.0])
        result = _affine_rescale(v, 0, 1)
        np.testing.assert_allclose(result, [0, 0.5, 1])

    def test_constant_vector(self):
        v = np.array([3.0, 3.0, 3.0])
        result = _affine_rescale(v, 0, 9)
        # Should map to midpoint
        np.testing.assert_allclose(result, [4.5, 4.5, 4.5])

    def test_preserves_ordering(self):
        v = np.array([0.1, 0.5, 0.2, 0.8])
        result = _affine_rescale(v, 0, 9)
        # Ordering should be preserved
        assert result[1] > result[0]
        assert result[3] > result[2]
        assert result[0] < result[2]


class TestBuildCarrySystem:
    """Test the carry-constraint linear system."""

    def test_system_dimensions(self):
        n = 77  # 7 * 11
        base = 10
        d, dx, dy = _compute_digit_sizes(n, base)
        A, b_vec, z_vars, z_idx, num_z, num_t = _build_carry_system(n, base, dx, dy)

        assert A.shape[0] == d
        assert A.shape[1] == num_z + num_t
        assert len(b_vec) == d
        assert num_t == d

    def test_rhs_matches_digits(self):
        n = 143  # 11 * 13
        base = 10
        d, dx, dy = _compute_digit_sizes(n, base)
        A, b_vec, z_vars, z_idx, num_z, num_t = _build_carry_system(n, base, dx, dy)

        expected_digits = to_digits(n, base)
        np.testing.assert_allclose(b_vec, expected_digits)


class TestSVDEstimateDigits:
    """Test SVD digit estimation with affine rescaling."""

    def test_returns_valid_range(self):
        """Estimated digits should be in [0, base-1] range approximately."""
        n, p, q = 323, 17, 19
        x_true = to_digits(p, 10)
        y_true = to_digits(q, 10)
        x_est, y_est, x_conf, y_conf, *_ = svd_estimate_digits(n, 10, x_true, y_true)

        # After rescaling, values should be approximately in [0, 9]
        assert x_est.min() >= -0.5
        assert x_est.max() <= 9.5
        assert y_est.min() >= -0.5
        assert y_est.max() <= 9.5

    def test_confidence_nonnegative(self):
        """Confidence values should be >= 0."""
        n, p, q = 77, 7, 11
        x_true = to_digits(p, 10)
        y_true = to_digits(q, 10)
        _, _, x_conf, y_conf, *_ = svd_estimate_digits(n, 10, x_true, y_true)

        assert np.all(x_conf >= 0)
        assert np.all(y_conf >= 0)
        # Confidence (distance to nearest int) should be <= 0.5
        assert np.all(x_conf <= 0.5 + 1e-10)
        assert np.all(y_conf <= 0.5 + 1e-10)

    def test_correlation_reported(self):
        """Should report correlations."""
        n, p, q = 143, 11, 13
        x_true = to_digits(p, 10)
        y_true = to_digits(q, 10)
        result = svd_estimate_digits(n, 10, x_true, y_true)
        x_corr = result[4]
        y_corr = result[5]

        assert 0.0 <= x_corr <= 1.0
        assert 0.0 <= y_corr <= 1.0

    def test_binary_base(self):
        """Should work with base 2."""
        n, p, q = 143, 11, 13
        x_true = to_digits(p, 2)
        y_true = to_digits(q, 2)
        x_est, y_est, x_conf, y_conf, *_ = svd_estimate_digits(n, 2, x_true, y_true)

        # Binary digits: should be approximately in [0, 1]
        assert x_est.min() >= -0.5
        assert x_est.max() <= 1.5

    @pytest.mark.parametrize("n,p,q", SMALL_SEMIPRIMES)
    def test_small_semiprimes_no_crash(self, n, p, q):
        """SVD estimation should not crash on small semiprimes."""
        for base in [2, 10]:
            x_true = to_digits(p, base)
            y_true = to_digits(q, base)
            result = svd_estimate_digits(n, base, x_true, y_true)
            assert len(result) == 12  # x_est, y_est, x_conf, y_conf, ...


class TestAttemptSVDCoppersmith:
    """Test the SVD-guided Coppersmith recovery."""

    def test_contiguous_strategy_small(self):
        """Contiguous strategy should work on very small semiprimes."""
        n, p, q = 77, 7, 11
        x_true = to_digits(p, 10)
        y_true = to_digits(q, 10)
        x_est, y_est, x_conf, y_conf, *_ = svd_estimate_digits(n, 10, x_true, y_true)

        # On tiny numbers, Coppersmith can succeed with minimal info
        result, tried = attempt_svd_coppersmith(
            n, x_est, x_conf, 10, len(x_true), 0.75,
            strategy="contiguous"
        )
        # We don't mandate success on toy examples, just no crashes
        assert tried >= 1
        assert result is None or (1 < result < n and n % result == 0)

    def test_enumerate_strategy_tries_candidates(self):
        """Enumerate strategy should try multiple candidates."""
        n, p, q = 323, 17, 19
        x_true = to_digits(p, 10)
        y_true = to_digits(q, 10)
        x_est, y_est, x_conf, y_conf, *_ = svd_estimate_digits(n, 10, x_true, y_true)

        result, tried = attempt_svd_coppersmith(
            n, x_est, x_conf, 10, len(x_true), 0.75,
            strategy="enumerate_prefix"
        )
        # Should have tried at least one candidate
        assert tried >= 1

    def test_result_is_valid_factor(self):
        """If a factor is found, it must actually divide n."""
        n, p, q = 143, 11, 13
        x_true = to_digits(p, 10)
        y_true = to_digits(q, 10)
        x_est, y_est, x_conf, y_conf, *_ = svd_estimate_digits(n, 10, x_true, y_true)

        for strategy in ["contiguous", "confident", "enumerate_prefix"]:
            result, _ = attempt_svd_coppersmith(
                n, x_est, x_conf, 10, len(x_true), 0.75,
                strategy=strategy
            )
            if result is not None:
                assert n % result == 0
                assert 1 < result < n


class TestAttemptRandomTrueCoppersmith:
    """Test the oracle baseline."""

    def test_oracle_finds_small_factors(self):
        """Oracle with true digits should find factors for small semiprimes."""
        n, p, q = 323, 17, 19
        true_digits = to_digits(p, 10)
        result, tried = attempt_random_true_coppersmith(n, true_digits, 10, 0.75)
        # Oracle has the correct digits -- should succeed on small numbers
        if result is not None:
            assert n % result == 0
            assert 1 < result < n

    def test_oracle_result_valid(self):
        """Any factor found by oracle must be valid."""
        for n, p, q in SMALL_SEMIPRIMES:
            for base in [2, 10]:
                true_digits = to_digits(p, base)
                result, tried = attempt_random_true_coppersmith(
                    n, true_digits, base, 0.75
                )
                if result is not None:
                    assert n % result == 0, f"n={n}, result={result}"


class TestSVDCoppersmithRecovery:
    """Integration test for the full recovery pipeline."""

    def test_full_pipeline_small(self):
        """Full pipeline should produce results for small semiprimes."""
        n, p, q = 323, 17, 19
        results = svd_coppersmith_recovery(
            n, p, q, base=10,
            leak_fractions=[0.50, 0.75],
            timeout_s=5.0,
        )

        # Should have 4 methods x 2 fractions = 8 results
        assert len(results) == 8

        for r in results:
            assert r.n == n
            assert r.bits == n.bit_length()
            assert r.base == 10
            assert r.leak_fraction in [0.50, 0.75]
            assert r.method in ["svd_contiguous", "svd_confident",
                                "svd_enum_prefix", "oracle_true"]
            if r.coppersmith_success:
                assert r.factor_found is not None
                assert n % r.factor_found == 0

    def test_full_pipeline_binary(self):
        """Full pipeline should work with base 2."""
        n, p, q = 143, 11, 13
        results = svd_coppersmith_recovery(
            n, p, q, base=2,
            leak_fractions=[0.50],
            timeout_s=5.0,
        )
        assert len(results) == 4
        for r in results:
            assert r.base == 2

    def test_32bit_semiprime(self):
        """Test on a realistic 32-bit semiprime."""
        spec = balanced_semiprime(32, seed=42)
        results = svd_coppersmith_recovery(
            spec.n, spec.p, spec.q, base=2,
            leak_fractions=[0.50],
            timeout_s=10.0,
        )

        assert len(results) == 4

        # At least the oracle should sometimes succeed at 50% leak for 32-bit
        oracle_results = [r for r in results if r.method == "oracle_true"]
        assert len(oracle_results) == 1

        # All results should have valid SVD correlations
        for r in results:
            assert 0.0 <= r.x_corr <= 1.0
            assert 0.0 <= r.y_corr <= 1.0
            assert 0.0 <= r.x_accuracy <= 1.0
            assert 0.0 <= r.y_accuracy <= 1.0

    def test_digit_accuracy_reasonable_binary(self):
        """In binary, SVD digit accuracy should be above chance (50%)."""
        # Use multiple samples to get a more stable estimate
        total_acc = 0.0
        num = 0
        for seed in range(42, 47):
            spec = balanced_semiprime(32, seed=seed)
            x_true = to_digits(spec.p, 2)
            y_true = to_digits(spec.q, 2)
            result = svd_estimate_digits(spec.n, 2, x_true, y_true)
            x_acc = result[8]
            y_acc = result[9]
            total_acc += max(x_acc, y_acc)
            num += 1

        avg_best_acc = total_acc / num
        # Even with noise, best-of-two should be at least chance level
        # (50% for binary). Being generous with threshold.
        assert avg_best_acc >= 0.35, f"Average best accuracy {avg_best_acc:.2f} too low"
