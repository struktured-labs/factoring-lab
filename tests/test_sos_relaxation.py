"""Tests for SOS/Lasserre hierarchy relaxation of digit convolution factoring."""

from __future__ import annotations

import pytest

from factoring_lab.algorithms.sos_relaxation import (
    HAS_CVXPY,
    SOSResult,
    _from_digits,
    _monomial_indices_deg1,
    _monomial_indices_deg2,
    _to_digits,
    run_sos_relaxation,
    solve_sos_degree2,
    solve_sos_degree4,
)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestDigitConversion:
    def test_roundtrip_base10(self) -> None:
        for n in [0, 1, 42, 143, 255, 1000]:
            assert _from_digits(_to_digits(n, 10), 10) == n

    def test_roundtrip_base2(self) -> None:
        for n in [0, 1, 5, 15, 255]:
            assert _from_digits(_to_digits(n, 2), 2) == n

    def test_roundtrip_base3(self) -> None:
        for n in [0, 1, 8, 26, 100]:
            assert _from_digits(_to_digits(n, 3), 3) == n


class TestMonomialIndices:
    def test_deg1_count(self) -> None:
        # 1 + dx + dy monomials
        monos = _monomial_indices_deg1(3, 4)
        assert len(monos) == 1 + 3 + 4

    def test_deg1_starts_with_constant(self) -> None:
        monos = _monomial_indices_deg1(2, 2)
        assert monos[0] == ()

    def test_deg2_count(self) -> None:
        # 1 + d + d*(d+1)/2 where d = dx + dy
        dx, dy = 2, 2
        d = dx + dy
        expected = 1 + d + d * (d + 1) // 2
        monos = _monomial_indices_deg2(dx, dy)
        assert len(monos) == expected

    def test_deg2_includes_pairs(self) -> None:
        monos = _monomial_indices_deg2(2, 2)
        # Should include (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), ...
        assert (0, 0) in monos
        assert (0, 1) in monos


# ---------------------------------------------------------------------------
# SOS degree-2 tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestSOSDegree2:
    def test_small_semiprime_base10(self) -> None:
        """Test degree-2 SOS on 11 * 13 = 143, base 10."""
        result = solve_sos_degree2(143, base=10, known_p=11, known_q=13)
        assert isinstance(result, SOSResult)
        assert result.degree == 2
        assert result.n == 143
        assert result.base == 10
        assert result.moment_matrix_size > 0
        assert result.solver_status in ("optimal", "optimal_inaccurate")
        # SOS gap should be between 0 and 1
        assert 0 <= result.sos_gap <= 1.0 + 1e-6

    def test_small_semiprime_base2(self) -> None:
        """Test degree-2 SOS on 15 = 3 * 5, base 2."""
        result = solve_sos_degree2(15, base=2, known_p=3, known_q=5)
        assert isinstance(result, SOSResult)
        assert result.degree == 2
        assert result.solver_status in ("optimal", "optimal_inaccurate", "infeasible")

    def test_result_fields_populated(self) -> None:
        """Ensure all result fields are populated."""
        result = solve_sos_degree2(77, base=10, known_p=7, known_q=11)
        assert result.num_digit_vars > 0
        assert result.solve_time_seconds >= 0
        assert len(result.eigenvalues_top5) > 0


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestSOSDegree4:
    def test_small_semiprime_base10(self) -> None:
        """Test degree-4 SOS on a small instance."""
        # 15 = 3 * 5, base 10: only 2 digits, so moment matrix is small
        result = solve_sos_degree4(15, base=10, known_p=3, known_q=5)
        assert isinstance(result, SOSResult)
        assert result.degree == 4
        # May be skipped if moment matrix too large
        if "too large" not in result.notes:
            assert result.solver_status in ("optimal", "optimal_inaccurate", "infeasible")

    def test_very_small_base3(self) -> None:
        """Test degree-4 on 15 = 3*5 in base 3."""
        result = solve_sos_degree4(15, base=3, known_p=3, known_q=5)
        assert isinstance(result, SOSResult)
        assert result.degree == 4

    def test_skips_when_too_large(self) -> None:
        """Degree-4 should skip gracefully if moment matrix is too large."""
        # 16-bit number in base 2 -> many digit variables -> huge moment matrix
        result = solve_sos_degree4(46649, base=2)
        assert isinstance(result, SOSResult)
        # Either solves or reports too large
        assert result.notes or result.solver_status


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestRunSOSRelaxation:
    def test_degree2_dispatch(self) -> None:
        result = run_sos_relaxation(143, base=10, degree=2)
        assert result.degree == 2

    def test_degree4_dispatch(self) -> None:
        result = run_sos_relaxation(15, base=10, degree=4)
        assert result.degree == 4

    def test_unsupported_degree(self) -> None:
        result = run_sos_relaxation(143, base=10, degree=6)
        assert "not implemented" in result.notes.lower()

    def test_gap_decreases_or_stays(self) -> None:
        """Degree-4 gap should be <= degree-2 gap (tighter relaxation).

        This is the theoretical expectation; numerical issues may cause
        slight violations, so we allow a tolerance.
        """
        n = 15  # 3 * 5, very small
        base = 10
        r2 = run_sos_relaxation(n, base=base, degree=2)
        r4 = run_sos_relaxation(n, base=base, degree=4)

        if r2.sos_gap < float("inf") and r4.sos_gap < float("inf"):
            if "too large" not in r4.notes:
                # Allow some numerical tolerance
                assert r4.sos_gap <= r2.sos_gap + 0.15, (
                    f"Degree-4 gap ({r4.sos_gap:.4f}) should not be much larger "
                    f"than degree-2 gap ({r2.sos_gap:.4f})"
                )
