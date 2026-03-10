"""Tests for prime gap analysis tools."""

from __future__ import annotations

import math

import pytest

from factoring_lab.analysis.prime_gaps import (
    average_gap_estimate,
    empirical_gap_stats,
    goldbach_check,
    prime_gaps_in_range,
)


class TestPrimeGapsInRange:
    """Tests for prime_gaps_in_range."""

    def test_small_range(self) -> None:
        # Primes in [2, 20]: 2, 3, 5, 7, 11, 13, 17, 19
        # Gaps: 1, 2, 2, 4, 2, 4, 2
        gaps = prime_gaps_in_range(2, 20)
        assert gaps == [1, 2, 2, 4, 2, 4, 2]

    def test_range_with_no_primes(self) -> None:
        # 24..28 contains no primes (29 is outside)
        gaps = prime_gaps_in_range(24, 28)
        assert gaps == []

    def test_single_prime(self) -> None:
        # Range containing exactly one prime
        gaps = prime_gaps_in_range(6, 8)
        assert gaps == []  # only prime 7, no gap

    def test_twin_primes(self) -> None:
        # 10..14: primes 11, 13 -> gap 2
        gaps = prime_gaps_in_range(10, 14)
        assert gaps == [2]

    def test_range_starting_at_2(self) -> None:
        gaps = prime_gaps_in_range(1, 7)
        # Primes: 2, 3, 5, 7 -> gaps: 1, 2, 2
        assert gaps == [1, 2, 2]

    def test_wider_range(self) -> None:
        # Primes in [90, 110]: 97, 101, 103, 107, 109
        gaps = prime_gaps_in_range(90, 110)
        assert gaps == [4, 2, 4, 2]


class TestAverageGapEstimate:
    """Tests for average_gap_estimate."""

    def test_basic_values(self) -> None:
        # For bits=10, estimate should be 10 * ln(2) ≈ 6.93
        est = average_gap_estimate(10)
        assert abs(est - 10 * math.log(2)) < 1e-10

    def test_bits_1(self) -> None:
        est = average_gap_estimate(1)
        assert abs(est - math.log(2)) < 1e-10

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError):
            average_gap_estimate(0)

    def test_grows_with_bits(self) -> None:
        assert average_gap_estimate(20) > average_gap_estimate(10)

    def test_known_ratio(self) -> None:
        # Ratio of estimates for bits b1 and b2 should be b1/b2
        r = average_gap_estimate(30) / average_gap_estimate(15)
        assert abs(r - 2.0) < 1e-10


class TestEmpiricalGapStats:
    """Tests for empirical_gap_stats."""

    def test_basic_stats_small_bits(self) -> None:
        stats = empirical_gap_stats(bits=10, count=50)
        assert "mean" in stats
        assert "median" in stats
        assert "max" in stats
        assert "min" in stats
        assert "std" in stats
        assert "pnt_estimate" in stats

    def test_min_less_than_max(self) -> None:
        stats = empirical_gap_stats(bits=10, count=100)
        assert stats["min"] <= stats["max"]

    def test_mean_positive(self) -> None:
        stats = empirical_gap_stats(bits=10, count=50)
        assert stats["mean"] > 0

    def test_mean_in_ballpark_of_pnt(self) -> None:
        # For moderate bit sizes, empirical mean should be roughly
        # within a factor of 3 of PNT estimate
        stats = empirical_gap_stats(bits=15, count=200)
        pnt = stats["pnt_estimate"]
        assert stats["mean"] > pnt * 0.3
        assert stats["mean"] < pnt * 3.0

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError):
            empirical_gap_stats(bits=1)

    def test_invalid_count(self) -> None:
        with pytest.raises(ValueError):
            empirical_gap_stats(bits=10, count=1)

    def test_count_field(self) -> None:
        stats = empirical_gap_stats(bits=10, count=50)
        assert stats["count"] == 49.0  # count-1 gaps


class TestGoldbachCheck:
    """Tests for goldbach_check."""

    def test_small_limit(self) -> None:
        # For small limits, check that known small primes behave correctly
        # 2 cannot be written as q+r for primes q,r (smallest is 2+2=4>2)
        # 3 = 2+1? No, 1 not prime. But 3-1=2 is even, 2=2+0? No.
        # However 3 can't be q+r for primes, AND 3-1=2=2+0 no.
        # So 2 and 3 should appear as failures.
        result = goldbach_check(10)
        # 2: can't be q+r (min sum is 2+2=4), can't be q+r+1 (need q+r=1, impossible)
        # 3: can't be q+r (need q+r=3, only 2+1 but 1 not prime), check q+r+1=3 -> q+r=2 -> only 1+1, not primes
        # 5: 2+3=5 yes!
        # 7: 2+5=7 yes!
        assert 2 in result
        assert 3 in result
        assert 5 not in result
        assert 7 not in result

    def test_moderate_limit(self) -> None:
        # Beyond small primes, all should be expressible
        result = goldbach_check(100)
        # Only 2 and 3 should fail
        assert set(result) == {2, 3}

    def test_limit_below_2(self) -> None:
        assert goldbach_check(1) == []
        assert goldbach_check(0) == []

    def test_limit_200(self) -> None:
        result = goldbach_check(200)
        # Should still be just {2, 3}
        assert set(result) == {2, 3}
