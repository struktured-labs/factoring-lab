"""Tests for digit convolution factoring."""

from factoring_lab.algorithms.digit_convolution import DigitConvolution


SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (21, {3, 7}),
    (35, {5, 7}),
    (77, {7, 11}),
    (143, {11, 13}),
    (221, {13, 17}),
]


class TestDigitConvolution:
    def test_base10_small(self):
        algo = DigitConvolution(base=10)
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, f"wrong factor for {n}: {result.factor}"

    def test_base2(self):
        algo = DigitConvolution(base=2)
        result = algo.factor(15)
        assert result.success
        assert result.factor in (3, 5)

    def test_base16(self):
        algo = DigitConvolution(base=16)
        result = algo.factor(221)
        assert result.success
        assert result.factor in (13, 17)

    def test_prime_fails(self):
        algo = DigitConvolution(base=10)
        result = algo.factor(97)
        assert not result.success

    def test_instrumentation(self):
        algo = DigitConvolution(base=10)
        result = algo.factor(77)
        assert result.iteration_count > 0
        assert result.modular_multiplies > 0

    def test_various_bases(self):
        """Same number factored in different bases should yield same factors."""
        n = 143  # 11 * 13
        for base in [2, 3, 5, 7, 10, 16]:
            algo = DigitConvolution(base=base)
            result = algo.factor(n)
            assert result.success, f"failed with base {base}: {result.notes}"
            assert result.factor in (11, 13), f"wrong factor with base {base}"
