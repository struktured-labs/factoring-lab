"""Tests for the Rust digit convolution factoring extension.

Mirrors test_digit_convolution.py and additionally cross-checks that the
Rust and Python implementations produce identical results.
"""

import pytest

rs = pytest.importorskip("factoring_kernels", reason="Rust extension not built")

from factoring_lab.algorithms.digit_convolution import DigitConvolution
from factoring_lab.algorithms.digit_convolution_rs import DigitConvolutionRust

SMALL_SEMIPRIMES = [
    (15, {3, 5}),
    (21, {3, 7}),
    (35, {5, 7}),
    (77, {7, 11}),
    (143, {11, 13}),
    (221, {13, 17}),
]


class TestDigitConvolutionRust:
    def test_base10_small(self):
        algo = DigitConvolutionRust(base=10)
        for n, factors in SMALL_SEMIPRIMES:
            result = algo.factor(n)
            assert result.success, f"failed on {n}: {result.notes}"
            assert result.factor in factors, f"wrong factor for {n}: {result.factor}"

    def test_base2(self):
        algo = DigitConvolutionRust(base=2)
        result = algo.factor(15)
        assert result.success
        assert result.factor in (3, 5)

    def test_base16(self):
        algo = DigitConvolutionRust(base=16)
        result = algo.factor(221)
        assert result.success
        assert result.factor in (13, 17)

    def test_prime_fails(self):
        algo = DigitConvolutionRust(base=10)
        result = algo.factor(97)
        assert not result.success

    def test_instrumentation(self):
        algo = DigitConvolutionRust(base=10)
        result = algo.factor(77)
        assert result.iteration_count > 0

    def test_various_bases(self):
        """Same number factored in different bases should yield same factors."""
        n = 143  # 11 * 13
        for base in [2, 3, 5, 7, 10, 16]:
            algo = DigitConvolutionRust(base=base)
            result = algo.factor(n)
            assert result.success, f"failed with base {base}: {result.notes}"
            assert result.factor in (11, 13), f"wrong factor with base {base}"


class TestRustPythonParity:
    """Verify Rust and Python produce identical factor results."""

    @pytest.mark.parametrize("n,factors", SMALL_SEMIPRIMES)
    def test_same_factor_base10(self, n, factors):
        py_result = DigitConvolution(base=10).factor(n)
        rs_result = DigitConvolutionRust(base=10).factor(n)
        assert py_result.success == rs_result.success
        assert py_result.factor == rs_result.factor

    @pytest.mark.parametrize("base", [2, 3, 5, 7, 10, 16])
    def test_same_factor_various_bases(self, base):
        n = 143
        py_result = DigitConvolution(base=base).factor(n)
        rs_result = DigitConvolutionRust(base=base).factor(n)
        assert py_result.success == rs_result.success
        assert py_result.factor == rs_result.factor

    def test_prime_parity(self):
        for p in [97, 101, 127, 131]:
            py_result = DigitConvolution(base=10).factor(p)
            rs_result = DigitConvolutionRust(base=10).factor(p)
            assert py_result.success == rs_result.success
