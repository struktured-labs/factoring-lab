"""Rust-accelerated digit convolution factoring.

Wraps the ``factoring_kernels`` Rust extension (built via maturin/PyO3)
in a class that follows the standard :class:`FactoringAlgorithm` interface.

Falls back gracefully if the extension is not installed.
"""

from __future__ import annotations

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext

try:
    from factoring_kernels import digit_convolution_factor as _rs_factor

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class DigitConvolutionRust(FactoringAlgorithm):
    """Rust-backed digit convolution factoring via backtracking.

    Identical algorithm to :class:`DigitConvolution` but executed in compiled
    Rust code for significantly better throughput.

    Raises :class:`RuntimeError` at construction time if the Rust extension
    has not been built.
    """

    def __init__(self, base: int = 10, max_digits: int | None = None) -> None:
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "factoring_kernels Rust extension not available. "
                "Build it with: cd rust/factoring_kernels && maturin develop --release"
            )
        self._base = base
        self._max_digits = max_digits

    @property
    def name(self) -> str:
        return f"digit_convolution_rs_b{self._base}"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        result = _rs_factor(n, self._base, self._max_digits)
        if result is None:
            return None, f"no factorization found via Rust digit convolution base {self._base}"
        factor, _cofactor, iterations = result
        # Backfill the iteration counter so FactoringResult gets it.
        ctx.iteration_count = iterations
        return factor, f"found via Rust digit convolution base {self._base}"
