"""SMT-based digit convolution factoring using Z3.

Encodes the same digit-level constraints as DigitConvolution but
uses Z3's conflict-driven clause learning (CDCL) and theory solvers
instead of naive backtracking.

The key question: does Z3 find structure in these constraints that
backtracking misses? How does runtime scale with bit size?

Original constraint formulation: struktured, November 2009.
SMT encoding: Phase 2, March 2026.
"""

from __future__ import annotations

import math

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext

try:
    from z3 import (
        BitVec,
        BitVecVal,
        Solver,
        ZeroExt,
        Extract,
        sat,
        And,
        ULT,
        ULE,
        UGT,
    )

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class SMTConvolution(FactoringAlgorithm):
    """Factor via SMT encoding of digit convolution constraints.

    Encodes the factoring problem n = x * y as:
    - x and y are bitvectors of appropriate width
    - x * y = n
    - 1 < x <= y < n  (avoid trivial and duplicate solutions)

    Optionally layers digit-level convolution constraints on top
    for a given base, which can help the solver prune faster.
    """

    def __init__(self, base: int | None = None, timeout_ms: int = 30_000) -> None:
        """
        Args:
            base: if set, add digit-level convolution constraints in this base.
                  If None, just use the raw x * y = n bitvector constraint.
            timeout_ms: Z3 solver timeout in milliseconds.
        """
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver is required: uv pip install z3-solver")
        self._base = base
        self._timeout_ms = timeout_ms

    @property
    def name(self) -> str:
        suffix = f"_b{self._base}" if self._base else "_raw"
        return f"smt_convolution{suffix}"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        bits = n.bit_length()
        # Each factor needs at most 'bits' bits (could be as small as 2)
        # Use bits + 1 for safety in multiplication
        bv_width = bits

        s = Solver()
        s.set("timeout", self._timeout_ms)

        x = BitVec("x", bv_width)
        y = BitVec("y", bv_width)

        n_bv = BitVecVal(n, bv_width)

        # Core constraint: x * y = n
        # Use extended width to avoid overflow in multiplication
        ext_width = bv_width * 2
        x_ext = ZeroExt(bv_width, x)
        y_ext = ZeroExt(bv_width, y)
        n_ext = BitVecVal(n, ext_width)

        s.add(x_ext * y_ext == n_ext)

        # Non-trivial: x > 1, y > 1
        s.add(UGT(x, BitVecVal(1, bv_width)))
        s.add(UGT(y, BitVecVal(1, bv_width)))

        # Symmetry breaking: x <= y
        s.add(ULE(x, y))

        # Upper bounds: x <= sqrt(n), y < n
        isqrt_n = math.isqrt(n)
        s.add(ULE(x, BitVecVal(isqrt_n, bv_width)))
        s.add(ULT(y, n_bv))

        # Optional: digit-level convolution constraints
        if self._base is not None:
            self._add_digit_constraints(s, x, y, n, bv_width, ctx)

        ctx.record_iteration()
        result = s.check()

        if result == sat:
            model = s.model()
            x_val = model[x].as_long()
            y_val = model[y].as_long()

            # Verify
            if x_val * y_val == n and x_val > 1 and y_val > 1:
                return min(x_val, y_val), f"Z3 solved ({self.name})"
            return None, f"Z3 model invalid: {x_val} * {y_val} != {n}"
        else:
            return None, f"Z3 returned {result}"

    def _add_digit_constraints(
        self,
        s: "Solver",
        x: "BitVec",
        y: "BitVec",
        n: int,
        bv_width: int,
        ctx: InstrumentedContext,
    ) -> None:
        """Add digit-level convolution constraints to help guide the solver."""
        b = self._base
        assert b is not None

        # Compute digits of n
        digits: list[int] = []
        temp = n
        while temp > 0:
            digits.append(temp % b)
            temp //= b
        d = len(digits)

        b_bv = BitVecVal(b, bv_width)

        # For each digit position, the product's digit must match n's digit
        # This is equivalent to: (x * y) mod b^(k+1) determines digits 0..k
        # We add constraints on partial products modulo increasing powers of b
        power = 1
        for k in range(min(d, 8)):  # limit depth to avoid constraint explosion
            power *= b
            ctx.record_mod_mul()
            # The value of x*y mod b^(k+1) must equal n mod b^(k+1)
            target = n % power

            if power.bit_length() <= bv_width:
                mod_bv = BitVecVal(power, bv_width)
                target_bv = BitVecVal(target, bv_width)
                # Use extended multiplication for the mod constraint
                ext_width = bv_width * 2
                x_ext = ZeroExt(bv_width, x)
                y_ext = ZeroExt(bv_width, y)
                mod_ext = BitVecVal(power, ext_width)
                target_ext = BitVecVal(target, ext_width)
                s.add((x_ext * y_ext) % mod_ext == target_ext)


class SMTConvolutionRaw(SMTConvolution):
    """Raw bitvector multiplication constraint, no digit structure."""

    def __init__(self, timeout_ms: int = 30_000) -> None:
        super().__init__(base=None, timeout_ms=timeout_ms)


class SMTConvolutionBase10(SMTConvolution):
    """Digit convolution with base-10 constraints."""

    def __init__(self, timeout_ms: int = 30_000) -> None:
        super().__init__(base=10, timeout_ms=timeout_ms)


class SMTConvolutionBase2(SMTConvolution):
    """Digit convolution with base-2 (binary) constraints."""

    def __init__(self, timeout_ms: int = 30_000) -> None:
        super().__init__(base=2, timeout_ms=timeout_ms)
