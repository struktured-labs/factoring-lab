"""SMT-based factoring with partial bit leaking.

Extends SMTConvolution to simulate an attacker who knows some fraction
of the least-significant bits of one factor.  This models side-channel
or partial-key-recovery scenarios studied in the literature
(e.g., Ajani et al. 2024 who needed ~50-60 % leaked bits for
circuit-based SAT approaches).

The key research question: does the digit convolution encoding benefit
more or less from leaked bits compared to circuit-based approaches?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from factoring_lab.algorithms.base import FactoringAlgorithm, FactoringResult, InstrumentedContext

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


class SMTLeakedBits(FactoringAlgorithm):
    """Factor via SMT with partial knowledge of one factor's bits.

    Given a semiprime n = p * q, fix a fraction of the least-significant
    bits of p as known constraints in the Z3 model.  This dramatically
    reduces the search space and lets us measure how much partial
    information the solver needs to crack a given bit size.

    Parameters
    ----------
    leak_fraction : float
        Fraction of p's bits to reveal (0.0 = no leak, 1.0 = full leak).
    known_p : int | None
        The actual value of p (needed to extract the leaked bits).
        If None, the algorithm cannot add leak constraints and falls
        back to plain SMT factoring.
    base : int | None
        Optional digit base for convolution constraints.
    timeout_ms : int
        Z3 solver timeout in milliseconds.
    """

    def __init__(
        self,
        leak_fraction: float = 0.0,
        known_p: int | None = None,
        base: int | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver is required: uv pip install z3-solver")
        if not (0.0 <= leak_fraction <= 1.0):
            raise ValueError(f"leak_fraction must be in [0, 1], got {leak_fraction}")
        self._leak_fraction = leak_fraction
        self._known_p = known_p
        self._base = base
        self._timeout_ms = timeout_ms

    @property
    def name(self) -> str:
        base_str = f"_b{self._base}" if self._base else "_raw"
        return f"smt_leaked{base_str}_f{self._leak_fraction:.2f}"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        bits = n.bit_length()
        bv_width = bits

        s = Solver()
        s.set("timeout", self._timeout_ms)

        x = BitVec("x", bv_width)
        y = BitVec("y", bv_width)

        n_bv = BitVecVal(n, bv_width)

        # Extended multiplication to avoid overflow
        ext_width = bv_width * 2
        x_ext = ZeroExt(bv_width, x)
        y_ext = ZeroExt(bv_width, y)
        n_ext = BitVecVal(n, ext_width)

        s.add(x_ext * y_ext == n_ext)

        # Non-trivial factors
        s.add(UGT(x, BitVecVal(1, bv_width)))
        s.add(UGT(y, BitVecVal(1, bv_width)))

        # Symmetry breaking: x <= y
        s.add(ULE(x, y))

        # Upper bounds
        isqrt_n = math.isqrt(n)
        s.add(ULE(x, BitVecVal(isqrt_n, bv_width)))
        s.add(ULT(y, n_bv))

        # --- Leaked bits constraints ---
        if self._known_p is not None and self._leak_fraction > 0:
            p = self._known_p
            p_bits = p.bit_length()
            num_leaked = max(1, int(p_bits * self._leak_fraction))
            self._add_leaked_constraints(s, x, p, num_leaked, bv_width)

        # Optional digit-level convolution constraints
        if self._base is not None:
            self._add_digit_constraints(s, x, y, n, bv_width, ctx)

        ctx.record_iteration()
        result = s.check()

        if result == sat:
            model = s.model()
            x_val = model[x].as_long()
            y_val = model[y].as_long()

            if x_val * y_val == n and x_val > 1 and y_val > 1:
                return min(x_val, y_val), f"Z3 solved ({self.name})"
            return None, f"Z3 model invalid: {x_val} * {y_val} != {n}"
        else:
            return None, f"Z3 returned {result}"

    @staticmethod
    def _add_leaked_constraints(
        s: "Solver",
        x: "BitVec",
        known_p: int,
        num_leaked: int,
        bv_width: int,
    ) -> None:
        """Fix the bottom *num_leaked* bits of x to match known_p."""
        # Mask for the leaked bits
        mask = (1 << num_leaked) - 1
        leaked_value = known_p & mask

        mask_bv = BitVecVal(mask, bv_width)
        leaked_bv = BitVecVal(leaked_value, bv_width)

        s.add((x & mask_bv) == leaked_bv)

    def _add_digit_constraints(
        self,
        s: "Solver",
        x: "BitVec",
        y: "BitVec",
        n: int,
        bv_width: int,
        ctx: InstrumentedContext,
    ) -> None:
        """Add digit-level convolution constraints (same as SMTConvolution)."""
        b = self._base
        assert b is not None

        digits: list[int] = []
        temp = n
        while temp > 0:
            digits.append(temp % b)
            temp //= b
        d = len(digits)

        power = 1
        for k in range(min(d, 8)):
            power *= b
            ctx.record_mod_mul()
            target = n % power

            if power.bit_length() <= bv_width:
                ext_width = bv_width * 2
                x_ext = ZeroExt(bv_width, x)
                y_ext = ZeroExt(bv_width, y)
                mod_ext = BitVecVal(power, ext_width)
                target_ext = BitVecVal(target, ext_width)
                s.add((x_ext * y_ext) % mod_ext == target_ext)

    def factor_with_leak(
        self,
        n: int,
        known_bits_of_p: int,
        num_known_bits: int,
        timeout_ms: int | None = None,
    ) -> FactoringResult:
        """Factor n given explicit partial knowledge of p's LSBs.

        This is a convenience method that does not require setting
        known_p / leak_fraction at construction time.

        Parameters
        ----------
        n : int
            The semiprime to factor.
        known_bits_of_p : int
            The value of the bottom *num_known_bits* bits of p.
        num_known_bits : int
            How many LSBs are known.
        timeout_ms : int | None
            Override the solver timeout.

        Returns
        -------
        FactoringResult
        """
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver is required")

        bits = n.bit_length()
        bv_width = bits
        timeout = timeout_ms or self._timeout_ms

        s = Solver()
        s.set("timeout", timeout)

        x = BitVec("x", bv_width)
        y = BitVec("y", bv_width)

        ext_width = bv_width * 2
        x_ext = ZeroExt(bv_width, x)
        y_ext = ZeroExt(bv_width, y)
        n_ext = BitVecVal(n, ext_width)

        s.add(x_ext * y_ext == n_ext)
        s.add(UGT(x, BitVecVal(1, bv_width)))
        s.add(UGT(y, BitVecVal(1, bv_width)))
        s.add(ULE(x, y))

        isqrt_n = math.isqrt(n)
        s.add(ULE(x, BitVecVal(isqrt_n, bv_width)))
        s.add(ULT(y, BitVecVal(n, bv_width)))

        # Leaked bits
        self._add_leaked_constraints(s, x, known_bits_of_p, num_known_bits, bv_width)

        if self._base is not None:
            ctx_inner = InstrumentedContext()
            self._add_digit_constraints(s, x, y, n, bv_width, ctx_inner)

        import time
        t0 = time.perf_counter()
        result = s.check()
        elapsed = time.perf_counter() - t0

        if result == sat:
            model = s.model()
            x_val = model[x].as_long()
            y_val = model[y].as_long()

            if x_val * y_val == n and x_val > 1 and y_val > 1:
                return FactoringResult(
                    algorithm_name=self.name,
                    n=n,
                    success=True,
                    factor=min(x_val, y_val),
                    cofactor=max(x_val, y_val),
                    runtime_seconds=elapsed,
                    notes=f"factor_with_leak({num_known_bits} bits)",
                )

        return FactoringResult(
            algorithm_name=self.name,
            n=n,
            success=False,
            runtime_seconds=elapsed,
            notes=f"Z3 returned {result}",
        )
