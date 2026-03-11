"""Circuit-based SAT encoding of integer factoring using Z3.

Implements the standard array multiplier circuit approach from the literature:
- x and y are represented as vectors of boolean variables
- Partial products are computed as x_i AND y_j
- Partial products are summed using full adder chains with carry propagation
- The output bits are constrained to equal n

This is the "textbook" encoding used in most SAT-based factoring papers,
providing a baseline for comparison against our digit convolution approach.

Reference encoding, March 2026.
"""

from __future__ import annotations

import math

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext

try:
    from z3 import (
        Bool,
        And,
        Or,
        Xor,
        Not,
        Solver,
        sat,
        is_true,
        BoolVal,
    )

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


def _full_adder(a, b, cin):
    """Full adder: returns (sum_bit, carry_out) as Z3 boolean expressions.

    sum  = a XOR b XOR cin
    cout = (a AND b) OR (cin AND (a XOR b))
    """
    a_xor_b = Xor(a, b)
    s = Xor(a_xor_b, cin)
    cout = Or(And(a, b), And(cin, a_xor_b))
    return s, cout


def _half_adder(a, b):
    """Half adder: returns (sum_bit, carry_out) as Z3 boolean expressions."""
    s = Xor(a, b)
    cout = And(a, b)
    return s, cout


def _bool_le(x_bits, y_bits):
    """Encode x <= y for boolean bit vectors (MSB first internally).

    x_bits and y_bits are lists with index 0 = LSB.
    We compare from MSB down: at the first differing bit,
    x must be 0 and y must be 1 for x < y.
    """
    n = len(x_bits)
    assert len(y_bits) == n

    # Build from LSB up using the recurrence:
    # le(i..n-1) = (y_i AND NOT x_i) OR (NOT(x_i XOR y_i) AND le(i+1..n-1))
    # Base case: le(empty) = True (equal so far means <=)
    le = BoolVal(True)
    for i in range(n - 1, -1, -1):  # MSB to LSB
        bits_equal = Not(Xor(x_bits[i], y_bits[i]))
        y_greater = And(y_bits[i], Not(x_bits[i]))
        le = Or(y_greater, And(bits_equal, le))
    return le


def _bool_gt_one(bits):
    """Encode x > 1 for a boolean bit vector (index 0 = LSB).

    x > 1 means either any bit above bit 0 is set,
    i.e., x >= 2. This is simply OR(bits[1], bits[2], ...).
    """
    if len(bits) <= 1:
        return BoolVal(False)
    return Or(*bits[1:])


class CircuitSAT(FactoringAlgorithm):
    """Factor via boolean circuit SAT encoding (array multiplier).

    Encodes factoring n = x * y by constructing an array multiplier
    circuit at the boolean level. Each bit of x and y is an independent
    boolean variable, and the multiplication is built from AND gates
    (partial products) and full/half adder chains.

    This is the standard encoding from the SAT factoring literature.
    """

    def __init__(self, timeout_ms: int = 30_000) -> None:
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver is required: uv pip install z3-solver")
        self._timeout_ms = timeout_ms

    @property
    def name(self) -> str:
        return "circuit_sat"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        n_bits = n.bit_length()
        # Each factor has at most n_bits bits (but we know both are < n)
        # For a balanced semiprime, each factor is ~n_bits/2 bits,
        # but we use n_bits to be safe (the constraints will limit the range).
        factor_bits = n_bits

        s = Solver()
        s.set("timeout", self._timeout_ms)

        # Create boolean variables for each bit of x and y
        x_bools = [Bool(f"x_{i}") for i in range(factor_bits)]
        y_bools = [Bool(f"y_{i}") for i in range(factor_bits)]

        # Build array multiplier circuit
        # Product has up to 2 * factor_bits bits
        product_bits = self._build_array_multiplier(s, x_bools, y_bools, ctx)

        # Constrain product bits to equal n
        for i in range(len(product_bits)):
            n_bit = BoolVal(bool((n >> i) & 1))
            s.add(product_bits[i] == n_bit)

        # Any remaining high bits of n must be 0 (they are, since product_bits
        # covers 2*factor_bits which is >= n_bits)

        # Symmetry breaking: x <= y
        s.add(_bool_le(x_bools, y_bools))

        # Exclude trivial factors: x > 1, y > 1
        s.add(_bool_gt_one(x_bools))
        s.add(_bool_gt_one(y_bools))

        # Solve
        ctx.record_iteration()
        result = s.check()

        if result == sat:
            model = s.model()
            x_val = sum(
                (1 << i) for i in range(factor_bits) if is_true(model[x_bools[i]])
            )
            y_val = sum(
                (1 << i) for i in range(factor_bits) if is_true(model[y_bools[i]])
            )

            # Verify
            if x_val * y_val == n and x_val > 1 and y_val > 1:
                return min(x_val, y_val), "circuit SAT solved"
            return None, f"circuit SAT model invalid: {x_val} * {y_val} != {n}"
        else:
            return None, f"circuit SAT returned {result}"

    def _build_array_multiplier(self, s, x_bools, y_bools, ctx):
        """Build an array multiplier circuit.

        For an m-bit x and m-bit y, produces a 2m-bit product.

        The array multiplier works row by row:
        - Row j contains partial products: x_i AND y_j for each i
        - Each row is shifted left by j positions
        - Rows are summed using ripple-carry adder chains

        Returns list of product bits (index 0 = LSB).
        """
        m = len(x_bools)

        # Generate partial product matrix
        # pp[j][i] = x_i AND y_j  (row j, column i+j in the product)
        pp = []
        for j in range(m):
            row = []
            for i in range(m):
                row.append(And(x_bools[i], y_bools[j]))
                ctx.record_mod_mul()
            pp.append(row)

        # Sum the partial products using array of adders
        # Start with row 0 as the initial partial sum
        # result_bits will hold the final product bits
        result_bits = [BoolVal(False)] * (2 * m)

        # The product bit at position k is determined by summing all
        # partial products at that column position.
        #
        # We use the standard iterative approach:
        # Accumulate rows one at a time into a running sum.

        # Initialize with first row (no addition needed)
        # Row 0 contributes to positions 0..m-1
        if m == 0:
            return result_bits

        # Use column-wise reduction
        # Collect all partial products by column position
        columns = [[] for _ in range(2 * m)]
        for j in range(m):
            for i in range(m):
                col = i + j
                columns[col].append(pp[j][i])

        # Reduce each column using a chain of full/half adders
        # Carries propagate to the next column
        for col in range(2 * m):
            bits = columns[col]
            if not bits:
                result_bits[col] = BoolVal(False)
                continue

            # Reduce this column: repeatedly apply full adders
            while len(bits) > 1:
                new_bits = []
                i = 0
                while i + 2 < len(bits):
                    # Full adder: 3 bits -> 1 sum + 1 carry
                    s_bit, c_bit = _full_adder(bits[i], bits[i + 1], bits[i + 2])
                    new_bits.append(s_bit)
                    # Carry goes to next column
                    if col + 1 < 2 * m:
                        columns[col + 1].append(c_bit)
                    i += 3
                if i + 1 < len(bits):
                    # Half adder: 2 bits -> 1 sum + 1 carry
                    s_bit, c_bit = _half_adder(bits[i], bits[i + 1])
                    new_bits.append(s_bit)
                    if col + 1 < 2 * m:
                        columns[col + 1].append(c_bit)
                    i += 2
                if i < len(bits):
                    new_bits.append(bits[i])
                bits = new_bits

            result_bits[col] = bits[0]

        return result_bits
