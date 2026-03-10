"""Integer programming / digit convolution factoring approach.

Based on the observation that n = x·y can be written as a system of
digit-level constraints in base b:

  α_k = Σ_{i=0}^{k} x_i · y_{k-i}

with recursive carry propagation:

  m_0 = α_0
  m_k = α_k + (m_{k-1} - c_{k-1}) / b
  m_k ≡ c_k (mod b)

This is essentially solving a polynomial multiplication problem
digit-by-digit with backtracking.

Original formulation: struktured, November 2009.
"""

from __future__ import annotations

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext


class DigitConvolution(FactoringAlgorithm):
    """Factor by solving digit-level convolution constraints with backtracking.

    For a given base b, decomposes n into digits c_0..c_{d-1} and searches
    for digit sequences x_0..x_{d-1}, y_0..y_{d-1} such that the convolution
    α_k = Σ x_i · y_{k-i} satisfies the carry-propagation constraints.

    This is a constraint-satisfaction / backtracking approach. Its worst case
    is O((b²)^d), but structural pruning may help on certain instances.
    """

    def __init__(self, base: int = 10, max_digits: int | None = None) -> None:
        self._base = base
        self._max_digits = max_digits

    @property
    def name(self) -> str:
        return f"digit_convolution_b{self._base}"

    def _to_digits(self, n: int) -> list[int]:
        """Convert n to base-b digits, least significant first."""
        if n == 0:
            return [0]
        digits = []
        while n > 0:
            digits.append(n % self._base)
            n //= self._base
        return digits

    def _from_digits(self, digits: list[int]) -> int:
        """Convert base-b digits back to integer."""
        result = 0
        for i in reversed(range(len(digits))):
            result = result * self._base + digits[i]
        return result

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        b = self._base
        c = self._to_digits(n)
        d = len(c)

        if self._max_digits is not None:
            d = min(d, self._max_digits)

        # x_digits and y_digits: digit assignments we're searching for
        # We search by position k = 0, 1, ..., d-1
        # At each position, we try all (x_k, y_k) pairs and check the
        # carry-propagation constraint.

        # Stack-based backtracking search
        # State: (position k, carry_in, x_digits_so_far, y_digits_so_far)
        # At position k, we need:
        #   total_k = Σ_{i=0}^{k} x_i * y_{k-i} + carry_in
        #   c_k = total_k mod b
        #   carry_out = total_k // b

        # We iterate over all possible (x_k, y_k) values
        # and for each, check if the constraint at position k is satisfiable
        # given the already-chosen x_0..x_{k-1}, y_0..y_{k-1}.

        x_digits: list[int] = [0] * d
        y_digits: list[int] = [0] * d

        # For each position k, we need to enumerate (x_k, y_k) pairs.
        # The constraint at position k involves all previous digits.
        # carry[k] is the carry into position k.

        def solve(k: int, carry: int) -> int | None:
            ctx.record_iteration()

            if k == d:
                # Check if carry is zero (clean factorization)
                if carry == 0:
                    x = self._from_digits(x_digits)
                    y = self._from_digits(y_digits)
                    if x > 1 and y > 1 and x * y == n:
                        return min(x, y)
                return None

            # At position k, the total contribution is:
            # sum_{i=0}^{k} x_i * y_{k-i} + carry
            # We need this mod b == c[k]
            # The new carry is (total) // b

            # We already know x_0..x_{k-1} and y_0..y_{k-1}.
            # We need to choose x_k and y_k such that:
            #   partial_sum + x_0*y_k + x_k*y_0 + x_k*y_k_cross... + carry ≡ c[k] (mod b)
            # Actually the cross terms only involve x_k * y_0 and x_0 * y_k
            # plus x_k * y_k is NOT included (that would be for position 2k).

            # More carefully: α_k = Σ_{i=0}^{k} x_i * y_{k-i}
            # The "known" part (from previously chosen digits): Σ_{i=1}^{k-1} x_i * y_{k-i}
            # The "new" part involving x_k and y_k: x_0 * y_k + x_k * y_0
            #   (and x_k * y_k only contributes to α_{2k}, not α_k)
            # Wait, no: α_k = x_0*y_k + x_1*y_{k-1} + ... + x_k*y_0
            # So x_k appears multiplied by y_0, and y_k appears multiplied by x_0.

            # Known partial sum from i=1 to k-1
            partial = sum(x_digits[i] * y_digits[k - i] for i in range(1, k))
            ctx.record_mod_mul(max(0, k - 1))

            # We need: (partial + x_digits[0]*y_k + x_k*y_digits[0] + carry) mod b == c[k]
            # And: carry_out = (partial + x_digits[0]*y_k + x_k*y_digits[0] + carry) // b

            x0 = x_digits[0]
            y0 = y_digits[0]

            for xk in range(b):
                for yk in range(b):
                    ctx.record_mod_mul()
                    total = partial + x0 * yk + xk * y0 + carry
                    if total % b == c[k]:
                        x_digits[k] = xk
                        y_digits[k] = yk
                        new_carry = total // b
                        result = solve(k + 1, new_carry)
                        if result is not None:
                            return result

            return None

        # Try all (x_0, y_0) pairs for the first digit
        for x0 in range(1, b):  # skip 0 to avoid trivial
            for y0 in range(x0, b):  # y0 >= x0 to avoid duplicates
                ctx.record_mod_mul()
                total = x0 * y0
                if total % b == c[0]:
                    x_digits[0] = x0
                    y_digits[0] = y0
                    carry = total // b
                    result = solve(1, carry)
                    if result is not None:
                        return result, f"found via digit convolution base {b}"

        return None, f"no factorization found via digit convolution base {b}"
