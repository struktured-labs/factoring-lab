"""Pollard's rho factoring algorithm."""

from __future__ import annotations

import math

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext


class PollardRho(FactoringAlgorithm):
    """Pollard's rho with Brent's cycle detection."""

    def __init__(
        self, max_iterations: int = 1_000_000, c: int = 1, max_retries: int = 5
    ) -> None:
        self._max_iterations = max_iterations
        self._c = c
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "pollard_rho"

    @staticmethod
    def _f(x: int, c: int, n: int, ctx: InstrumentedContext) -> int:
        ctx.record_mod_mul()
        return (x * x + c) % n

    def _try_once(self, n: int, c: int, ctx: InstrumentedContext) -> int | None:
        """Single rho attempt with a given c. Returns factor or None."""
        y = 2
        r = 1
        q = 1
        x = y
        ys = y

        while True:
            x = y
            for _ in range(r):
                y = self._f(y, c, n, ctx)

            k = 0
            while k < r:
                ys = y
                batch = min(128, r - k)
                for _ in range(batch):
                    ctx.record_iteration()
                    if ctx.iteration_count > self._max_iterations:
                        return None
                    y = self._f(y, c, n, ctx)
                    q = (q * abs(x - y)) % n
                    ctx.record_mod_mul(2)

                ctx.record_gcd()
                g = math.gcd(q, n)
                if g != 1:
                    if g == n:
                        while True:
                            ys = self._f(ys, c, n, ctx)
                            ctx.record_gcd()
                            g = math.gcd(abs(x - ys), n)
                            if g != 1:
                                break
                    if g == n:
                        return None  # this c failed
                    return g
                k += batch
            r *= 2

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        c = self._c
        for attempt in range(self._max_retries):
            result = self._try_once(n, c, ctx)
            if result is not None:
                return result, f"found after {ctx.iteration_count} iters (c={c})"
            c += 1  # try next polynomial
        return None, f"failed after {self._max_retries} retries"
