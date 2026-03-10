"""Trial division factoring."""

from __future__ import annotations

import math

from factoring_lab.algorithms.base import FactoringAlgorithm, InstrumentedContext


class TrialDivision(FactoringAlgorithm):
    """Trial division up to sqrt(n) or a configurable limit."""

    def __init__(self, limit: int | None = None) -> None:
        self._limit = limit

    @property
    def name(self) -> str:
        return "trial_division"

    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        bound = self._limit or math.isqrt(n) + 1

        # Check small primes explicitly
        for p in (2, 3):
            ctx.record_iteration()
            if n % p == 0:
                return p, f"found at d={p}"

        # 6k +/- 1 wheel
        d = 5
        step = 2
        while d <= bound:
            ctx.record_iteration()
            ctx.record_mod_mul()  # modular reduction
            if n % d == 0:
                return d, f"found at d={d}"
            d += step
            step = 6 - step  # alternates between 2 and 4

        return None, f"no factor up to {bound}"
