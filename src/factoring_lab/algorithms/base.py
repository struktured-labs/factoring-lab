"""Base classes and result types for factoring algorithms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactoringResult:
    """Structured result from a factoring attempt."""

    algorithm_name: str
    n: int
    success: bool
    factor: int | None = None
    cofactor: int | None = None
    runtime_seconds: float = 0.0
    iteration_count: int = 0
    gcd_calls: int = 0
    modular_multiplies: int = 0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def trivial(self) -> bool:
        """True if the found factor is 1 or n (trivial)."""
        return self.factor is not None and self.factor in (1, self.n)


class InstrumentedContext:
    """Tracks operation counts during a factoring run."""

    def __init__(self) -> None:
        self.iteration_count: int = 0
        self.gcd_calls: int = 0
        self.modular_multiplies: int = 0
        self._start_time: float = 0.0

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    def record_gcd(self) -> None:
        self.gcd_calls += 1

    def record_mod_mul(self, count: int = 1) -> None:
        self.modular_multiplies += count

    def record_iteration(self) -> None:
        self.iteration_count += 1


class FactoringAlgorithm(ABC):
    """Abstract base for factoring algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for logging and results."""
        ...

    @abstractmethod
    def _run(self, n: int, ctx: InstrumentedContext) -> tuple[int | None, str]:
        """Run the algorithm. Return (factor_or_None, notes)."""
        ...

    def factor(self, n: int) -> FactoringResult:
        """Run the algorithm on n and return a structured result."""
        if n < 2:
            return FactoringResult(
                algorithm_name=self.name, n=n, success=False, notes="n < 2"
            )
        if n % 2 == 0:
            return FactoringResult(
                algorithm_name=self.name,
                n=n,
                success=True,
                factor=2,
                cofactor=n // 2,
                notes="even",
            )

        ctx = InstrumentedContext()
        ctx.start()
        found, notes = self._run(n, ctx)
        elapsed = ctx.elapsed()

        success = found is not None and found not in (1, n)
        cofactor = n // found if (found and success) else None

        return FactoringResult(
            algorithm_name=self.name,
            n=n,
            success=success,
            factor=found if success else None,
            cofactor=cofactor,
            runtime_seconds=elapsed,
            iteration_count=ctx.iteration_count,
            gcd_calls=ctx.gcd_calls,
            modular_multiplies=ctx.modular_multiplies,
            notes=notes,
        )
