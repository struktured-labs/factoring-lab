"""Structured integer generators for factoring experiments."""

from factoring_lab.generators.semiprimes import (
    balanced_semiprime,
    unbalanced_semiprime,
    smooth_pm1_semiprime,
    random_semiprime,
    SemiprimeSpec,
)

__all__ = [
    "balanced_semiprime",
    "unbalanced_semiprime",
    "smooth_pm1_semiprime",
    "random_semiprime",
    "SemiprimeSpec",
]
