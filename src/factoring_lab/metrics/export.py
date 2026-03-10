"""Export factoring results to various formats."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from factoring_lab.algorithms.base import FactoringResult
from factoring_lab.generators.semiprimes import SemiprimeSpec


RESULT_FIELDS = [
    "algorithm_name",
    "n",
    "n_bits",
    "family",
    "p",
    "q",
    "balance_ratio",
    "success",
    "factor",
    "cofactor",
    "runtime_seconds",
    "iteration_count",
    "gcd_calls",
    "modular_multiplies",
    "notes",
]


def result_to_dict(
    result: FactoringResult, spec: SemiprimeSpec | None = None
) -> dict[str, Any]:
    """Convert a FactoringResult (and optional spec) to a flat dict."""
    d: dict[str, Any] = {
        "algorithm_name": result.algorithm_name,
        "n": result.n,
        "n_bits": result.n.bit_length(),
        "success": result.success,
        "factor": result.factor,
        "cofactor": result.cofactor,
        "runtime_seconds": result.runtime_seconds,
        "iteration_count": result.iteration_count,
        "gcd_calls": result.gcd_calls,
        "modular_multiplies": result.modular_multiplies,
        "notes": result.notes,
    }
    if spec is not None:
        d["family"] = spec.family
        d["p"] = spec.p
        d["q"] = spec.q
        d["balance_ratio"] = spec.balance_ratio
    return d


def results_to_dicts(
    pairs: list[tuple[FactoringResult, SemiprimeSpec | None]],
) -> list[dict[str, Any]]:
    """Convert a list of (result, spec) pairs to flat dicts."""
    return [result_to_dict(r, s) for r, s in pairs]


def export_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write results to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else RESULT_FIELDS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
