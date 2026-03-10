"""Experiment runner for factoring benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from factoring_lab.algorithms.base import FactoringAlgorithm, FactoringResult
from factoring_lab.generators.semiprimes import SemiprimeSpec, generate_family
from factoring_lab.metrics.export import result_to_dict


@dataclass
class ExperimentConfig:
    """Specification for a benchmark experiment."""

    name: str
    family: str
    bits: int
    count: int
    algorithms: list[FactoringAlgorithm]
    seed: int = 42
    generator_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a complete experiment run."""

    config: ExperimentConfig
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run all algorithms against all instances in an experiment."""
    rows: list[dict[str, Any]] = []
    success_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}

    instances = list(
        generate_family(
            family=config.family,
            bits=config.bits,
            count=config.count,
            seed=config.seed,
            **config.generator_kwargs,
        )
    )

    for spec in instances:
        for algo in config.algorithms:
            result = algo.factor(spec.n)
            row = result_to_dict(result, spec)
            row["experiment"] = config.name
            rows.append(row)

            name = algo.name
            total_counts[name] = total_counts.get(name, 0) + 1
            if result.success:
                success_counts[name] = success_counts.get(name, 0) + 1

    summary: dict[str, Any] = {
        "experiment": config.name,
        "family": config.family,
        "bits": config.bits,
        "instance_count": config.count,
        "algorithms": {},
    }
    for name in total_counts:
        total = total_counts[name]
        successes = success_counts.get(name, 0)
        algo_rows = [r for r in rows if r["algorithm_name"] == name]
        runtimes = [r["runtime_seconds"] for r in algo_rows]
        summary["algorithms"][name] = {
            "success_rate": successes / total if total > 0 else 0.0,
            "successes": successes,
            "total": total,
            "avg_runtime": sum(runtimes) / len(runtimes) if runtimes else 0.0,
            "total_runtime": sum(runtimes),
        }

    return ExperimentResult(config=config, rows=rows, summary=summary)
