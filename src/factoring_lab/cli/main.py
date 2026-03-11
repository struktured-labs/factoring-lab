"""Command-line interface for factoring-lab."""

from __future__ import annotations

import argparse
import json
import sys

from factoring_lab.algorithms import TrialDivision, PollardRho, PollardPM1, ECM
from factoring_lab.benchmarks.runner import ExperimentConfig, run_experiment
from factoring_lab.metrics.export import export_csv


ALL_ALGORITHMS = {
    "trial_division": TrialDivision,
    "pollard_rho": PollardRho,
    "pollard_pm1": PollardPM1,
    "ecm": ECM,
}

FAMILIES = ["balanced", "unbalanced", "smooth_pm1", "random"]


def cmd_factor(args: argparse.Namespace) -> None:
    """Factor a single integer with all algorithms."""
    n = args.n
    algos = [cls() for cls in ALL_ALGORITHMS.values()]
    print(f"Factoring N = {n} ({n.bit_length()} bits)\n")
    for algo in algos:
        result = algo.factor(n)
        status = "OK" if result.success else "FAIL"
        factor_str = str(result.factor) if result.factor else "-"
        print(
            f"  {result.algorithm_name:20s} [{status:4s}] "
            f"factor={factor_str:>12s}  "
            f"time={result.runtime_seconds:.6f}s  "
            f"iters={result.iteration_count}  "
            f"gcds={result.gcd_calls}  "
            f"muls={result.modular_multiplies}"
        )
        if result.notes:
            print(f"    notes: {result.notes}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run a benchmark experiment."""
    algo_names = args.algorithms.split(",") if args.algorithms else list(ALL_ALGORITHMS)
    algos = []
    for name in algo_names:
        name = name.strip()
        if name not in ALL_ALGORITHMS:
            print(f"Unknown algorithm: {name}", file=sys.stderr)
            sys.exit(1)
        algos.append(ALL_ALGORITHMS[name]())

    config = ExperimentConfig(
        name=args.name or f"{args.family}_{args.bits}bit",
        family=args.family,
        bits=args.bits,
        count=args.count,
        algorithms=algos,
        seed=args.seed,
    )

    print(f"Running experiment: {config.name}")
    print(f"  Family: {config.family}, Bits: {config.bits}, Count: {config.count}")
    print(f"  Algorithms: {[a.name for a in algos]}")
    print()

    result = run_experiment(config)

    # Print summary
    for algo_name, stats in result.summary["algorithms"].items():
        rate = stats["success_rate"] * 100
        print(
            f"  {algo_name:20s}  "
            f"success={rate:5.1f}%  "
            f"avg_time={stats['avg_runtime']:.6f}s"
        )

    # Export if requested
    if args.output:
        path = export_csv(result.rows, args.output)
        print(f"\nResults exported to {path}")

    if args.json:
        print(f"\nSummary:\n{json.dumps(result.summary, indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="factoring-lab",
        description="Research harness for classical integer factorization",
    )
    subparsers = parser.add_subparsers(dest="command")

    # factor command
    p_factor = subparsers.add_parser("factor", help="Factor a single integer")
    p_factor.add_argument("n", type=int, help="Integer to factor")
    p_factor.set_defaults(func=cmd_factor)

    # benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Run a benchmark experiment")
    p_bench.add_argument("--family", choices=FAMILIES, required=True)
    p_bench.add_argument("--bits", type=int, required=True)
    p_bench.add_argument("--count", type=int, default=20)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--algorithms", type=str, default=None,
                         help="Comma-separated algorithm names")
    p_bench.add_argument("--name", type=str, default=None, help="Experiment name")
    p_bench.add_argument("--output", type=str, default=None, help="CSV output path")
    p_bench.add_argument("--json", action="store_true", help="Print JSON summary")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
