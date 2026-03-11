#!/usr/bin/env python3
"""Analyze multi-instance benchmark results.

Reads both classical and SMT CSV files, computes summary statistics,
and prints a publication-ready comparison table. No matplotlib required.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    """Load CSV, converting types."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "bits": int(r["bits"]),
                "seed": int(r["seed"]),
                "algorithm": r["algorithm"],
                "n": int(r["n"]),
                "success": r["success"] == "True",
                "runtime_seconds": float(r["runtime_seconds"]),
                "iteration_count": int(r["iteration_count"]),
                "gcd_calls": int(r["gcd_calls"]),
                "modular_multiplies": int(r["modular_multiplies"]),
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    """Compute summary statistics for a group of rows."""
    n = len(rows)
    if n == 0:
        return {}
    successes = sum(1 for r in rows if r["success"])
    runtimes = [r["runtime_seconds"] for r in rows]
    iterations = [r["iteration_count"] for r in rows]
    return {
        "count": n,
        "success_rate": successes / n,
        "successes": successes,
        "mean_time": statistics.mean(runtimes),
        "median_time": statistics.median(runtimes),
        "std_time": statistics.stdev(runtimes) if n > 1 else 0.0,
        "min_time": min(runtimes),
        "max_time": max(runtimes),
        "mean_iterations": statistics.mean(iterations),
    }


def print_table(all_rows: list[dict], title: str) -> None:
    """Print a publication-ready summary table."""
    # Get unique bit sizes and algorithms, preserving order
    bit_sizes = sorted(set(r["bits"] for r in all_rows))
    algorithms = []
    for r in all_rows:
        if r["algorithm"] not in algorithms:
            algorithms.append(r["algorithm"])

    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")
    header = (
        f"{'Bits':>4s}  {'Algorithm':>25s}  {'Success':>8s}  "
        f"{'Mean(s)':>10s}  {'Med(s)':>10s}  {'Std(s)':>10s}  "
        f"{'Min(s)':>10s}  {'Max(s)':>10s}  {'Avg Iter':>10s}"
    )
    print(header)
    print("-" * 120)

    for bits in bit_sizes:
        for algo in algorithms:
            group = [r for r in all_rows if r["bits"] == bits and r["algorithm"] == algo]
            if not group:
                continue
            s = summarize(group)
            print(
                f"{bits:>4d}  {algo:>25s}  {s['success_rate']:>7.0%}  "
                f"{s['mean_time']:>10.6f}  {s['median_time']:>10.6f}  {s['std_time']:>10.6f}  "
                f"{s['min_time']:>10.6f}  {s['max_time']:>10.6f}  {s['mean_iterations']:>10.1f}"
            )
        print("-" * 120)


def print_comparison(classical: list[dict], smt: list[dict]) -> None:
    """Print cross-paradigm comparison at each bit size."""
    all_rows = classical + smt
    bit_sizes = sorted(set(r["bits"] for r in all_rows))

    print(f"\n{'=' * 100}")
    print("  CROSS-PARADIGM COMPARISON: Classical vs SMT")
    print(f"{'=' * 100}")

    for bits in bit_sizes:
        print(f"\n  --- {bits}-bit semiprimes (50 instances) ---")
        print(f"  {'Algorithm':>25s}  {'Success':>8s}  {'Mean(s)':>10s}  {'Med(s)':>10s}  {'Std(s)':>10s}")
        print(f"  {'-' * 80}")

        for source, label in [(classical, "CLASSICAL"), (smt, "SMT")]:
            group_bits = [r for r in source if r["bits"] == bits]
            algorithms = []
            for r in group_bits:
                if r["algorithm"] not in algorithms:
                    algorithms.append(r["algorithm"])

            for algo in algorithms:
                group = [r for r in group_bits if r["algorithm"] == algo]
                s = summarize(group)
                if not s:
                    continue
                print(
                    f"  {algo:>25s}  {s['success_rate']:>7.0%}  "
                    f"{s['mean_time']:>10.6f}  {s['median_time']:>10.6f}  {s['std_time']:>10.6f}"
                )


def print_variance_analysis(all_rows: list[dict]) -> None:
    """Analyze variance and outliers."""
    bit_sizes = sorted(set(r["bits"] for r in all_rows))
    algorithms = []
    for r in all_rows:
        if r["algorithm"] not in algorithms:
            algorithms.append(r["algorithm"])

    print(f"\n{'=' * 100}")
    print("  VARIANCE & OUTLIER ANALYSIS")
    print(f"{'=' * 100}")

    for bits in bit_sizes:
        for algo in algorithms:
            group = [r for r in all_rows if r["bits"] == bits and r["algorithm"] == algo]
            if not group:
                continue
            s = summarize(group)
            runtimes = sorted([r["runtime_seconds"] for r in group])

            # Coefficient of variation
            cv = (s["std_time"] / s["mean_time"]) if s["mean_time"] > 0 else 0.0

            # Check for failures
            failures = [r for r in group if not r["success"]]

            if cv > 1.0 or len(failures) > 0:
                print(f"\n  {bits}-bit / {algo}:")
                print(f"    CV = {cv:.2f} (mean={s['mean_time']:.6f}, std={s['std_time']:.6f})")
                if failures:
                    print(f"    FAILURES: {len(failures)} / {len(group)} instances")
                    for f in failures[:5]:
                        print(f"      seed={f['seed']}, n={f['n']}")
                if cv > 2.0:
                    # Show quantiles
                    q25 = runtimes[len(runtimes) // 4]
                    q75 = runtimes[3 * len(runtimes) // 4]
                    print(f"    Quantiles: Q25={q25:.6f}, Q75={q75:.6f}")


def main() -> None:
    reports = Path(__file__).resolve().parent.parent / "reports"
    classical_path = reports / "multi_instance_classical.csv"
    smt_path = reports / "multi_instance_smt.csv"

    if not classical_path.exists():
        print(f"ERROR: {classical_path} not found. Run multi_instance_benchmark.py first.")
        sys.exit(1)
    if not smt_path.exists():
        print(f"ERROR: {smt_path} not found. Run multi_instance_smt.py first.")
        sys.exit(1)

    classical = load_csv(classical_path)
    smt = load_csv(smt_path)

    print(f"Loaded {len(classical)} classical rows, {len(smt)} SMT rows")

    print_table(classical, "CLASSICAL ALGORITHMS (50 instances per bit size)")
    print_table(smt, "SMT ALGORITHMS (50 instances per bit size)")
    print_comparison(classical, smt)
    print_variance_analysis(classical + smt)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
