#!/usr/bin/env python3
"""Multi-instance classical benchmark: 50 instances per bit size.

Runs TrialDivision, PollardRho, PollardPM1(B=10000), and ECM against
50 balanced semiprimes (seeds 0-49) for bit sizes [16, 20, 24, 28, 32].
Exports to reports/multi_instance_classical.csv and prints summary stats.
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factoring_lab.generators.semiprimes import balanced_semiprime
from factoring_lab.algorithms import TrialDivision, PollardRho, PollardPM1, ECM

BIT_SIZES = [16, 20, 24, 28, 32]
NUM_SEEDS = 50
ALGORITHMS = [
    TrialDivision(),
    PollardRho(),
    PollardPM1(bound=10000),
    ECM(),
]


def main() -> None:
    rows: list[dict] = []

    for bits in BIT_SIZES:
        print(f"\n=== {bits}-bit semiprimes ===")
        for seed in range(NUM_SEEDS):
            sp = balanced_semiprime(bits=bits, seed=seed)
            for algo in ALGORITHMS:
                t0 = time.perf_counter()
                result = algo.factor(sp.n)
                elapsed = time.perf_counter() - t0

                row = {
                    "bits": bits,
                    "seed": seed,
                    "algorithm": algo.name,
                    "n": sp.n,
                    "success": result.success,
                    "runtime_seconds": round(elapsed, 6),
                    "iteration_count": result.iteration_count,
                    "gcd_calls": result.gcd_calls,
                    "modular_multiplies": result.modular_multiplies,
                }
                rows.append(row)

            if (seed + 1) % 10 == 0:
                print(f"  Completed seed {seed + 1}/{NUM_SEEDS}")

    # Export CSV
    csv_path = Path(__file__).resolve().parent.parent / "reports" / "multi_instance_classical.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bits", "seed", "algorithm", "n", "success",
        "runtime_seconds", "iteration_count", "gcd_calls", "modular_multiplies",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nExported {len(rows)} rows to {csv_path}")

    # Summary stats
    print("\n" + "=" * 100)
    print(f"{'bits':>4s}  {'algorithm':>20s}  {'success_rate':>12s}  {'mean_t':>10s}  "
          f"{'median_t':>10s}  {'std_t':>10s}  {'min_t':>10s}  {'max_t':>10s}")
    print("-" * 100)

    for bits in BIT_SIZES:
        for algo in ALGORITHMS:
            algo_rows = [r for r in rows if r["bits"] == bits and r["algorithm"] == algo.name]
            successes = sum(1 for r in algo_rows if r["success"])
            runtimes = [r["runtime_seconds"] for r in algo_rows]
            n = len(algo_rows)
            success_rate = successes / n if n else 0.0
            mean_t = statistics.mean(runtimes)
            median_t = statistics.median(runtimes)
            std_t = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
            min_t = min(runtimes)
            max_t = max(runtimes)

            print(f"{bits:>4d}  {algo.name:>20s}  {success_rate:>11.1%}  {mean_t:>10.6f}  "
                  f"{median_t:>10.6f}  {std_t:>10.6f}  {min_t:>10.6f}  {max_t:>10.6f}")
        print("-" * 100)


if __name__ == "__main__":
    main()
