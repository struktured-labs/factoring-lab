#!/usr/bin/env python3
"""SOS/Lasserre hierarchy experiment for digit convolution factoring.

Measures the SOS gap at degrees 2 and 4 across varying bit sizes and bases.
Exports results to reports/sos_experiment.csv.
"""

from __future__ import annotations

import csv
import os
import sys
import time

# Ensure the project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from factoring_lab.algorithms.sos_relaxation import run_sos_relaxation, HAS_CVXPY
from factoring_lab.generators.semiprimes import balanced_semiprime


def main() -> None:
    if not HAS_CVXPY:
        print("ERROR: cvxpy is required but not installed.")
        print("Install with: uv pip install cvxpy")
        sys.exit(1)

    print("=" * 80)
    print("SOS / Lasserre Hierarchy Experiment")
    print("=" * 80)
    print()

    # Parameters
    bit_sizes = [8, 10, 12, 14, 16]
    bases = [2, 3, 5, 10]
    degrees = [2, 4]
    seed = 42

    results: list[dict] = []

    for bits in bit_sizes:
        spec = balanced_semiprime(bits, seed=seed)
        n = spec.n
        p = spec.p
        q = spec.q
        print(f"--- {bits}-bit semiprime: n={n} = {p} * {q} ---")

        for base in bases:
            for degree in degrees:
                label = f"  base={base}, degree={degree}"
                print(f"{label} ... ", end="", flush=True)

                t0 = time.perf_counter()
                result = run_sos_relaxation(n, base, degree, known_p=p, known_q=q)
                wall_time = time.perf_counter() - t0

                row = {
                    "bits": bits,
                    "n": n,
                    "p": p,
                    "q": q,
                    "base": base,
                    "degree": degree,
                    "num_digit_vars": result.num_digit_vars,
                    "moment_matrix_size": result.moment_matrix_size,
                    "solve_time_s": round(result.solve_time_seconds, 4),
                    "wall_time_s": round(wall_time, 4),
                    "solver_status": result.solver_status,
                    "sos_gap": round(result.sos_gap, 6) if result.sos_gap < float("inf") else "inf",
                    "recovery_success": result.recovery_success,
                    "recovered_p": result.recovered_p,
                    "recovered_q": result.recovered_q,
                    "objective_value": (
                        round(result.objective_value, 4)
                        if result.objective_value < float("inf")
                        else "inf"
                    ),
                    "top_eigenvalue": (
                        round(result.eigenvalues_top5[0], 4)
                        if result.eigenvalues_top5
                        else ""
                    ),
                    "notes": result.notes,
                }
                results.append(row)

                status_str = "OK" if result.solver_status == "optimal" else result.solver_status
                gap_str = f"gap={result.sos_gap:.4f}" if result.sos_gap < float("inf") else "gap=inf"
                recover_str = "RECOVERED" if result.recovery_success else "no"
                time_str = f"{wall_time:.2f}s"
                print(f"{status_str}, {gap_str}, recover={recover_str}, {time_str}")

                if result.notes:
                    print(f"    note: {result.notes}")

        print()

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'bits':>5} {'base':>5} {'deg':>4} {'mm_size':>8} {'gap':>10} {'recover':>8} {'time':>8} {'status'}")
    print("-" * 80)

    for r in results:
        gap_str = f"{r['sos_gap']:.4f}" if r["sos_gap"] != "inf" else "inf"
        print(
            f"{r['bits']:>5} {r['base']:>5} {r['degree']:>4} "
            f"{r['moment_matrix_size']:>8} {gap_str:>10} "
            f"{'YES' if r['recovery_success'] else 'no':>8} "
            f"{r['wall_time_s']:>8.2f} {r['solver_status']}"
        )

    # Export CSV
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, "sos_experiment.csv")

    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults exported to {csv_path}")

    # Analysis summary
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    deg2_results = [r for r in results if r["degree"] == 2 and r["sos_gap"] != "inf"]
    deg4_results = [r for r in results if r["degree"] == 4 and r["sos_gap"] != "inf"]

    if deg2_results:
        avg_gap2 = sum(float(r["sos_gap"]) for r in deg2_results) / len(deg2_results)
        n_recover2 = sum(1 for r in deg2_results if r["recovery_success"])
        print(f"Degree-2 SOS: avg gap = {avg_gap2:.4f}, recovered = {n_recover2}/{len(deg2_results)}")

    if deg4_results:
        avg_gap4 = sum(float(r["sos_gap"]) for r in deg4_results) / len(deg4_results)
        n_recover4 = sum(1 for r in deg4_results if r["recovery_success"])
        print(f"Degree-4 SOS: avg gap = {avg_gap4:.4f}, recovered = {n_recover4}/{len(deg4_results)}")

    if deg2_results and deg4_results:
        improvement = avg_gap2 - avg_gap4
        print(f"Gap improvement (deg2 -> deg4): {improvement:.4f}")

    # Per-base analysis
    for base in bases:
        base_results = [r for r in results if r["base"] == base and r["sos_gap"] != "inf"]
        if base_results:
            avg_gap = sum(float(r["sos_gap"]) for r in base_results) / len(base_results)
            n_recover = sum(1 for r in base_results if r["recovery_success"])
            print(f"  Base {base}: avg gap = {avg_gap:.4f}, recovered = {n_recover}/{len(base_results)}")


if __name__ == "__main__":
    main()
