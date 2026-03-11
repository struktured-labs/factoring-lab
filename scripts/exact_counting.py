#!/usr/bin/env python3
"""Exact lattice point counting for small semiprimes across multiple bases.

Validates (or refutes) the heuristic estimate from Lemma 4:
    |Lambda_n cap B| ~ ((b-1)^2 + 1)^{dx*dy} / b^d

Exports results to reports/exact_counting.csv.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from factoring_lab.analysis.lattice_counting import (
    LatticeCountResult,
    count_lattice_points_exact,
    to_digits,
)


def main() -> None:
    bases = [2, 3, 5, 10]
    semiprimes = [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
    ]

    results: list[dict] = []

    # Determine feasibility: base-2 with large n can have many digits
    # and huge search spaces. We'll set a timeout per case.
    MAX_SECONDS = 120  # 2 minutes per case

    print("=" * 100)
    print("EXACT LATTICE POINT COUNTING FOR CARRY-PROPAGATION LATTICE")
    print("=" * 100)
    print()
    print(f"{'n':>6} {'(p,q)':>10} {'base':>5} {'d':>3} {'dx':>3} {'dy':>3} "
          f"{'#z_vars':>7} {'exact':>12} {'heuristic':>14} "
          f"{'ratio':>10} {'log2_ex':>8} {'log2_h':>8} {'rank1':>6} {'time_s':>8}")
    print("-" * 100)

    for base in bases:
        for n, p, q in semiprimes:
            c = to_digits(n, base)
            d = len(c)

            # Estimate feasibility: the search space per position k is
            # product of ((b-1)^2 + 1) for each z_{ij} at that position.
            # For base 10, even d=3 can have huge spaces. Be conservative.
            dx = (d + 1) // 2 + 1
            dy = (d + 1) // 2 + 1
            num_z_at_0 = min(1, dx) * min(1, dy)  # rough

            # Quick feasibility check: total z-variable space
            max_z = (base - 1) ** 2 + 1
            total_z = 0
            for i in range(dx):
                for j in range(dy):
                    if i + j < d:
                        total_z += 1
            estimated_space = max_z ** total_z / base ** d

            # Skip if estimated search space is way too large
            # The actual enumeration is digit-by-digit so it's much smaller,
            # but for safety:
            if estimated_space > 1e15:
                row = {
                    "n": n, "p": p, "q": q, "base": base, "d": d,
                    "dx": dx, "dy": dy, "num_z_vars": total_z,
                    "exact": "SKIPPED", "heuristic": f"{estimated_space:.2e}",
                    "ratio": "N/A", "log2_exact": "N/A",
                    "log2_heuristic": f"{estimated_space:.2e}",
                    "rank1": "N/A", "time_s": 0,
                }
                results.append(row)
                print(f"{n:>6} ({p},{q}){' '*(7-len(f'({p},{q})'))} {base:>5} "
                      f"{d:>3} {dx:>3} {dy:>3} {total_z:>7} "
                      f"{'SKIPPED':>12} {estimated_space:>14.2e} "
                      f"{'N/A':>10} {'N/A':>8} "
                      f"{estimated_space:>8.1e} {'N/A':>6} {'--':>8}")
                continue

            t0 = time.time()
            try:
                result = count_lattice_points_exact(n, base, dx, dy)
                elapsed = time.time() - t0
            except Exception as e:
                elapsed = time.time() - t0
                print(f"{n:>6} ({p},{q}){' '*(7-len(f'({p},{q})'))} {base:>5} "
                      f"{d:>3} {dx:>3} {dy:>3} {total_z:>7} "
                      f"{'ERROR':>12} {'':>14} "
                      f"{'':>10} {'':>8} {'':>8} {'':>6} {elapsed:>8.2f}")
                results.append({
                    "n": n, "p": p, "q": q, "base": base, "d": d,
                    "dx": dx, "dy": dy, "num_z_vars": total_z,
                    "exact": f"ERROR: {e}", "heuristic": "N/A",
                    "ratio": "N/A", "log2_exact": "N/A",
                    "log2_heuristic": "N/A", "rank1": "N/A",
                    "time_s": elapsed,
                })
                continue

            row = {
                "n": n, "p": p, "q": q, "base": base, "d": d,
                "dx": result.dx, "dy": result.dy,
                "num_z_vars": result.num_z_vars,
                "exact": result.total_lattice_points,
                "heuristic": f"{result.heuristic_estimate:.2f}",
                "ratio": f"{result.ratio_exact_over_heuristic:.4f}",
                "log2_exact": f"{result.log2_exact:.2f}",
                "log2_heuristic": f"{result.log2_heuristic:.2f}",
                "rank1": result.rank1_points,
                "time_s": f"{elapsed:.2f}",
            }
            results.append(row)

            factorizations = result.rank1_factorizations
            fact_str = ", ".join(f"{a}*{b}" for a, b in factorizations) if factorizations else "none"

            print(f"{n:>6} ({p},{q}){' '*(7-len(f'({p},{q})'))} {base:>5} "
                  f"{d:>3} {result.dx:>3} {result.dy:>3} {result.num_z_vars:>7} "
                  f"{result.total_lattice_points:>12} {result.heuristic_estimate:>14.2f} "
                  f"{result.ratio_exact_over_heuristic:>10.4f} "
                  f"{result.log2_exact:>8.2f} {result.log2_heuristic:>8.2f} "
                  f"{result.rank1_points:>6} {elapsed:>8.2f}")

            if elapsed > MAX_SECONDS:
                print(f"  [TIMEOUT WARNING: took {elapsed:.1f}s, skipping larger cases for base {base}]")
                break

    print()
    print("=" * 100)

    # Summary analysis
    print("\nSUMMARY ANALYSIS")
    print("-" * 60)
    computed = [r for r in results if isinstance(r.get("exact"), int)]

    if computed:
        print(f"\nComputed {len(computed)} exact counts.")
        print("\nRank-1 check (should be 2 for each semiprime):")
        for r in computed:
            status = "OK" if r["rank1"] == 2 else f"UNEXPECTED ({r['rank1']})"
            print(f"  n={r['n']}, base={r['base']}: rank1={r['rank1']} [{status}]")

        print("\nRatio (exact/heuristic) distribution:")
        ratios = [float(r["ratio"]) for r in computed]
        print(f"  min={min(ratios):.4f}, max={max(ratios):.4f}, "
              f"mean={sum(ratios)/len(ratios):.4f}")

        print("\nGrowth rate analysis (log2 of exact count):")
        by_base: dict[int, list[tuple[int, float]]] = {}
        for r in computed:
            b = r["base"]
            if b not in by_base:
                by_base[b] = []
            by_base[b].append((r["d"], float(r["log2_exact"])))
        for b, pairs in sorted(by_base.items()):
            pairs.sort()
            print(f"  base {b}: " + ", ".join(f"d={d}->log2={v:.1f}" for d, v in pairs))

    # Export CSV
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_path = reports_dir / "exact_counting.csv"

    fieldnames = ["n", "p", "q", "base", "d", "dx", "dy", "num_z_vars",
                  "exact", "heuristic", "ratio", "log2_exact", "log2_heuristic",
                  "rank1", "time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults exported to {csv_path}")


if __name__ == "__main__":
    main()
