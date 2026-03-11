#!/usr/bin/env python3
"""Experiment: hybrid digit-convolution + Coppersmith factoring.

For several bit sizes, generate balanced semiprimes and test the hybrid
approach at different enumeration depths. Compare with SMTLeakedBits at
equivalent leak fractions.

Outputs: reports/hybrid_coppersmith.csv
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

# Ensure the package is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from factoring_lab.algorithms.hybrid_coppersmith import HybridCoppersmith
from factoring_lab.generators.semiprimes import balanced_semiprime

BIT_SIZES = [32, 48, 64]
NUM_SAMPLES = 10
DEPTHS = [1, 2, 3]
BASES = [10]
TIMEOUT_S = 30.0


def equivalent_leak_fraction(base: int, depth: int, factor_bits: int) -> float:
    """Estimate the leak fraction equivalent to knowing `depth` digits in `base`.

    Knowing depth digits in base b gives us log2(b^depth) bits of information,
    which is equivalent to leaking that many bits out of factor_bits total.
    """
    known_bits = depth * math.log2(base)
    return min(1.0, known_bits / factor_bits)


def run_experiment() -> list[dict]:
    rows: list[dict] = []

    for bits in BIT_SIZES:
        print(f"\n{'='*70}")
        print(f"  BIT SIZE: {bits}")
        print(f"{'='*70}")

        for sample_idx in range(NUM_SAMPLES):
            seed = 42 + sample_idx
            spec = balanced_semiprime(bits, seed=seed)
            n = spec.n
            p = spec.p
            q = spec.q
            factor_bits = p.bit_length()

            print(f"\n  Sample {sample_idx+1}/{NUM_SAMPLES}: "
                  f"n={n} ({n.bit_length()}b), p={p} ({p.bit_length()}b), q={q}")

            for depth in DEPTHS:
                for base in BASES:
                    algo = HybridCoppersmith(
                        base=base,
                        depth=depth,
                        timeout_s=TIMEOUT_S,
                    )

                    t0 = time.perf_counter()
                    result = algo.factor(n)
                    elapsed = time.perf_counter() - t0

                    leak_equiv = equivalent_leak_fraction(base, depth, factor_bits)

                    status = "OK" if result.success else "FAIL"
                    print(
                        f"    base={base:2d} depth={depth}  "
                        f"leak_equiv={leak_equiv:.2f}  "
                        f"{status:4s}  {elapsed:7.2f}s"
                        + (f"  factor={result.factor}" if result.success else "")
                    )

                    rows.append({
                        "bits": bits,
                        "sample": sample_idx,
                        "n": n,
                        "p": p,
                        "q": q,
                        "base": base,
                        "depth": depth,
                        "leak_fraction_equiv": round(leak_equiv, 4),
                        "success": result.success,
                        "factor_found": result.factor if result.success else None,
                        "runtime_s": round(elapsed, 4),
                        "iterations": result.iteration_count,
                        "notes": result.notes,
                    })

    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"\n\n{'='*80}")
    print("SUMMARY: Success rates by bit size and depth")
    print(f"{'='*80}")

    for bits in BIT_SIZES:
        print(f"\n  {bits}-bit semiprimes:")
        for depth in DEPTHS:
            bit_depth_rows = [
                r for r in rows
                if r["bits"] == bits and r["depth"] == depth
            ]
            if not bit_depth_rows:
                continue
            successes = sum(1 for r in bit_depth_rows if r["success"])
            total = len(bit_depth_rows)
            avg_time = sum(r["runtime_s"] for r in bit_depth_rows) / total
            leak_equiv = bit_depth_rows[0]["leak_fraction_equiv"]
            print(
                f"    depth={depth} (leak_equiv={leak_equiv:.2f}): "
                f"{successes}/{total} success, avg {avg_time:.2f}s"
            )


def export_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "bits", "sample", "n", "p", "q", "base", "depth",
        "leak_fraction_equiv", "success", "factor_found",
        "runtime_s", "iterations", "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults exported to {path}")


def main() -> None:
    rows = run_experiment()
    print_summary(rows)

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "reports", "hybrid_coppersmith.csv"
    )
    export_csv(rows, csv_path)


if __name__ == "__main__":
    main()
