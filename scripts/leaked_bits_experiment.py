#!/usr/bin/env python3
"""Experiment: how does partial bit leaking affect SMT factoring scalability?

For each target bit size, generate a balanced semiprime and test increasing
fractions of leaked LSBs of p.  Records success/failure and runtime.
"""

from __future__ import annotations

import csv
import os
import sys
import time

# Ensure the package is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from factoring_lab.algorithms.smt_leaked import SMTLeakedBits
from factoring_lab.generators.semiprimes import balanced_semiprime

BIT_SIZES = [32, 48, 64, 80, 96, 128]
LEAK_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TIMEOUT_MS = 30_000
SEED = 42


def run_experiment() -> list[dict]:
    rows: list[dict] = []

    for bits in BIT_SIZES:
        spec = balanced_semiprime(bits, seed=SEED)
        p = spec.p
        print(f"\n{'='*60}")
        print(f"Bit size: {bits}  |  n = {spec.n}  |  p = {spec.p}  |  q = {spec.q}")
        print(f"p has {p.bit_length()} bits, q has {spec.q.bit_length()} bits")
        print(f"{'='*60}")

        for frac in LEAK_FRACTIONS:
            solver = SMTLeakedBits(
                leak_fraction=frac,
                known_p=p,
                timeout_ms=TIMEOUT_MS,
            )

            t0 = time.perf_counter()
            result = solver.factor(spec.n)
            elapsed = time.perf_counter() - t0

            status = "OK" if result.success else "FAIL"
            num_leaked = max(1, int(p.bit_length() * frac)) if frac > 0 else 0
            print(
                f"  leak={frac:.1f} ({num_leaked:3d}/{p.bit_length()} bits)  "
                f"{status:4s}  {elapsed:7.2f}s"
                + (f"  factor={result.factor}" if result.success else "")
            )

            rows.append(
                {
                    "bits": bits,
                    "leak_fraction": frac,
                    "num_leaked_bits": num_leaked,
                    "p_bits": p.bit_length(),
                    "success": result.success,
                    "runtime_s": round(elapsed, 4),
                    "factor": result.factor if result.success else None,
                }
            )

    return rows


def print_table(rows: list[dict]) -> None:
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    header = f"{'bits':>5} {'leak':>5} {'leaked_bits':>11} {'success':>7} {'runtime':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['bits']:>5} {r['leak_fraction']:>5.1f} "
            f"{r['num_leaked_bits']:>5}/{r['p_bits']:<5} "
            f"{'YES' if r['success'] else 'no':>7} "
            f"{r['runtime_s']:>9.2f}s"
        )


def export_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults exported to {path}")


def main() -> None:
    rows = run_experiment()
    print_table(rows)

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "reports", "leaked_bits.csv"
    )
    export_csv(rows, csv_path)

    # Summary: minimum leak fraction for each bit size
    print(f"\n{'='*60}")
    print("MINIMUM LEAK FRACTION FOR SUCCESS")
    print(f"{'='*60}")
    for bits in BIT_SIZES:
        bit_rows = [r for r in rows if r["bits"] == bits and r["success"]]
        if bit_rows:
            min_frac = min(r["leak_fraction"] for r in bit_rows)
            print(f"  {bits:>3}-bit:  {min_frac:.1f}  ({int(min_frac * bit_rows[0]['p_bits'])} of {bit_rows[0]['p_bits']} bits)")
        else:
            print(f"  {bits:>3}-bit:  UNSOLVABLE (within {TIMEOUT_MS/1000:.0f}s timeout)")


if __name__ == "__main__":
    main()
