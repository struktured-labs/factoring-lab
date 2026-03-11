#!/usr/bin/env python3
"""Extended base sweep: larger bit sizes with power-of-2 bases.

Tests bit sizes [20, 24, 28, 32, 36, 40] with power-of-2 bases
[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] plus raw (no-base) control.
Uses 30-second timeout and seed=42 for reproducibility.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factoring_lab.generators.semiprimes import balanced_semiprime
from factoring_lab.algorithms.smt_convolution import SMTConvolution, SMTConvolutionRaw

# ── Configuration ──────────────────────────────────────────────────────
BIT_SIZES = [20, 24, 28, 32, 36, 40]
BASES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEED = 42
TIMEOUT_MS = 30_000  # 30 seconds


def main() -> None:
    # Generate semiprimes
    semiprimes = {}
    print("=== Generating semiprimes ===")
    for bits in BIT_SIZES:
        sp = balanced_semiprime(bits=bits, seed=SEED)
        semiprimes[bits] = sp
        print(f"  {bits}-bit: N={sp.n}  (p={sp.p}, q={sp.q})")
    print()

    # Prepare results
    rows: list[dict] = []

    header = f"{'base':>6s}  {'bits':>4s}  {'N':>20s}  {'runtime_s':>10s}  {'success':>7s}"
    sep = "-" * 60
    print(header)
    print(sep)

    # Run raw control for each semiprime
    for bits in BIT_SIZES:
        sp = semiprimes[bits]
        algo = SMTConvolutionRaw(timeout_ms=TIMEOUT_MS)
        t0 = time.perf_counter()
        result = algo.factor(sp.n)
        elapsed = time.perf_counter() - t0

        row = {
            "bits": bits,
            "base": "raw",
            "runtime_seconds": round(elapsed, 4),
            "success": result.success,
        }
        rows.append(row)
        print(
            f"{'raw':>6s}  {bits:>4d}  {sp.n:>20d}  {elapsed:>10.4f}  {str(result.success):>7s}"
        )

    # Run each base
    for base in BASES:
        for bits in BIT_SIZES:
            sp = semiprimes[bits]
            algo = SMTConvolution(base=base, timeout_ms=TIMEOUT_MS)
            t0 = time.perf_counter()
            result = algo.factor(sp.n)
            elapsed = time.perf_counter() - t0

            row = {
                "bits": bits,
                "base": base,
                "runtime_seconds": round(elapsed, 4),
                "success": result.success,
            }
            rows.append(row)
            print(
                f"{base:>6d}  {bits:>4d}  {sp.n:>20d}  {elapsed:>10.4f}  {str(result.success):>7s}"
            )

    print(sep)
    print(f"Total runs: {len(rows)}")

    # Export to CSV
    csv_path = Path(__file__).resolve().parent.parent / "reports" / "extended_base_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bits", "base", "runtime_seconds", "success"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults exported to {csv_path}")


if __name__ == "__main__":
    main()
