#!/usr/bin/env python3
"""Sweep bases 2..512 on fixed semiprimes and profile Z3 runtime.

Generates balanced semiprimes at 20, 24, 28, and 32 bits (seed=42),
then runs SMTConvolution with various bases plus a raw (no-base) control.
Records runtime, success, and exports results to CSV.
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
BIT_SIZES = [20, 24, 28, 32]
BASES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 32, 64, 100, 128, 256, 512]
SEED = 42
TIMEOUT_MS = 15_000  # 15 seconds


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

    header = f"{'base':>6s}  {'bits':>4s}  {'N':>14s}  {'runtime_s':>10s}  {'success':>7s}  {'factor':>10s}  {'notes'}"
    sep = "-" * 80
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
            "base": "raw",
            "bits": bits,
            "N": sp.n,
            "runtime_seconds": round(elapsed, 4),
            "success": result.success,
            "factor": result.factor,
            "notes": result.notes,
        }
        rows.append(row)
        print(
            f"{'raw':>6s}  {bits:>4d}  {sp.n:>14d}  {elapsed:>10.4f}  {str(result.success):>7s}  "
            f"{str(result.factor):>10s}  {result.notes}"
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
                "base": base,
                "bits": bits,
                "N": sp.n,
                "runtime_seconds": round(elapsed, 4),
                "success": result.success,
                "factor": result.factor,
                "notes": result.notes,
            }
            rows.append(row)
            print(
                f"{base:>6d}  {bits:>4d}  {sp.n:>14d}  {elapsed:>10.4f}  {str(result.success):>7s}  "
                f"{str(result.factor):>10s}  {result.notes}"
            )

    print(sep)
    print(f"Total runs: {len(rows)}")

    # Export to CSV
    csv_path = Path(__file__).resolve().parent.parent / "reports" / "base_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["base", "bits", "N", "runtime_seconds", "success", "factor", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults exported to {csv_path}")


if __name__ == "__main__":
    main()
