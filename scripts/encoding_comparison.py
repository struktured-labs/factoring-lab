#!/usr/bin/env python3
"""Head-to-head comparison: Circuit SAT vs SMT Convolution encodings.

For each bit size, generates a balanced semiprime (seed=42) and runs:
  1. CircuitSAT          — standard boolean array multiplier encoding
  2. SMTConvolutionRaw   — Z3 bitvector multiplication, no digit structure
  3. SMTConvolution(base) — digit convolution with best power-of-2 base

Records runtime, success, and variable counts; exports CSV and prints table.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factoring_lab.generators.semiprimes import balanced_semiprime
from factoring_lab.algorithms.circuit_sat import CircuitSAT
from factoring_lab.algorithms.smt_convolution import SMTConvolution, SMTConvolutionRaw

# ── Configuration ──────────────────────────────────────────────────────
BIT_SIZES = [12, 16, 20, 24, 28, 32]
SEED = 42
TIMEOUT_MS = 30_000  # 30 seconds


def best_power_of_2_base(n_bits: int) -> int:
    """Pick a reasonable power-of-2 base for digit convolution.

    Heuristic: use base = 2^(n_bits // 4), clamped to [4, 256].
    This gives enough digits to constrain without exploding constraint count.
    """
    exp = max(2, n_bits // 4)
    return min(2 ** exp, 256)


def count_z3_variables(solver) -> int:
    """Attempt to count variables in the solver's assertions."""
    try:
        from z3 import is_const, is_bool, BoolSort
        seen = set()

        def _walk(expr):
            if expr.num_args() == 0:
                if is_const(expr) and expr.sort() == BoolSort():
                    name = str(expr)
                    if name not in ("True", "False"):
                        seen.add(name)
            else:
                for i in range(expr.num_args()):
                    _walk(expr.arg(i))

        for a in solver.assertions():
            try:
                _walk(a)
            except Exception:
                pass
        return len(seen)
    except Exception:
        return -1


def run_one(algo, n: int) -> dict:
    """Run a single algorithm and return result dict."""
    t0 = time.perf_counter()
    result = algo.factor(n)
    elapsed = time.perf_counter() - t0
    return {
        "encoding": result.algorithm_name,
        "runtime_seconds": round(elapsed, 4),
        "success": result.success,
        "factor": result.factor,
        "notes": result.notes,
    }


def main() -> None:
    # Generate all semiprimes upfront
    semiprimes = {}
    print("=== Generating semiprimes (seed=42) ===")
    for bits in BIT_SIZES:
        sp = balanced_semiprime(bits=bits, seed=SEED)
        semiprimes[bits] = sp
        print(f"  {bits:>2d}-bit: N={sp.n:>12d}  (p={sp.p}, q={sp.q})")
    print()

    rows: list[dict] = []

    # Header
    fmt = "{encoding:<28s} {bits:>4s} {N:>14s} {runtime:>10s} {success:>7s} {factor:>12s}"
    print(fmt.format(
        encoding="encoding", bits="bits", N="N",
        runtime="runtime_s", success="success", factor="factor"
    ))
    print("-" * 82)

    for bits in BIT_SIZES:
        sp = semiprimes[bits]
        base = best_power_of_2_base(bits)

        algos = [
            CircuitSAT(timeout_ms=TIMEOUT_MS),
            SMTConvolutionRaw(timeout_ms=TIMEOUT_MS),
            SMTConvolution(base=base, timeout_ms=TIMEOUT_MS),
        ]

        for algo in algos:
            info = run_one(algo, sp.n)
            info["bits"] = bits
            info["N"] = sp.n
            rows.append(info)

            print(fmt.format(
                encoding=info["encoding"],
                bits=str(bits),
                N=str(sp.n),
                runtime=f"{info['runtime_seconds']:.4f}",
                success=str(info["success"]),
                factor=str(info["factor"]),
            ))

        print()  # blank line between bit sizes

    # Summary table: fastest per bit size
    print("\n=== Summary: fastest encoding per bit size ===")
    for bits in BIT_SIZES:
        bit_rows = [r for r in rows if r["bits"] == bits and r["success"]]
        if bit_rows:
            best = min(bit_rows, key=lambda r: r["runtime_seconds"])
            print(f"  {bits:>2d}-bit: {best['encoding']:<28s} {best['runtime_seconds']:.4f}s")
        else:
            print(f"  {bits:>2d}-bit: all timed out or failed")

    # Export CSV
    csv_path = Path(__file__).resolve().parent.parent / "reports" / "encoding_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["encoding", "bits", "N", "runtime_seconds", "success", "factor", "notes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults exported to {csv_path}")


if __name__ == "__main__":
    main()
