"""Benchmark: Python vs Rust digit convolution factoring.

Generates semiprimes of various bit sizes and compares runtime and
iteration counts between the pure-Python and Rust implementations.

Exports results to reports/rust_vs_python.csv.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from sympy import isprime

from factoring_lab.algorithms.digit_convolution import DigitConvolution
from factoring_lab.algorithms.digit_convolution_rs import DigitConvolutionRust


def random_prime(bits: int) -> int:
    """Generate a random prime with the given number of bits."""
    while True:
        # Ensure the high bit is set so we get exactly `bits` bits
        n = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if isprime(n):
            return n


def generate_semiprime(bits: int) -> tuple[int, int, int]:
    """Generate a semiprime of approximately `bits` total bits.

    Each prime factor has approximately bits/2 bits.
    """
    half = bits // 2
    p = random_prime(half)
    q = random_prime(bits - half)
    return p * q, p, q


def benchmark_one(n: int, base: int = 10, timeout: float = 30.0) -> dict:
    """Run both Python and Rust on n, return timing data."""
    row = {"n": n, "bits": n.bit_length(), "base": base}

    # Python
    py_algo = DigitConvolution(base=base)
    t0 = time.perf_counter()
    py_result = py_algo.factor(n)
    py_time = time.perf_counter() - t0
    row["py_success"] = py_result.success
    row["py_factor"] = py_result.factor
    row["py_time_s"] = py_time
    row["py_iterations"] = py_result.iteration_count

    # Rust
    rs_algo = DigitConvolutionRust(base=base)
    t0 = time.perf_counter()
    rs_result = rs_algo.factor(n)
    rs_time = time.perf_counter() - t0
    row["rs_success"] = rs_result.success
    row["rs_factor"] = rs_result.factor
    row["rs_time_s"] = rs_time
    row["rs_iterations"] = rs_result.iteration_count

    # Speedup
    if rs_time > 0:
        row["speedup"] = py_time / rs_time
    else:
        row["speedup"] = float("inf")

    return row


def main():
    random.seed(42)
    bit_sizes = [12, 16, 20, 24]
    samples_per_size = 5
    base = 10

    rows = []
    for bits in bit_sizes:
        print(f"\n--- {bits}-bit semiprimes (base {base}) ---")
        for i in range(samples_per_size):
            n, p, q = generate_semiprime(bits)
            print(f"  [{i+1}/{samples_per_size}] n={n} ({n.bit_length()} bits) = {p} * {q}")
            row = benchmark_one(n, base=base)
            row["p"] = p
            row["q"] = q
            print(
                f"    Python: {row['py_time_s']:.6f}s  ({row['py_iterations']} iters)"
                f"  |  Rust: {row['rs_time_s']:.6f}s  ({row['rs_iterations']} iters)"
                f"  |  Speedup: {row['speedup']:.1f}x"
            )
            rows.append(row)

    # Write CSV
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rust_vs_python.csv"
    fields = [
        "bits", "n", "p", "q", "base",
        "py_success", "py_factor", "py_time_s", "py_iterations",
        "rs_success", "rs_factor", "rs_time_s", "rs_iterations",
        "speedup",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path}")

    # Summary
    print("\n=== Summary ===")
    for bits in bit_sizes:
        subset = [r for r in rows if r["bits"] >= bits - 1 and r["bits"] <= bits + 1]
        if not subset:
            continue
        avg_py = sum(r["py_time_s"] for r in subset) / len(subset)
        avg_rs = sum(r["rs_time_s"] for r in subset) / len(subset)
        avg_speedup = sum(r["speedup"] for r in subset) / len(subset)
        print(
            f"  {bits}-bit: Python avg {avg_py:.6f}s, Rust avg {avg_rs:.6f}s, "
            f"avg speedup {avg_speedup:.1f}x"
        )


if __name__ == "__main__":
    main()
