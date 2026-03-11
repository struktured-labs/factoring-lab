"""Compare SDP/alternating projection vs backtracking vs Z3 across bit sizes.

Generates semiprimes of various bit sizes and compares factoring approaches.
"""

import random
import time
import math
from sympy import isprime, nextprime

from factoring_lab.algorithms.sdp_convolution import (
    SDPConvolution,
    AlternatingProjection,
    SDPAnalysis,
)
from factoring_lab.algorithms.digit_convolution import DigitConvolution


def generate_semiprime(bits: int, rng: random.Random) -> tuple[int, int, int]:
    """Generate a semiprime n = p * q with approximately `bits` total bits."""
    half = bits // 2
    lo = max(3, 2 ** (half - 1))
    hi = 2**half - 1
    while True:
        p_candidate = rng.randint(lo, hi)
        p = nextprime(p_candidate)
        if p > hi:
            continue
        q_candidate = rng.randint(lo, hi)
        q = nextprime(q_candidate)
        if q > hi or p == q:
            continue
        n = p * q
        if n.bit_length() >= bits - 1 and n.bit_length() <= bits + 1:
            return n, min(p, q), max(p, q)


def main():
    rng = random.Random(42)
    bit_sizes = [12, 16, 20, 24]
    num_instances = 5

    print("=" * 90)
    print("SDP Relaxation vs Backtracking Comparison")
    print("=" * 90)
    print()

    # Algorithms to compare
    sdp = SDPConvolution(base=10, max_restarts=50, max_iters=50, seed=42)
    alt_proj = AlternatingProjection(
        base=10, max_restarts=200, max_iters_per_restart=50, seed=42
    )
    backtrack = DigitConvolution(base=10)

    # Check if Z3 is available
    try:
        from factoring_lab.algorithms.smt_convolution import SMTConvolution
        smt = SMTConvolution(base=10, timeout_seconds=5)
        has_smt = True
    except Exception:
        has_smt = False
        smt = None

    algorithms = [
        ("SDP Relaxation", sdp),
        ("Alternating Proj", alt_proj),
        ("Backtracking", backtrack),
    ]
    if has_smt:
        algorithms.append(("Z3/SMT", smt))

    # Results table
    results_by_bits: dict[int, list[dict]] = {b: [] for b in bit_sizes}

    for bits in bit_sizes:
        print(f"\n--- {bits}-bit semiprimes ---")
        print(
            f"{'N':>12} {'p':>8} {'q':>8} | "
            + " | ".join(f"{name:>16}" for name, _ in algorithms)
        )
        print("-" * (40 + 19 * len(algorithms)))

        for inst in range(num_instances):
            n, p, q = generate_semiprime(bits, rng)
            row = {"n": n, "p": p, "q": q, "bits": bits}

            timings = []
            for algo_name, algo in algorithms:
                t0 = time.perf_counter()
                try:
                    result = algo.factor(n)
                    elapsed = time.perf_counter() - t0
                    if result.success:
                        timings.append(f"{elapsed:.4f}s OK")
                        row[algo_name] = {
                            "success": True,
                            "time": elapsed,
                            "iters": result.iteration_count,
                        }
                    else:
                        timings.append(f"{elapsed:.4f}s FAIL")
                        row[algo_name] = {
                            "success": False,
                            "time": elapsed,
                        }
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    timings.append(f"{elapsed:.4f}s ERR")
                    row[algo_name] = {"success": False, "time": elapsed, "error": str(e)}

            print(
                f"{n:>12} {p:>8} {q:>8} | "
                + " | ".join(f"{t:>16}" for t in timings)
            )
            results_by_bits[bits].append(row)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for bits in bit_sizes:
        print(f"\n{bits}-bit semiprimes:")
        for algo_name, _ in algorithms:
            successes = sum(
                1
                for r in results_by_bits[bits]
                if r.get(algo_name, {}).get("success", False)
            )
            times = [
                r[algo_name]["time"]
                for r in results_by_bits[bits]
                if r.get(algo_name, {}).get("success", False)
            ]
            avg_time = sum(times) / len(times) if times else float("nan")
            print(
                f"  {algo_name:>20}: {successes}/{num_instances} succeeded, "
                f"avg time={avg_time:.4f}s"
            )

    # Integrality gap analysis
    print("\n" + "=" * 90)
    print("INTEGRALITY GAP ANALYSIS")
    print("=" * 90)

    analyzer = SDPAnalysis(base=10)
    for bits in [12, 16, 20]:
        print(f"\n{bits}-bit semiprimes:")
        for row in results_by_bits[bits][:3]:  # first 3
            p, q = row["p"], row["q"]
            analysis = analyzer.analyze_integrality_gap(p, q)
            print(
                f"  {p}*{q}={p*q}: "
                f"true_trace={analysis['true_trace']:.1f}, "
                f"rank1_ratio={analysis['true_rank1_ratio']:.4f}, "
                f"avg_gap={analysis.get('avg_relaxed_gap', 'N/A')}"
            )


if __name__ == "__main__":
    main()
