#!/usr/bin/env python3
"""Analyze scaling laws from base sweep data.

Reads both base_sweep.csv and extended_base_sweep.csv, finds optimal bases,
fits exponential scaling models, and compares growth rates with vs without
digit constraints. No matplotlib required -- prints numerical results only.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from collections import defaultdict


def read_csv(path: Path) -> list[dict]:
    """Read CSV and return list of dicts."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(row)
    return rows


def parse_rows(rows: list[dict]) -> list[dict]:
    """Normalize row fields."""
    parsed = []
    for row in rows:
        bits = int(row["bits"])
        base_str = row["base"].strip()
        runtime = float(row["runtime_seconds"])
        success = row["success"].strip() == "True"
        parsed.append({
            "bits": bits,
            "base": base_str,
            "runtime": runtime,
            "success": success,
        })
    return parsed


def fit_exponential(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit y = a * 2^(b * x) via linear regression on log2(y).

    Returns (a, b, r_squared).
    """
    # log2(y) = log2(a) + b * x
    n = len(xs)
    if n < 2:
        return (0.0, 0.0, 0.0)

    log_ys = [math.log2(y) for y in ys]

    x_mean = sum(xs) / n
    ly_mean = sum(log_ys) / n

    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_xy = sum((x - x_mean) * (ly - ly_mean) for x, ly in zip(xs, log_ys))
    ss_yy = sum((ly - ly_mean) ** 2 for ly in log_ys)

    if ss_xx == 0:
        return (0.0, 0.0, 0.0)

    b = ss_xy / ss_xx
    log2_a = ly_mean - b * x_mean
    a = 2 ** log2_a

    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0

    return (a, b, r_squared)


def main() -> None:
    reports_dir = Path(__file__).resolve().parent.parent / "reports"

    # Load both datasets
    base_sweep_path = reports_dir / "base_sweep.csv"
    extended_path = reports_dir / "extended_base_sweep.csv"

    all_rows: list[dict] = []

    if base_sweep_path.exists():
        raw = read_csv(base_sweep_path)
        all_rows.extend(parse_rows(raw))
        print(f"Loaded {len(raw)} rows from base_sweep.csv")

    if extended_path.exists():
        raw = read_csv(extended_path)
        all_rows.extend(parse_rows(raw))
        print(f"Loaded {len(raw)} rows from extended_base_sweep.csv")

    print(f"Total rows: {len(all_rows)}\n")

    # ── 1. Find optimal base for each bit size ──────────────────────────
    # Group by (bits, base), take best (fastest successful) runtime
    best_runtime: dict[tuple[int, str], float] = {}
    for row in all_rows:
        if not row["success"]:
            continue
        key = (row["bits"], row["base"])
        if key not in best_runtime or row["runtime"] < best_runtime[key]:
            best_runtime[key] = row["runtime"]

    # Get all bit sizes
    all_bits = sorted(set(k[0] for k in best_runtime.keys()))

    print("=" * 70)
    print("OPTIMAL BASE PER BIT SIZE")
    print("=" * 70)
    print(f"{'bits':>6s}  {'optimal_base':>12s}  {'runtime_s':>10s}  {'raw_runtime':>12s}  {'speedup':>8s}")
    print("-" * 60)

    optimal_bases: dict[int, tuple[str, float]] = {}
    raw_runtimes: dict[int, float] = {}

    for bits in all_bits:
        # Find best base (excluding raw)
        best_base = None
        best_time = float("inf")
        for (b, base_str), t in best_runtime.items():
            if b == bits and base_str != "raw":
                if t < best_time:
                    best_time = t
                    best_base = base_str

        raw_t = best_runtime.get((bits, "raw"), float("nan"))
        raw_runtimes[bits] = raw_t

        if best_base is not None:
            optimal_bases[bits] = (best_base, best_time)
            speedup = raw_t / best_time if best_time > 0 else float("inf")
            print(f"{bits:>6d}  {best_base:>12s}  {best_time:>10.4f}  {raw_t:>12.4f}  {speedup:>8.2f}x")
        else:
            print(f"{bits:>6d}  {'N/A':>12s}")

    # ── 2. Test hypothesis: optimal_base ~ 2^(bits/4) ──────────────────
    print()
    print("=" * 70)
    print("HYPOTHESIS: optimal_base ~ 2^(bits/4)")
    print("=" * 70)
    print(f"{'bits':>6s}  {'optimal':>12s}  {'2^(bits/4)':>12s}  {'2^(bits/3)':>12s}  {'2^(bits/2)':>12s}")
    print("-" * 60)

    for bits in all_bits:
        if bits in optimal_bases:
            opt_str = optimal_bases[bits][0]
            pred_quarter = 2 ** (bits / 4)
            pred_third = 2 ** (bits / 3)
            pred_half = 2 ** (bits / 2)
            print(f"{bits:>6d}  {opt_str:>12s}  {pred_quarter:>12.1f}  {pred_third:>12.1f}  {pred_half:>12.1f}")

    # ── 3. Fit exponential scaling: time = a * 2^(b * bits) ────────────
    print()
    print("=" * 70)
    print("EXPONENTIAL SCALING FIT: time = a * 2^(b * bits)")
    print("=" * 70)

    # For each base, gather (bits, runtime) pairs from successful runs
    base_data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in all_rows:
        if not row["success"]:
            continue
        base_data[row["base"]].append((row["bits"], row["runtime"]))

    # Deduplicate: for each (base, bits), keep the best runtime
    for base_str in base_data:
        by_bits: dict[int, float] = {}
        for bits, t in base_data[base_str]:
            if bits not in by_bits or t < by_bits[bits]:
                by_bits[bits] = t
        base_data[base_str] = sorted(by_bits.items())

    print(f"\n{'base':>8s}  {'a':>12s}  {'b':>10s}  {'R^2':>8s}  {'n_pts':>6s}  {'doubling_bits':>14s}")
    print("-" * 70)

    fit_results: dict[str, tuple[float, float, float]] = {}

    # Sort bases: raw first, then numeric
    base_keys = sorted(base_data.keys(), key=lambda x: (-1, x) if x == "raw" else (0, int(x)))

    for base_str in base_keys:
        data = base_data[base_str]
        if len(data) < 3:
            continue
        xs = [float(b) for b, _ in data]
        ys = [t for _, t in data]

        a, b, r2 = fit_exponential(xs, ys)
        fit_results[base_str] = (a, b, r2)

        # Doubling: every 1/b bits, runtime doubles
        doubling = 1.0 / b if b > 0 else float("inf")
        print(f"{base_str:>8s}  {a:>12.6f}  {b:>10.4f}  {r2:>8.4f}  {len(data):>6d}  {doubling:>14.2f}")

    # ── 4. Compare scaling WITH vs WITHOUT digit constraints ───────────
    print()
    print("=" * 70)
    print("SCALING COMPARISON: raw vs best digit-constrained bases")
    print("=" * 70)

    if "raw" in fit_results:
        a_raw, b_raw, r2_raw = fit_results["raw"]
        print(f"Raw (no constraints):  b = {b_raw:.4f}  (runtime doubles every {1/b_raw:.1f} bits)")
        print()

        # Find bases with best (lowest) scaling exponent among those with good R^2
        good_fits = [(bs, a, b, r2) for bs, (a, b, r2) in fit_results.items()
                     if bs != "raw" and r2 > 0.7 and b > 0]
        good_fits.sort(key=lambda x: x[2])  # sort by b (scaling exponent)

        if good_fits:
            print("Top 5 bases by scaling exponent (lower = better scaling):")
            print(f"{'base':>8s}  {'b_exponent':>10s}  {'R^2':>8s}  {'ratio_vs_raw':>14s}")
            print("-" * 50)
            for bs, a, b, r2 in good_fits[:5]:
                ratio = b / b_raw if b_raw > 0 else float("inf")
                print(f"{bs:>8s}  {b:>10.4f}  {r2:>8.4f}  {ratio:>14.3f}")

            best_bs, best_a, best_b, best_r2 = good_fits[0]
            print(f"\nBest scaling base: {best_bs}")
            print(f"  Exponent ratio (best/raw): {best_b/b_raw:.3f}")

            if best_b / b_raw < 0.5:
                print("  => Digit constraints CHANGE the scaling class (more than 2x reduction)")
            elif best_b / b_raw < 0.9:
                print("  => Digit constraints REDUCE the scaling exponent (meaningful improvement)")
            else:
                print("  => Digit constraints only change the CONSTANT FACTOR (same scaling class)")

    # ── 5. Optimal base scaling with bits ──────────────────────────────
    print()
    print("=" * 70)
    print("OPTIMAL BASE AS FUNCTION OF BIT SIZE")
    print("=" * 70)

    # For bits where we have optimal bases that are numeric
    opt_numeric = []
    for bits in all_bits:
        if bits in optimal_bases:
            base_str, _ = optimal_bases[bits]
            try:
                base_val = int(base_str)
                opt_numeric.append((bits, base_val))
            except ValueError:
                pass

    if len(opt_numeric) >= 3:
        xs = [float(b) for b, _ in opt_numeric]
        ys_log2 = [math.log2(v) for _, v in opt_numeric]

        # Fit log2(optimal_base) = c * bits + d
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys_log2) / n
        ss_xx = sum((x - x_mean) ** 2 for x in xs)
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys_log2))
        ss_yy = sum((y - y_mean) ** 2 for y in ys_log2)

        if ss_xx > 0:
            c = ss_xy / ss_xx
            d = y_mean - c * x_mean
            r2 = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0

            print(f"Fit: log2(optimal_base) = {c:.4f} * bits + {d:.4f}")
            print(f"  => optimal_base ~ 2^({c:.4f} * bits)")
            print(f"  R^2 = {r2:.4f}")
            print(f"  (hypothesis was 2^(bits/4), i.e. coefficient = 0.2500)")
            print()
            print("Predicted vs actual:")
            print(f"{'bits':>6s}  {'actual':>8s}  {'predicted':>10s}")
            for bits, base_val in opt_numeric:
                pred = 2 ** (c * bits + d)
                print(f"{bits:>6d}  {base_val:>8d}  {pred:>10.1f}")

    # ── 6. Summary ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if "raw" in fit_results:
        _, b_raw, r2_raw = fit_results["raw"]
        print(f"1. Raw scaling exponent b = {b_raw:.4f} (R^2 = {r2_raw:.4f})")
        print(f"   Runtime doubles every {1/b_raw:.1f} bits")

    good_fits_all = [(bs, a, b, r2) for bs, (a, b, r2) in fit_results.items()
                     if bs != "raw" and r2 > 0.5 and b > 0]
    if good_fits_all:
        good_fits_all.sort(key=lambda x: x[2])
        best_bs, best_a, best_b, best_r2 = good_fits_all[0]
        print(f"2. Best digit-constrained scaling: base {best_bs}, b = {best_b:.4f} (R^2 = {best_r2:.4f})")
        print(f"   Runtime doubles every {1/best_b:.1f} bits")

        if "raw" in fit_results:
            ratio = best_b / b_raw
            print(f"3. Exponent ratio: {ratio:.3f}")
            print(f"   Digit constraints {'change' if ratio < 0.5 else 'reduce' if ratio < 0.9 else 'do not change'} the scaling class")


if __name__ == "__main__":
    main()
