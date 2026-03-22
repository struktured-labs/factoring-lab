#!/usr/bin/env python3
"""SVD -> Coppersmith hybrid factoring experiment.

The SVD of the carry-constrained least-squares solution Z* gives estimates
of factor digits with varying confidence. Coppersmith's method can recover
a full factor given ~50% of its bits. This script tests whether SVD-informed
digit selection outperforms random digit selection for the Coppersmith step.

Critical insight: the raw SVD singular vectors have unit norm and small magnitude.
We must AFFINE RESCALE them into the valid digit range [0, base-1] before rounding.

Strategy:
1. Build carry system, solve least-squares, SVD -> rank-1 approximation
2. Affine-rescale singular vectors into digit range
3. Per-digit confidence = distance to nearest integer (after rescaling)
4. Build p_low for Coppersmith from contiguous low-order SVD-estimated digits
5. Also try enumerating over uncertain low-order positions
6. Compare SVD-informed selection vs random selection vs true-value oracle

Key question: does the spectral signal provide value beyond random guessing?
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import product as cart_product

import numpy as np

# Ensure the package is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from factoring_lab.algorithms.hybrid_coppersmith import (
    coppersmith_lattice_factor_base,
    _lattice_recover_factor,
)
from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    to_digits,
    from_digits,
)
from factoring_lab.generators.semiprimes import balanced_semiprime


# ---------------------------------------------------------------------------
# SVD estimation with proper rescaling
# ---------------------------------------------------------------------------

def _build_carry_system(n: int, base: int, dx: int, dy: int):
    """Build the linear system A @ v = b for carry constraints."""
    c = to_digits(n, base)
    d = len(c)

    z_vars = []
    z_idx = {}
    for i in range(dx):
        for j in range(dy):
            if i + j < d:
                z_idx[(i, j)] = len(z_vars)
                z_vars.append((i, j))

    num_z = len(z_vars)
    num_t = d
    num_vars = num_z + num_t

    A = np.zeros((d, num_vars))
    b_vec = np.zeros(d)

    for k in range(d):
        b_vec[k] = c[k]
        for i in range(min(k + 1, dx)):
            j = k - i
            if 0 <= j < dy and (i, j) in z_idx:
                A[k, z_idx[(i, j)]] = 1.0
        if k > 0:
            A[k, num_z + k - 1] = 1.0
        A[k, num_z + k] = -base

    return A, b_vec, z_vars, z_idx, num_z, num_t


def _affine_rescale(v: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Affine-rescale vector v to range [lo, hi]."""
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-15:
        return np.full_like(v, (lo + hi) / 2.0)
    return lo + (hi - lo) * (v - vmin) / (vmax - vmin)


def svd_estimate_digits(n: int, base: int, x_true: list[int], y_true: list[int]):
    """Compute SVD-based estimates of factor digits with proper rescaling.

    Returns:
        x_est: array of estimated x-digits (float, in [0, base-1])
        y_est: array of estimated y-digits (float, in [0, base-1])
        x_confidence: per-digit confidence (lower = more certain)
        y_confidence: per-digit confidence
        x_corr: absolute correlation with true x
        y_corr: absolute correlation with true y
        dx, dy: digit counts
        x_accuracy: fraction of correctly rounded digits
        y_accuracy: fraction of correctly rounded digits
    """
    d, dx, dy = _compute_digit_sizes(n, base)
    A, b_vec, z_vars, z_idx, num_z, num_t = _build_carry_system(n, base, dx, dy)

    # Minimum-norm least-squares solution
    v_star, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

    # Extract Z matrix
    Z_star = np.zeros((dx, dy))
    for (i, j), idx in z_idx.items():
        Z_star[i, j] = v_star[idx]

    # SVD
    U, S, Vt = np.linalg.svd(Z_star, full_matrices=False)

    # Raw singular vectors (unit norm)
    u1 = U[:, 0]
    v1 = Vt[0, :]

    # Pad true digits
    x_true_arr = np.array(x_true + [0] * (dx - len(x_true)), dtype=float)
    y_true_arr = np.array(y_true + [0] * (dy - len(y_true)), dtype=float)

    def _corr(a, b):
        n = min(len(a), len(b))
        if np.std(a[:n]) < 1e-12 or np.std(b[:n]) < 1e-12:
            return 0.0
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    # Determine optimal sign for each factor
    # Try both (u1, v1) and (-u1, -v1), and also swapping x<->y
    best_x_corr = 0.0
    best_y_corr = 0.0
    best_x_est = None
    best_y_est = None

    for x_sign in [1, -1]:
        for y_sign in [1, -1]:
            # Rescale into [0, base-1]
            x_scaled = _affine_rescale(x_sign * u1, 0, base - 1)
            y_scaled = _affine_rescale(y_sign * v1, 0, base - 1)

            cx = abs(_corr(x_scaled, x_true_arr))
            cy = abs(_corr(y_scaled, y_true_arr))

            if cx + cy > best_x_corr + best_y_corr:
                best_x_corr = cx
                best_y_corr = cy
                best_x_est = x_scaled
                best_y_est = y_scaled

    # Also try swapping x <-> y (SVD doesn't know which is p vs q)
    for x_sign in [1, -1]:
        for y_sign in [1, -1]:
            x_scaled = _affine_rescale(x_sign * v1, 0, base - 1)
            y_scaled = _affine_rescale(y_sign * u1, 0, base - 1)

            cx = abs(_corr(x_scaled, x_true_arr))
            cy = abs(_corr(y_scaled, y_true_arr))

            if cx + cy > best_x_corr + best_y_corr:
                best_x_corr = cx
                best_y_corr = cy
                best_x_est = x_scaled
                best_y_est = y_scaled

    x_est = best_x_est
    y_est = best_y_est

    # Per-digit confidence: distance to nearest valid integer
    # Lower = closer to integer = more confident
    def digit_confidence(est):
        conf = np.zeros(len(est))
        for i, v in enumerate(est):
            vc = max(0.0, min(float(base - 1), v))
            nearest = round(vc)
            conf[i] = abs(vc - nearest)
        return conf

    x_conf = digit_confidence(x_est)
    y_conf = digit_confidence(y_est)

    # Digit accuracy
    x_rounded = np.array([max(0, min(base - 1, int(round(v)))) for v in x_est])
    y_rounded = np.array([max(0, min(base - 1, int(round(v)))) for v in y_est])
    x_acc = np.mean(x_rounded[:len(x_true)] == x_true_arr[:len(x_true)])
    y_acc = np.mean(y_rounded[:len(y_true)] == y_true_arr[:len(y_true)])

    return (x_est, y_est, x_conf, y_conf,
            best_x_corr, best_y_corr, dx, dy,
            float(x_acc), float(y_acc),
            S[0], S[1] if len(S) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Coppersmith recovery strategies
# ---------------------------------------------------------------------------

def _try_coppersmith(n: int, p_low: int, base: int, num_known: int) -> int | None:
    """Try Coppersmith recovery, return factor or None."""
    if p_low <= 1:
        return None
    result = coppersmith_lattice_factor_base(n, p_low, base, num_known)
    if result is not None and 1 < result < n:
        return int(result)
    return None


def attempt_svd_coppersmith(
    n: int,
    digit_est: np.ndarray,
    confidence: np.ndarray,
    base: int,
    num_true_digits: int,
    leak_fraction: float,
    timeout_s: float = 10.0,
    strategy: str = "contiguous",
) -> tuple[int | None, int]:
    """Attempt Coppersmith recovery using SVD-estimated digits.

    Strategies:
        "contiguous": round the lowest k digits, feed to Coppersmith directly
        "confident": fix most-confident digits, enumerate uncertain ones in prefix
        "enumerate_prefix": try ALL base^k combinations for uncertain low digits

    Returns (factor_or_None, num_candidates_tried)
    """
    num_known = max(1, int(num_true_digits * leak_fraction))
    num_known = min(num_known, len(digit_est))
    t0 = time.perf_counter()
    tried = 0

    if strategy == "contiguous":
        # Simply round the lowest num_known digits
        digits = [max(0, min(base - 1, int(round(digit_est[i]))))
                  for i in range(num_known)]
        p_low = from_digits(digits, base)
        tried = 1
        result = _try_coppersmith(n, p_low, base, num_known)
        return result, tried

    elif strategy == "confident":
        # Sort by confidence, fix confident digits, enumerate uncertain in prefix
        sorted_pos = sorted(range(min(num_known, len(digit_est))),
                            key=lambda i: confidence[i])
        confident_set = set(sorted_pos[:max(1, num_known // 2)])

        # Build template: confident digits get SVD value, others unknown
        prefix_len = num_known
        uncertain = [i for i in range(prefix_len) if i not in confident_set]

        # Cap enumeration
        max_uncertain = min(len(uncertain), 4 if base <= 10 else 2)
        uncertain = uncertain[:max_uncertain]

        template = [max(0, min(base - 1, int(round(digit_est[i]))))
                    for i in range(prefix_len)]

        if not uncertain:
            p_low = from_digits(template, base)
            tried = 1
            return _try_coppersmith(n, p_low, base, prefix_len), tried

        for combo in cart_product(*[range(base) for _ in uncertain]):
            if time.perf_counter() - t0 > timeout_s:
                break
            digits = list(template)
            for idx, pos in enumerate(uncertain):
                digits[pos] = combo[idx]
            p_low = from_digits(digits, base)
            tried += 1
            result = _try_coppersmith(n, p_low, base, prefix_len)
            if result is not None:
                return result, tried

        return None, tried

    elif strategy == "enumerate_prefix":
        # Enumerate all combinations for uncertain positions in prefix
        # Only feasible for small bases or few uncertain positions
        prefix_len = num_known

        # Determine which positions have high confidence (close to integer)
        threshold = 0.2  # positions with confidence < this are "fixed"
        fixed = {}
        uncertain = []
        for i in range(prefix_len):
            if i < len(confidence) and confidence[i] < threshold:
                fixed[i] = max(0, min(base - 1, int(round(digit_est[i]))))
            else:
                uncertain.append(i)

        # Cap enumeration
        max_uncertain = min(len(uncertain), 5 if base == 2 else 3)
        if len(uncertain) > max_uncertain:
            # Keep only the least confident positions uncertain
            uncertain_sorted = sorted(uncertain,
                                      key=lambda i: confidence[i]
                                      if i < len(confidence) else 0.5,
                                      reverse=True)
            for pos in uncertain_sorted[max_uncertain:]:
                fixed[pos] = max(0, min(base - 1, int(round(
                    digit_est[pos] if pos < len(digit_est) else 0))))
            uncertain = uncertain_sorted[:max_uncertain]

        for combo in cart_product(*[range(base) for _ in uncertain]):
            if time.perf_counter() - t0 > timeout_s:
                break
            digits = [0] * prefix_len
            for pos, val in fixed.items():
                digits[pos] = val
            for idx, pos in enumerate(uncertain):
                digits[pos] = combo[idx]
            p_low = from_digits(digits, base)
            tried += 1
            result = _try_coppersmith(n, p_low, base, prefix_len)
            if result is not None:
                return result, tried

        return None, tried

    return None, 0


def attempt_random_true_coppersmith(
    n: int,
    true_digits: list[int],
    base: int,
    leak_fraction: float,
    num_trials: int = 10,
    seed: int = 42,
) -> tuple[int | None, int]:
    """Attempt Coppersmith with randomly selected TRUE digit values.

    This is the oracle baseline: random selection + perfect values.
    Uses contiguous low-order prefix only.
    """
    num_known = max(1, int(len(true_digits) * leak_fraction))
    num_known = min(num_known, len(true_digits))
    tried = 0

    # With true contiguous digits, just try the exact low-order prefix
    for k in range(1, num_known + 1):
        p_low = from_digits(true_digits[:k], base)
        tried += 1
        result = _try_coppersmith(n, p_low, base, k)
        if result is not None:
            return result, tried

    # Also try random subsets with contiguous enforcement
    for trial in range(num_trials):
        rng = random.Random(seed + trial)
        positions = list(range(len(true_digits)))
        rng.shuffle(positions)
        selected = set(positions[:num_known])

        # Find contiguous prefix
        prefix_len = 0
        for i in range(len(true_digits)):
            if i in selected:
                prefix_len = i + 1
            else:
                break
        if prefix_len < 1:
            continue

        p_low = from_digits(true_digits[:prefix_len], base)
        tried += 1
        result = _try_coppersmith(n, p_low, base, prefix_len)
        if result is not None:
            return result, tried

    return None, tried


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SVDCoppersmithResult:
    """Result of one SVD-Coppersmith hybrid trial."""
    n: int
    bits: int
    base: int
    p_true: int
    q_true: int
    method: str
    leak_fraction: float
    num_digits_total: int
    num_digits_used: int
    p_low_tried: int
    coppersmith_success: bool
    factor_found: int | None
    x_corr: float
    y_corr: float
    x_accuracy: float
    y_accuracy: float
    sigma_ratio: float
    runtime_s: float


# ---------------------------------------------------------------------------
# Core experiment: one semiprime
# ---------------------------------------------------------------------------

def svd_coppersmith_recovery(
    n: int,
    p_true: int,
    q_true: int,
    base: int,
    leak_fractions: list[float] | None = None,
    timeout_s: float = 30.0,
) -> list[SVDCoppersmithResult]:
    """Run the SVD -> Coppersmith hybrid on a single semiprime."""
    if leak_fractions is None:
        leak_fractions = [0.25, 0.50, 0.75]

    results = []
    bits = n.bit_length()

    x_true = to_digits(p_true, base)
    y_true = to_digits(q_true, base)

    # SVD estimation with proper rescaling
    t0 = time.perf_counter()
    (x_est, y_est, x_conf, y_conf,
     x_corr, y_corr, dx, dy,
     x_acc, y_acc, s1, s2) = svd_estimate_digits(n, base, x_true, y_true)
    svd_time = time.perf_counter() - t0

    sigma_ratio = s1 / s2 if s2 > 1e-15 else float("inf")

    for frac in leak_fractions:
        # ---- Method 1: SVD contiguous (direct round lowest k digits) ----
        t0 = time.perf_counter()
        found_x, tried_x = attempt_svd_coppersmith(
            n, x_est, x_conf, base, len(x_true), frac,
            timeout_s=timeout_s, strategy="contiguous"
        )
        found_y, tried_y = (None, 0)
        if found_x is None:
            found_y, tried_y = attempt_svd_coppersmith(
                n, y_est, y_conf, base, len(y_true), frac,
                timeout_s=timeout_s, strategy="contiguous"
            )
        found = found_x or found_y
        elapsed = time.perf_counter() - t0
        results.append(SVDCoppersmithResult(
            n=n, bits=bits, base=base, p_true=p_true, q_true=q_true,
            method="svd_contiguous",
            leak_fraction=frac,
            num_digits_total=len(x_true),
            num_digits_used=max(1, int(len(x_true) * frac)),
            p_low_tried=tried_x + tried_y,
            coppersmith_success=found is not None,
            factor_found=found,
            x_corr=x_corr, y_corr=y_corr,
            x_accuracy=x_acc, y_accuracy=y_acc,
            sigma_ratio=sigma_ratio,
            runtime_s=round(elapsed + svd_time, 4),
        ))

        # ---- Method 2: SVD confident (enumerate uncertain low positions) ----
        t0 = time.perf_counter()
        found_x, tried_x = attempt_svd_coppersmith(
            n, x_est, x_conf, base, len(x_true), frac,
            timeout_s=timeout_s, strategy="confident"
        )
        found_y, tried_y = (None, 0)
        if found_x is None:
            found_y, tried_y = attempt_svd_coppersmith(
                n, y_est, y_conf, base, len(y_true), frac,
                timeout_s=timeout_s, strategy="confident"
            )
        found = found_x or found_y
        elapsed = time.perf_counter() - t0
        results.append(SVDCoppersmithResult(
            n=n, bits=bits, base=base, p_true=p_true, q_true=q_true,
            method="svd_confident",
            leak_fraction=frac,
            num_digits_total=len(x_true),
            num_digits_used=max(1, int(len(x_true) * frac)),
            p_low_tried=tried_x + tried_y,
            coppersmith_success=found is not None,
            factor_found=found,
            x_corr=x_corr, y_corr=y_corr,
            x_accuracy=x_acc, y_accuracy=y_acc,
            sigma_ratio=sigma_ratio,
            runtime_s=round(elapsed + svd_time, 4),
        ))

        # ---- Method 3: SVD enumerate prefix ----
        t0 = time.perf_counter()
        found_x, tried_x = attempt_svd_coppersmith(
            n, x_est, x_conf, base, len(x_true), frac,
            timeout_s=timeout_s, strategy="enumerate_prefix"
        )
        found_y, tried_y = (None, 0)
        if found_x is None:
            found_y, tried_y = attempt_svd_coppersmith(
                n, y_est, y_conf, base, len(y_true), frac,
                timeout_s=timeout_s, strategy="enumerate_prefix"
            )
        found = found_x or found_y
        elapsed = time.perf_counter() - t0
        results.append(SVDCoppersmithResult(
            n=n, bits=bits, base=base, p_true=p_true, q_true=q_true,
            method="svd_enum_prefix",
            leak_fraction=frac,
            num_digits_total=len(x_true),
            num_digits_used=max(1, int(len(x_true) * frac)),
            p_low_tried=tried_x + tried_y,
            coppersmith_success=found is not None,
            factor_found=found,
            x_corr=x_corr, y_corr=y_corr,
            x_accuracy=x_acc, y_accuracy=y_acc,
            sigma_ratio=sigma_ratio,
            runtime_s=round(elapsed + svd_time, 4),
        ))

        # ---- Method 4: random TRUE digits (oracle baseline) ----
        t0 = time.perf_counter()
        found_x, tried_x = attempt_random_true_coppersmith(
            n, x_true, base, frac
        )
        found_y, tried_y = (None, 0)
        if found_x is None:
            found_y, tried_y = attempt_random_true_coppersmith(
                n, y_true, base, frac
            )
        found = found_x or found_y
        elapsed = time.perf_counter() - t0
        results.append(SVDCoppersmithResult(
            n=n, bits=bits, base=base, p_true=p_true, q_true=q_true,
            method="oracle_true",
            leak_fraction=frac,
            num_digits_total=len(x_true),
            num_digits_used=max(1, int(len(x_true) * frac)),
            p_low_tried=tried_x + tried_y,
            coppersmith_success=found is not None,
            factor_found=found,
            x_corr=1.0, y_corr=1.0,
            x_accuracy=1.0, y_accuracy=1.0,
            sigma_ratio=sigma_ratio,
            runtime_s=round(elapsed, 4),
        ))

    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

BIT_SIZES = [32, 48, 64]
BASES = [2, 10]
LEAK_FRACTIONS = [0.25, 0.50, 0.75]
NUM_SAMPLES = 5
TIMEOUT_S = 30.0


def run_experiment() -> list[SVDCoppersmithResult]:
    """Run the full SVD-Coppersmith comparison."""
    all_results: list[SVDCoppersmithResult] = []

    print("=" * 110)
    print("SVD -> COPPERSMITH HYBRID EXPERIMENT")
    print("=" * 110)
    print()
    print("Question: does SVD-informed digit selection outperform random")
    print("          for Coppersmith lattice recovery?")
    print()
    print("Methods:")
    print("  svd_contiguous  - round lowest k SVD-estimated digits, feed to Coppersmith")
    print("  svd_confident   - fix confident digits, enumerate uncertain ones in prefix")
    print("  svd_enum_prefix - enumerate uncertain positions in low-order prefix")
    print("  oracle_true     - use TRUE digit values (oracle baseline)")
    print()

    for bits in BIT_SIZES:
        print(f"\n{'='*100}")
        print(f"  BIT SIZE: {bits}")
        print(f"{'='*100}")

        for sample_idx in range(NUM_SAMPLES):
            seed = 42 + sample_idx
            spec = balanced_semiprime(bits, seed=seed)
            n, p, q = spec.n, spec.p, spec.q

            print(f"\n  Sample {sample_idx + 1}/{NUM_SAMPLES}: "
                  f"n={n} ({bits}b), p={p}, q={q}")

            for base in BASES:
                results = svd_coppersmith_recovery(
                    n, p, q, base,
                    leak_fractions=LEAK_FRACTIONS,
                    timeout_s=TIMEOUT_S,
                )

                # Print SVD diagnostics for first leak fraction only
                r0 = results[0]
                print(f"    base={base:2d}  "
                      f"|corr(x)|={r0.x_corr:.3f}  "
                      f"|corr(y)|={r0.y_corr:.3f}  "
                      f"digit_acc_x={r0.x_accuracy:.2f}  "
                      f"digit_acc_y={r0.y_accuracy:.2f}  "
                      f"sigma_ratio={r0.sigma_ratio:.1f}")

                for r in results:
                    status = "OK" if r.coppersmith_success else "--"
                    print(
                        f"      {r.method:<16s}  "
                        f"leak={r.leak_fraction:.2f}  "
                        f"digits={r.num_digits_used}/{r.num_digits_total}  "
                        f"tried={r.p_low_tried:5d}  "
                        f"{status:4s}  {r.runtime_s:7.3f}s"
                        + (f"  -> {r.factor_found}" if r.coppersmith_success else "")
                    )

                all_results.extend(results)

    return all_results


def print_comparison_table(results: list[SVDCoppersmithResult]) -> None:
    """Print the main comparison table."""
    print(f"\n\n{'='*130}")
    print("COMPARISON TABLE: SVD-informed vs Oracle digit selection for Coppersmith")
    print(f"{'='*130}")

    for base in BASES:
        print(f"\n  --- Base {base} ---")
        header = (
            f"  {'bits':>4} | {'leak':>5} | "
            f"{'svd_contig':>10} | {'svd_confid':>10} | {'svd_enum':>10} | "
            f"{'oracle':>8} | "
            f"{'corr(x)':>8} | {'corr(y)':>8} | "
            f"{'acc_x':>6} | {'acc_y':>6}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for bits in BIT_SIZES:
            for frac in LEAK_FRACTIONS:
                def get_rows(method):
                    return [r for r in results
                            if r.bits == bits and abs(r.leak_fraction - frac) < 0.01
                            and r.method == method and r.base == base]

                def success_str(rows):
                    if not rows:
                        return "N/A"
                    s = sum(1 for r in rows if r.coppersmith_success)
                    return f"{s}/{len(rows)}"

                def avg_val(rows, attr):
                    if not rows:
                        return 0.0
                    return sum(getattr(r, attr) for r in rows) / len(rows)

                contig = get_rows("svd_contiguous")
                confid = get_rows("svd_confident")
                enum_ = get_rows("svd_enum_prefix")
                oracle = get_rows("oracle_true")

                x_corr = avg_val(contig, "x_corr")
                y_corr = avg_val(contig, "y_corr")
                x_acc = avg_val(contig, "x_accuracy")
                y_acc = avg_val(contig, "y_accuracy")

                print(
                    f"  {bits:>4} | {frac:>5.2f} | "
                    f"{success_str(contig):>10s} | "
                    f"{success_str(confid):>10s} | "
                    f"{success_str(enum_):>10s} | "
                    f"{success_str(oracle):>8s} | "
                    f"{x_corr:>8.3f} | {y_corr:>8.3f} | "
                    f"{x_acc:>6.2f} | {y_acc:>6.2f}"
                )


def print_summary(results: list[SVDCoppersmithResult]) -> None:
    """Print summary statistics and key findings."""
    print(f"\n\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    for base in BASES:
        print(f"\n--- Base {base} ---")
        print("\nSuccess rates by method and bit size:")
        for bits in BIT_SIZES:
            print(f"\n  {bits}-bit semiprimes:")
            for method in ["svd_contiguous", "svd_confident", "svd_enum_prefix",
                           "oracle_true"]:
                rows = [r for r in results
                        if r.bits == bits and r.method == method and r.base == base]
                if not rows:
                    continue
                successes = sum(1 for r in rows if r.coppersmith_success)
                total = len(rows)
                avg_time = sum(r.runtime_s for r in rows) / total
                print(f"    {method:<16s}: {successes:>3}/{total} success "
                      f"({100 * successes / total:5.1f}%), avg {avg_time:.3f}s")

        print(f"\n  Average SVD digit accuracy (base {base}):")
        for bits in BIT_SIZES:
            rows = [r for r in results
                    if r.bits == bits and r.method == "svd_contiguous"
                    and r.base == base]
            if not rows:
                continue
            avg_x_acc = sum(r.x_accuracy for r in rows) / len(rows)
            avg_y_acc = sum(r.y_accuracy for r in rows) / len(rows)
            avg_x_corr = sum(r.x_corr for r in rows) / len(rows)
            avg_y_corr = sum(r.y_corr for r in rows) / len(rows)
            print(f"    {bits:>3}-bit:  "
                  f"|corr(x)|={avg_x_corr:.3f}  |corr(y)|={avg_y_corr:.3f}  "
                  f"acc_x={avg_x_acc:.2f}  acc_y={avg_y_acc:.2f}")

    print(f"\n{'='*100}")
    print("KEY FINDINGS")
    print(f"{'='*100}")

    for base in BASES:
        print(f"\n  Base {base}:")
        for bits in BIT_SIZES:
            svd_rows = [r for r in results
                        if r.bits == bits and r.method == "svd_enum_prefix"
                        and r.base == base]
            oracle_rows = [r for r in results
                           if r.bits == bits and r.method == "oracle_true"
                           and r.base == base]
            if not svd_rows or not oracle_rows:
                continue

            svd_rate = (sum(1 for r in svd_rows if r.coppersmith_success)
                        / len(svd_rows))
            oracle_rate = (sum(1 for r in oracle_rows if r.coppersmith_success)
                           / len(oracle_rows))

            if svd_rate > oracle_rate + 0.05:
                verdict = "SVD OUTPERFORMS oracle"
            elif oracle_rate > svd_rate + 0.05:
                verdict = "Oracle outperforms SVD"
            else:
                verdict = "No significant difference"

            avg_acc = sum(r.x_accuracy for r in svd_rows) / len(svd_rows)
            print(f"    {bits:>3}-bit: SVD={svd_rate:.0%} oracle={oracle_rate:.0%} "
                  f"digit_acc={avg_acc:.2f}  -- {verdict}")

    # Theoretical interpretation
    print(f"\n{'='*100}")
    print("THEORETICAL INTERPRETATION")
    print(f"{'='*100}")
    print()
    print("  The SVD of the minimum-norm carry-constrained Z* provides a continuous")
    print("  approximation to the factor digit vectors. After affine rescaling, the")
    print("  digit accuracy reveals whether the spectral signal is sufficient for")
    print("  Coppersmith to cross the ~50% leak threshold.")
    print()
    print("  If SVD digit accuracy < 50%: insufficient signal for Coppersmith.")
    print("  If SVD digit accuracy >= 50% but Coppersmith fails: the WRONG digits")
    print("    are correct (scattered, not contiguous low-order).")
    print("  If SVD digit accuracy >= 50% and Coppersmith succeeds: a viable hybrid.")


def export_csv(results: list[SVDCoppersmithResult], path: str) -> None:
    """Export results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "n", "bits", "base", "p_true", "q_true",
        "method", "leak_fraction", "num_digits_total", "num_digits_used",
        "p_low_tried", "coppersmith_success", "factor_found",
        "x_corr", "y_corr", "x_accuracy", "y_accuracy",
        "sigma_ratio", "runtime_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "n": r.n, "bits": r.bits, "base": r.base,
                "p_true": r.p_true, "q_true": r.q_true,
                "method": r.method, "leak_fraction": r.leak_fraction,
                "num_digits_total": r.num_digits_total,
                "num_digits_used": r.num_digits_used,
                "p_low_tried": r.p_low_tried,
                "coppersmith_success": r.coppersmith_success,
                "factor_found": r.factor_found,
                "x_corr": round(r.x_corr, 4),
                "y_corr": round(r.y_corr, 4),
                "x_accuracy": round(r.x_accuracy, 4),
                "y_accuracy": round(r.y_accuracy, 4),
                "sigma_ratio": round(r.sigma_ratio, 2),
                "runtime_s": r.runtime_s,
            })
    print(f"\nResults exported to {path}")


def main() -> None:
    results = run_experiment()
    print_comparison_table(results)
    print_summary(results)

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "reports", "svd_coppersmith_hybrid.csv"
    )
    export_csv(results, csv_path)


if __name__ == "__main__":
    main()
