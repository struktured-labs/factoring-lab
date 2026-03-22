"""Conditional correlation experiment: does SVD signal survive digit conditioning?

CRITICAL QUESTION: When we fix known digits of x, does the SVD correlation
for the REMAINING unknown digits increase? This determines whether spectral
methods can be bootstrapped:

  - If correlation INCREASES with more fixed digits → spectral methods can be
    iteratively improved (fix easy digits, re-estimate, fix more).
  - If correlation stays FLAT or DROPS → the signal is non-local and cannot
    be incrementally exploited. Spectral direction is dead.

APPROACH (iterative rank-1 projection with partial knowledge):
  1. Compute Z_ls = minimum-norm least-squares solution to carry system.
  2. SVD → x_est, y_est (baseline, 0% fixed).
  3. Fix the first k digits of x to their TRUE values.
  4. For each known x_i, replace row i of Z with x_i * y_est (current y estimate).
  5. Re-SVD the modified Z → improved y_est.
  6. Use improved y_est to re-estimate unknown x digits:
       x_i_est = Z_ls[i, :] @ y_est / ||y_est||^2
  7. Measure correlation of refined estimates (on UNKNOWN digits only) with truth.
  8. Also measure: how many unknown digits does naive rounding get correct?
"""

import sys
from dataclasses import dataclass
from math import log2

import numpy as np

from factoring_lab.analysis.lattice_counting import (
    _compute_digit_sizes,
    from_digits,
    to_digits,
)


def _build_carry_system(n: int, base: int, dx: int, dy: int):
    """Build the linear system A @ z_flat = b for carry constraints.

    Variables: z_{ij} for i < dx, j < dy, i+j < d
    Plus carry variables t_k for k = 0..d-1

    Constraints: sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k
    """
    c = to_digits(n, base)
    d = len(c)

    # Map (i,j) pairs to variable indices
    z_vars = []
    z_idx = {}
    for i in range(dx):
        for j in range(dy):
            if i + j < d:
                z_idx[(i, j)] = len(z_vars)
                z_vars.append((i, j))

    num_z = len(z_vars)
    num_t = d  # carry variables t_0..t_{d-1}
    num_vars = num_z + num_t

    # Build constraint matrix A and rhs b
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


def _corr(a, b):
    """Pearson correlation, handling degenerate cases."""
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class ConditionalResult:
    """Result for one (n, base, fix_fraction) configuration."""
    n: int
    base: int
    fix_fraction: float
    num_fixed: int
    num_unknown: int
    # Correlation on unknown x-digits
    corr_x_unknown: float
    # Correlation on y-digits (always all unknown)
    corr_y: float
    # Fraction of unknown x-digits correctly rounded
    x_rounding_accuracy: float
    # Fraction of y-digits correctly rounded
    y_rounding_accuracy: float
    # Whether the correlation is degenerate (truth has zero variance)
    x_corr_degenerate: bool = False


def conditional_svd_experiment(
    n: int, p: int, q: int, base: int, fix_fraction: float
) -> ConditionalResult:
    """Run conditional SVD recovery with a given fraction of x-digits fixed.

    Steps:
      1. Solve minimum-norm least-squares for carry system → Z_ls (dx x dy).
      2. SVD of Z_ls → baseline x_est, y_est.
      3. Fix the first k = floor(fix_fraction * dx) digits of x to true values.
      4. For known rows i, set Z_mod[i, :] = x_true[i] * y_est.
      5. Re-SVD Z_mod → improved y_est'.
      6. For unknown rows i, estimate x_i = Z_ls[i, :] @ y_est' / ||y_est'||^2.
      7. Measure correlation and rounding accuracy on unknown x and all y.
    """
    c = to_digits(n, base)
    d = len(c)
    _, dx, dy = _compute_digit_sizes(n, base)

    # True factor digits
    x_true = to_digits(p, base)
    y_true = to_digits(q, base)
    while len(x_true) < dx:
        x_true.append(0)
    while len(y_true) < dy:
        y_true.append(0)
    x_true = np.array(x_true, dtype=float)
    y_true = np.array(y_true, dtype=float)

    # Build carry system and solve least-squares
    A, b_vec, z_vars, z_idx, num_z, num_t = _build_carry_system(n, base, dx, dy)
    v_star, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

    # Extract Z_ls matrix
    Z_ls = np.zeros((dx, dy))
    for (i, j), idx in z_idx.items():
        Z_ls[i, j] = v_star[idx]

    # Step 2: Baseline SVD
    U, S, Vt = np.linalg.svd(Z_ls, full_matrices=False)
    s1 = S[0] if len(S) > 0 else 1.0
    scale = np.sqrt(s1) if s1 > 0 else 1.0
    y_est = scale * Vt[0, :]

    # Step 3: Determine which digits to fix
    num_fixed = int(fix_fraction * dx)
    known_indices = list(range(num_fixed))
    unknown_indices = list(range(num_fixed, dx))

    if num_fixed > 0 and np.linalg.norm(y_est) > 1e-12:
        # Step 4: Replace known rows of Z with x_true[i] * y_est
        Z_mod = Z_ls.copy()
        for i in known_indices:
            Z_mod[i, :] = x_true[i] * y_est

        # Step 5: Re-SVD of modified Z
        U2, S2, Vt2 = np.linalg.svd(Z_mod, full_matrices=False)
        s1_new = S2[0] if len(S2) > 0 else 1.0
        scale_new = np.sqrt(s1_new) if s1_new > 0 else 1.0
        y_est_improved = scale_new * Vt2[0, :]

        # Ensure sign consistency: y_est_improved should correlate positively
        # with the columns implied by the known rows.
        # Use the known constraint: for known x_i, Z[i,:] should ~ x_i * y.
        # Compare sign of y_est_improved with the known-row average.
        if num_fixed > 0:
            # Average implied y from known rows (weighted by x_true)
            nonzero_known = [i for i in known_indices if abs(x_true[i]) > 1e-12]
            if nonzero_known:
                implied_y = np.mean(
                    [Z_ls[i, :] / x_true[i] for i in nonzero_known], axis=0
                )
                if _corr(y_est_improved, implied_y) < 0:
                    y_est_improved = -y_est_improved
    else:
        y_est_improved = y_est.copy()

    # Step 6: Estimate unknown x-digits using improved y
    y_norm_sq = np.dot(y_est_improved, y_est_improved)
    if y_norm_sq < 1e-12:
        x_est_unknown = np.zeros(len(unknown_indices))
    else:
        x_est_unknown = np.array([
            np.dot(Z_ls[i, :], y_est_improved) / y_norm_sq
            for i in unknown_indices
        ])

    # Also get full y correlation from improved estimate
    # For y, use the rank-1 approximation: y ~ Z_ls^T @ x_est_full / ||x_est_full||^2
    # But more directly, y_est_improved is our y estimate from the re-SVD.

    # Step 7: Measure correlation and rounding accuracy
    # Correlation on unknown x-digits
    x_true_unknown = x_true[unknown_indices]
    x_degenerate = len(unknown_indices) < 2 or np.std(x_true_unknown) < 1e-12
    corr_x_unk = _corr(x_est_unknown, x_true_unknown)

    # Correlation on y
    corr_y = _corr(y_est_improved, y_true)

    # Rounding accuracy on unknown x-digits
    if len(unknown_indices) > 0:
        x_rounded = np.array([
            max(0, min(base - 1, int(round(v)))) for v in x_est_unknown
        ])
        x_correct = np.sum(x_rounded == x_true_unknown.astype(int))
        x_acc = float(x_correct) / len(unknown_indices)
    else:
        x_acc = 1.0  # all digits are known

    # Rounding accuracy on y
    y_rounded = np.array([
        max(0, min(base - 1, int(round(v)))) for v in y_est_improved
    ])
    y_correct = np.sum(y_rounded == y_true.astype(int))
    y_acc = float(y_correct) / dy

    return ConditionalResult(
        n=n,
        base=base,
        fix_fraction=fix_fraction,
        num_fixed=num_fixed,
        num_unknown=len(unknown_indices),
        corr_x_unknown=corr_x_unk,
        corr_y=corr_y,
        x_rounding_accuracy=x_acc,
        y_rounding_accuracy=y_acc,
        x_corr_degenerate=x_degenerate,
    )


def run_experiment():
    """Run the full conditional correlation experiment."""
    cases = [
        (15, 3, 5),
        (77, 7, 11),
        (323, 17, 19),
        (1073, 29, 37),
        (5183, 71, 73),
        (10403, 101, 103),
    ]
    bases = [2, 10]
    fix_fractions = [0.0, 0.25, 0.50, 0.75]

    print("=" * 100)
    print("CONDITIONAL CORRELATION EXPERIMENT")
    print("Does SVD correlation survive conditioning on fixed digits?")
    print("=" * 100)
    print()
    print("For each semiprime, we fix the first k digits of x to their true values,")
    print("then measure SVD correlation for the REMAINING unknown digits.")
    print()
    print("KEY: If |corr_x_unknown| INCREASES with more fixed digits → bootstrapping is viable.")
    print("     If it stays FLAT or DROPS → spectral signal is non-local, cannot exploit incrementally.")
    print()

    # ---- PART 1: CORRELATION TABLE ----
    print("-" * 100)
    print("PART 1: SVD Correlation on Unknown x-Digits (|corr|)")
    print("  (* = <2 unknown digits, # = truth has zero variance — both are degenerate)")
    print("-" * 100)
    header = f"{'n':>8} {'base':>5} {'dx':>4}"
    for frac in fix_fractions:
        header += f" {'%d%%fixed' % int(frac * 100):>12}"
    print(header)
    print("-" * 100)

    all_results: dict[tuple[int, int], list[ConditionalResult]] = {}

    for n, p, q in cases:
        for base in bases:
            results = []
            for frac in fix_fractions:
                r = conditional_svd_experiment(n, p, q, base, frac)
                results.append(r)
            all_results[(n, base)] = results

            _, dx, _ = _compute_digit_sizes(n, base)
            row = f"{n:>8} {base:>5} {dx:>4}"
            for r in results:
                val = abs(r.corr_x_unknown)
                if r.num_unknown < 2:
                    marker = "*"
                elif r.x_corr_degenerate:
                    marker = "#"
                else:
                    marker = " "
                row += f" {val:>11.4f}{marker}"
            print(row)

    # ---- PART 2: Y-CORRELATION TABLE ----
    print()
    print("-" * 100)
    print("PART 2: SVD Correlation on y-Digits (|corr_y|) — does y-estimate improve?")
    print("-" * 100)
    header = f"{'n':>8} {'base':>5}"
    for frac in fix_fractions:
        header += f" {'%d%%fixed' % int(frac * 100):>12}"
    print(header)
    print("-" * 100)

    for n, p, q in cases:
        for base in bases:
            results = all_results[(n, base)]
            row = f"{n:>8} {base:>5}"
            for r in results:
                row += f" {abs(r.corr_y):>12.4f}"
            print(row)

    # ---- PART 3: ROUNDING ACCURACY TABLE ----
    print()
    print("-" * 100)
    print("PART 3: Rounding Accuracy on Unknown x-Digits (fraction correct)")
    print("-" * 100)
    header = f"{'n':>8} {'base':>5} {'dx':>4}"
    for frac in fix_fractions:
        header += f" {'%d%%fixed' % int(frac * 100):>12}"
    print(header)
    print("-" * 100)

    for n, p, q in cases:
        for base in bases:
            results = all_results[(n, base)]
            _, dx, _ = _compute_digit_sizes(n, base)
            row = f"{n:>8} {base:>5} {dx:>4}"
            for r in results:
                row += f" {r.x_rounding_accuracy:>12.3f}"
            print(row)

    # ---- PART 4: Y-ROUNDING ACCURACY TABLE ----
    print()
    print("-" * 100)
    print("PART 4: Rounding Accuracy on y-Digits (fraction correct)")
    print("-" * 100)
    header = f"{'n':>8} {'base':>5} {'dy':>4}"
    for frac in fix_fractions:
        header += f" {'%d%%fixed' % int(frac * 100):>12}"
    print(header)
    print("-" * 100)

    for n, p, q in cases:
        for base in bases:
            results = all_results[(n, base)]
            _, _, dy = _compute_digit_sizes(n, base)
            row = f"{n:>8} {base:>5} {dy:>4}"
            for r in results:
                row += f" {r.y_rounding_accuracy:>12.3f}"
            print(row)

    # ---- PART 5: TREND ANALYSIS ----
    print()
    print("=" * 100)
    print("TREND ANALYSIS")
    print("=" * 100)

    # --- 5a: 0% → best non-degenerate level ---
    print()
    print("--- 5a: x-correlation trend (0% → highest non-degenerate fix level) ---")
    print()

    improving_x = 0
    flat_x = 0
    declining_x = 0
    total_x = 0

    for n, p, q in cases:
        for base in bases:
            results = all_results[(n, base)]
            corrs = [abs(r.corr_x_unknown) for r in results]
            unknowns = [r.num_unknown for r in results]
            degens = [r.x_corr_degenerate for r in results]
            # Find highest fix level with non-degenerate correlation
            best_idx = 0
            for idx in range(len(results) - 1, 0, -1):
                if unknowns[idx] >= 2 and not degens[idx]:
                    best_idx = idx
                    break
            if best_idx == 0:
                # Only 0% is non-degenerate — skip
                _, dx, _ = _compute_digit_sizes(n, base)
                print(
                    f"  n={n:>8}, base={base:>2} (dx={dx}): "
                    f"SKIPPED — too few non-degenerate levels for comparison"
                )
                continue

            total_x += 1
            frac_label = int(fix_fractions[best_idx] * 100)
            delta = corrs[best_idx] - corrs[0]
            if delta > 0.05:
                improving_x += 1
                tag = "IMPROVING (+)"
            elif delta < -0.05:
                declining_x += 1
                tag = "DECLINING (-)"
            else:
                flat_x += 1
                tag = "FLAT (~)"

            detail = "  ".join(
                f"{int(fix_fractions[i]*100)}%={corrs[i]:.3f}"
                for i in range(best_idx + 1)
                if unknowns[i] >= 2 and not degens[i]
            )
            print(
                f"  n={n:>8}, base={base:>2}: "
                f"{detail}  "
                f"delta(0→{frac_label}%)={delta:+.3f}  [{tag}]"
            )

    print()
    print(f"  Eligible series: {total_x}")
    print(f"    Improving:  {improving_x}")
    print(f"    Flat:       {flat_x}")
    print(f"    Declining:  {declining_x}")

    # --- 5b: y-correlation trend ---
    print()
    print("--- 5b: y-correlation trend (0% → 75% x-digits fixed) ---")
    print()

    improving_y = 0
    flat_y = 0
    declining_y = 0
    total_y = 0

    for n, p, q in cases:
        for base in bases:
            results = all_results[(n, base)]
            y_corrs = [abs(r.corr_y) for r in results]
            total_y += 1
            delta_y = y_corrs[-1] - y_corrs[0]
            if delta_y > 0.05:
                improving_y += 1
                tag = "IMPROVING (+)"
            elif delta_y < -0.05:
                declining_y += 1
                tag = "DECLINING (-)"
            else:
                flat_y += 1
                tag = "FLAT (~)"
            print(
                f"  n={n:>8}, base={base:>2}: "
                f"0%={y_corrs[0]:.3f} → 75%={y_corrs[-1]:.3f}  "
                f"delta={delta_y:+.3f}  [{tag}]"
            )

    print()
    print(f"  Total series: {total_y}")
    print(f"    Improving:  {improving_y}")
    print(f"    Flat:       {flat_y}")
    print(f"    Declining:  {declining_y}")

    # --- 5c: Rounding accuracy trend ---
    print()
    print("--- 5c: x-rounding accuracy trend (0% → 50% fixed) ---")
    print()

    for n, p, q in cases:
        for base in bases:
            results = all_results[(n, base)]
            # Compare 0% and 50% rounding accuracy (50% always has >= 2 unknown for base 2)
            acc_0 = results[0].x_rounding_accuracy
            acc_50 = results[2].x_rounding_accuracy
            delta_acc = acc_50 - acc_0
            tag = "+" if delta_acc > 0.01 else ("-" if delta_acc < -0.01 else "~")
            print(
                f"  n={n:>8}, base={base:>2}: "
                f"0%={acc_0:.3f} → 50%={acc_50:.3f}  "
                f"delta={delta_acc:+.3f}  [{tag}]"
            )

    # --- FINAL VERDICT ---
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print()

    # Use x-correlation as primary signal
    if improving_x > declining_x and improving_x > flat_x:
        print("PRIMARY: Spectral bootstrapping shows PROMISE.")
        print("  x-digit correlation IMPROVES when conditioning on known digits.")
        print("  This suggests an iterative algorithm: fix confident digits, re-estimate, repeat.")
    elif declining_x > improving_x:
        print("PRIMARY: Spectral bootstrapping is UNLIKELY to work.")
        print("  x-digit correlation DEGRADES when conditioning on known digits.")
        print("  The spectral signal is disrupted by partial fixing.")
    else:
        print("PRIMARY: INCONCLUSIVE for x-digit correlation bootstrapping.")

    print()

    # Use y-correlation as secondary signal
    if flat_y >= total_y * 0.6:
        print("SECONDARY: y-correlation is FLAT across conditioning levels.")
        print("  Fixing x-digits does NOT improve the y-estimate from SVD.")
        print("  The information from known x-digits does not propagate to y through")
        print("  the spectral structure of the least-squares Z matrix.")
    elif improving_y > declining_y:
        print("SECONDARY: y-correlation IMPROVES with more known x-digits.")
        print("  Information propagates from x to y through the spectral channel.")
    else:
        print("SECONDARY: y-correlation shows MIXED behavior.")

    print()
    print("INTERPRETATION:")
    if flat_y >= total_y * 0.6 and improving_x > 0:
        print("  The x-correlation improvement is likely MECHANICAL: fixing digits reduces the")
        print("  subspace dimension, making the remaining problem easier by elimination, NOT")
        print("  because the SVD is finding better signal. The y-flatness confirms this —")
        print("  the spectral structure of Z_ls does not carry exploitable rank-1 information")
        print("  about the factor digits. Spectral bootstrapping is a mirage.")
    elif improving_x > 0 and improving_y > 0:
        print("  BOTH x and y correlations improve with partial knowledge. This is genuine")
        print("  spectral signal amplification. An iterative spectral algorithm may work.")
    else:
        print("  No clear evidence of exploitable spectral structure in the carry-constrained")
        print("  least-squares solution.")


if __name__ == "__main__":
    run_experiment()
