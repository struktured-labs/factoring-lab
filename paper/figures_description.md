# Figures and Tables for "Digit Convolution Constraints for Integer Factorization"

This document describes the figures and tables needed for the paper. Data sources are listed for each.

---

## Figure 1: Base Sweep Heatmap

**Description:** Heatmap showing Z3 runtime (seconds) with bit size (20, 24, 28, 32) on the y-axis and base (2, 3, 4, ..., 512) on the x-axis. Color scale should be log-transformed. Highlight power-of-2 bases with markers or borders.

**Data source:** `docs/journal/2026-03-10_base_sweep_results.md`, raw results table.

**Key point to convey:** Power-of-2 bases (16, 64, 128, 512) are consistently fast; non-power-of-2 bases are volatile.

---

## Figure 2: Scaling Laws (Log-Linear Plot)

**Description:** Log-linear plot of runtime vs. bit size (20--40 bits) for selected configurations: raw, base-8, base-16, base-64, base-512. Y-axis: log2(runtime in seconds). X-axis: bit size. Show fitted exponential lines with R^2 values in legend.

**Data source:** `docs/journal/2026-03-10_scaling_laws.md`.

**Key point to convey:** All configurations follow the same exponential growth class; digit constraints improve predictability (higher R^2) but not the exponent.

---

## Figure 3: Multi-Instance Box Plots

**Description:** Side-by-side box plots of Z3 runtime across 50 instances at each bit size (16, 20, 24, 28, 32) for three encodings: raw, base-16, base-256. Include individual data points as jittered dots.

**Data source:** `reports/multi_instance_smt.csv`.

**Key point to convey:** The distributions overlap substantially at 32 bits; no statistically significant difference between encodings.

---

## Figure 4: Classical vs. SMT Runtime Comparison

**Description:** Log-scale bar chart or scatter plot comparing median runtimes of TrialDivision, PollardRho, ECM, SMT-raw, SMT-base16, SMT-base256 at 32 bits across 50 instances. Use error bars showing IQR.

**Data source:** `reports/multi_instance_smt.csv`, `reports/multi_instance_classical.csv`.

**Key point to convey:** Classical algorithms are 4+ orders of magnitude faster. PollardRho median ~0.1 ms vs. SMT median ~1.3 s.

---

## Figure 5: Leaked Bits Phase Transition

**Description:** Line plot with bit size on x-axis (32, 48, 64, 80, 96, 128) and minimum leak fraction on y-axis. Secondary plot or inset: runtime vs. leak fraction for a fixed bit size (e.g., 64 bits) showing the sharp transition.

**Data source:** `docs/journal/2026-03-10_leaked_bits_results.md`, `reports/leaked_bits.csv`.

**Key point to convey:** Sharp phase transition at 50--70% leaked bits; the threshold increases roughly linearly with bit size.

---

## Figure 6: Lattice Structure Diagram

**Description:** Schematic (not data-driven) illustrating the decomposition of the factoring problem:
- Left: The carry-propagation lattice $\Lambda_n$ (depicted as a grid/lattice in 2D).
- Center: The rank-1 variety (depicted as a curved surface or manifold).
- Right: The intersection $\Lambda_n \cap \mathcal{R}_1$ (depicted as isolated points).
- Annotate with dimensions: lattice dim $= d_x d_y$, variety dim $= d_x + d_y - 1$.

**Key point to convey:** The intersection of the high-dimensional lattice with the low-dimensional rank-1 variety yields an exponentially sparse set of valid factorizations.

---

## Figure 7: SDP Integrality Gap

**Description:** Bar chart showing the integrality gap (fraction of spectral mass outside the leading singular value) for random feasible Z matrices at different bit sizes (12, 16, 20, 24). Include a horizontal line at 0 for the true rank-1 solution.

**Data source:** `docs/journal/2026-03-10_sdp_relaxation.md`.

**Key point to convey:** The gap grows with problem size (29% at small sizes, 39% at 20-bit), confirming that convex relaxation is too loose.

---

## Table 1: Summary of Theoretical Results

**Description:** Table summarizing all theorems and their status.

| Result | Type | Statement (informal) |
|--------|------|---------------------|
| Theorem 1 | Unconditional | Black-box: $\Omega(\|\Lambda_n \cap \mathcal{B}\|)$ queries |
| Theorem 2 | Unconditional | Randomized black-box: $\Omega(\|\Lambda_n \cap \mathcal{B}\|/6)$ expected |
| Theorem 3 | Conditional | If FACTORING $\notin$ BPP, then superpolynomial queries |
| Theorem 4 | Unconditional | Rank-1 fraction $\le 2^{-\Omega(d^2)}$ |
| Theorem 5 | Unconditional (generic) | Generic oracle: $2^{\Omega(d^2/\log d)}$ queries |

---

## Table 2: Rust vs. Python Backtracking Performance

**Description:** Already in Appendix A of the paper. Four-column table: bit size, Python time, Rust time, speedup.

**Data source:** `docs/journal/2026-03-10_rust_port.md`.

---

## Notes on Figure Generation

- All plots should use a consistent color palette (e.g., Viridis for heatmaps, categorical palette for line/bar plots).
- Use matplotlib or seaborn for Python-based generation.
- Scripts should be placed in `scripts/generate_figures.py` and read from `reports/*.csv`.
- Export as both PDF (for LaTeX) and PNG (for markdown preview).
- Figure 6 (schematic) should be created manually or with TikZ if targeting LaTeX output.
