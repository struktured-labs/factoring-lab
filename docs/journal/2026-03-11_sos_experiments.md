# SOS / Lasserre Hierarchy Experiments for Digit Convolution Factoring

*March 11, 2026*

## Motivation

The plain SDP relaxation of the rank-1 digit convolution constraint (from `sdp_convolution.py`) exhibited 30-39% integrality gaps. The Lasserre/SOS hierarchy is a systematic way to tighten SDP relaxations: degree-2k SOS captures all polynomial constraints up to degree 2k, and at sufficiently high degree becomes exact.

**Research question:** At what SOS degree does the relaxation become tight enough to distinguish the true factorization from spurious lattice points?

## Experimental Setup

- **Instances:** Balanced semiprimes at 8, 10, 12, 14, 16 bits (seed=42)
- **Bases:** 2, 3, 5, 10
- **SOS degrees:** 2 (standard SDP) and 4 (Lasserre level 2)
- **Solver:** cvxpy with SCS backend
- **Measurements:** SOS gap (fraction of eigenvalue mass outside top eigenvalue), factor recovery success, computation time

### Test semiprimes used

| Bits | n      | p   | q   |
|------|--------|-----|-----|
| 8    | 143    | 11  | 13  |
| 10   | 323    | 17  | 19  |
| 12   | 1927   | 41  | 47  |
| 14   | 5293   | 67  | 79  |
| 16   | 25591  | 157 | 163 |

## Key Findings

### Finding 1: Degree-4 SOS drives the gap to near zero

The degree-4 SOS relaxation achieves SOS gap < 0.002 in every solvable instance, compared to degree-2 gaps ranging from 0% to 9.3%. This is a dramatic tightening.

| Degree | Average SOS Gap | Instances with gap < 0.001 |
|--------|-----------------|---------------------------|
| 2      | 0.0284          | 6/20 (30%)                |
| 4      | 0.0001          | 18/19 (95%)               |

**Interpretation:** The 2x2 minor constraints (rank-1 conditions) and localizing matrices at degree 4 effectively eliminate the integrality gap. The moment matrix at degree 4 is powerful enough to capture the essential structure of the rank-1 constraint.

### Finding 2: Base-2 is special

For base 2, the degree-2 SOS already achieves gap = 0.0000 across all bit sizes. This is because binary digits are in {0, 1}, and the constraint x_i^2 = x_i (which follows from x_i*(1-x_i) >= 0 combined with x_i*(x_i-0) >= 0) fully pins down binary variables at the SDP level. In effect, for base 2, degree-2 SOS already captures the integrality constraint.

For larger bases, degree-2 shows non-trivial gaps (3-9%), which degree-4 then eliminates.

### Finding 3: Gap closure does NOT imply easy factor recovery

Despite near-zero SOS gaps, factor recovery from the relaxed solution succeeds in only:
- 4/20 cases at degree 2
- 5/19 cases at degree 4

The low recovery rate (26%) despite zero gap suggests:

1. The SDP solution is a *fractional* point very close to the rank-1 variety but not exactly on it.
2. Simple rounding of first moments (E[x_i]) to nearest integers loses critical correlation information.
3. The carries introduce coupling between digit positions that makes digit-by-digit rounding unreliable.

**This is a key negative result:** closing the SOS gap is necessary but not sufficient for factoring. The rounding problem remains hard even with a tight relaxation.

### Finding 4: Computational cost scales steeply

| Degree | Moment matrix size for base b, d digits | Typical solve time |
|--------|----------------------------------------|-------------------|
| 2      | 1 + dx + dy                            | 0.01 - 0.07s      |
| 4      | 1 + d + d(d+1)/2 where d = dx+dy       | 0.17 - 13.8s      |

The 16-bit / base-2 instance (d=17 digit variables) produces a 153x153 moment matrix at degree 4, which we cap as "too large" for reliable SCS performance. At degree 6, the moment matrix would be O(d^3), making even 12-bit instances prohibitive.

### Finding 5: Base affects the gap profile

Average SOS gap by base (both degrees combined):

| Base | Avg Gap  | Recovery Rate |
|------|----------|---------------|
| 2    | 0.0000   | 3/9 (33%)     |
| 3    | 0.0134   | 2/10 (20%)    |
| 5    | 0.0211   | 3/10 (30%)    |
| 10   | 0.0224   | 1/10 (10%)    |

Smaller bases have smaller gaps because:
- Fewer digit values to distinguish (tighter bounds)
- The quadratic bound x*(b-1-x) >= 0 is tighter when b-1 is small

But smaller bases mean more digit positions (d ~ log_b n), which makes degree-4 more expensive.

## Connection to SOS Hardness Literature

These results connect to several threads in the SOS/Lasserre hardness literature:

### Degree-4 sufficiency is surprising

For many combinatorial optimization problems (MAX-CUT, coloring, planted clique), degree-2 SOS already has significant gaps, and degree O(log n) or even O(n^epsilon) is needed to close them. Our finding that degree 4 essentially closes the gap for digit convolution factoring is noteworthy.

**Possible explanation:** The rank-1 constraint is *algebraically simple* (defined by degree-2 polynomials -- the 2x2 minors). The Lasserre hierarchy at degree 2k captures all polynomial constraints of degree <= 2k. Since the rank-1 conditions are degree-2 polynomials, their products (degree 4) are exactly what degree-4 SOS adds. This is why degree 4 is the "natural" level for this problem.

### The rounding barrier

The SOS literature distinguishes between:
1. **Relaxation tightness** (small integrality gap)
2. **Rounding algorithms** (extracting integer solutions from relaxed ones)

For planted problems (CSPs, clique), spectral rounding from the SOS solution often works. For factoring, the structure is different: there is exactly ONE valid solution (up to (p,q) vs (q,p)), hidden among exponentially many near-integer lattice points. Standard rounding heuristics fail because:
- The landscape near the solution has no "basin of attraction"
- Adjacent integer points (off by 1 digit) give n values that are far from the target
- Carry propagation creates long-range correlations between digits

### Conjecture

Based on these experiments, we conjecture:

**Conjecture (SOS degree for digit convolution).** *For an n-bit balanced semiprime in base b, degree-4 SOS is sufficient to make the relaxation essentially tight (gap < epsilon) for any constant epsilon > 0. However, rounding the degree-4 SOS solution to an integer factorization requires solving a problem that is at least as hard as the original factoring problem.*

This would mean that the Lasserre hierarchy quickly "understands" the algebraic structure of factoring, but converting that understanding to a discrete answer remains hard -- consistent with the broader theme that convex relaxations capture algebraic but not combinatorial structure.

### Connection to Theorem 4 from restricted_model_lower_bound.md

Theorem 4 shows rank-1 points are a $2^{-\Omega(d^2)}$ fraction of lattice points. Our SOS experiments show the continuous relaxation can approach these points (small gap) but cannot reliably land on them (low recovery). This supports the view that the "needle in a haystack" structure persists even with sophisticated convex relaxations.

## Reproducibility

```bash
uv run python scripts/sos_experiment.py
```

Results exported to `reports/sos_experiment.csv`.

## Files

- `src/factoring_lab/algorithms/sos_relaxation.py` -- SOS degree 2 and 4 implementations
- `scripts/sos_experiment.py` -- Experiment runner
- `tests/test_sos_relaxation.py` -- 17 tests (all passing)
- `reports/sos_experiment.csv` -- Raw results
