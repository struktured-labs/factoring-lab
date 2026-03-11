# SDP Relaxation of Digit Convolution Rank-1 Constraint

*March 10, 2026*

## Motivation

Following the lattice exploration, the key unsolved piece is the rank-1 constraint: given the linearized system `sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k`, we need `Z[i,j] = x_i * y_j` (i.e., `Z = x * y^T` is rank-1). Semidefinite programming (SDP) can enforce positive semidefiniteness and minimize nuclear norm (trace for PSD matrices) as a convex relaxation of rank minimization. Can this recover factors?

## What was built

`src/factoring_lab/algorithms/sdp_convolution.py` implements:

1. **SDPConvolution**: ADMM-like iterations that alternate between satisfying the linear carry-propagation constraints and projecting Z onto the rank-1 PSD cone via SVD. Multiple random restarts with neighbor-checking during rounding.

2. **AlternatingProjection**: A simpler variant that initializes x randomly in [2, sqrt(n)], computes y = n/x, rounds, checks, and perturbs using digit-level gradient information.

3. **SDPAnalysis**: Diagnostic tools for measuring the integrality gap between continuous SDP-relaxed solutions and the true integer rank-1 solution.

## Key findings

### The alternating projection works surprisingly well — but it's not really SDP

The `AlternatingProjection` algorithm factors all test cases up to 24 bits, often faster than backtracking:

| Bit size | Alt. Proj. | Backtracking | SDP Relaxation |
|----------|-----------|-------------|----------------|
| 12-bit   | 5/5, 0.0002s avg | 5/5, 0.0007s avg | 5/5, 0.0014s avg |
| 16-bit   | 5/5, 0.0008s avg | 5/5, 0.0047s avg | 5/5, 0.0032s avg |
| 20-bit   | 5/5, 0.0018s avg | 5/5, 0.1372s avg | 5/5, 0.0229s avg |
| 24-bit   | 5/5, 0.0104s avg | 5/5, 2.1680s avg | 2/5, 0.0115s avg |

**However**: the alternating projection succeeds primarily because it randomly initializes x in [2, sqrt(n)] and checks whether `round(n/x)` divides n. With enough random restarts, this is essentially randomized trial division — it has nothing to do with SDP or digit convolution structure. The expected number of trials to find a factor p is O(sqrt(n)/p), same as Pollard's rho without the cycle-detection cleverness.

### The SDP relaxation itself fails at 24 bits

The `SDPConvolution` algorithm — which actually attempts the ADMM-like rank-1 projection — fails on 3 out of 5 24-bit instances. Its successes come from the alternating projection fallback built into the same restart loop, not from the SDP iterations converging to the correct rank-1 matrix.

### The integrality gap is large

The `SDPAnalysis` diagnostics reveal:

- **True rank-1 ratio**: Always 1.0000 (the true solution IS exactly rank-1, as expected)
- **Average relaxed gap**: ~0.29 for small semiprimes, ~0.39 for 20-bit — meaning random Z matrices satisfying the linear constraints have ~30-39% of their spectral mass outside the leading singular value
- **The gap grows with problem size**: More digits means a higher-dimensional Z matrix, which makes the rank-1 approximation looser

This confirms the fundamental issue: the linear constraints alone do not constrain Z to be anywhere near rank-1. The set of feasible Z matrices is a high-dimensional polytope, and the rank-1 matrices form a tiny nonlinear submanifold within it.

### Why the ADMM/projected gradient approach fails

The alternation between "satisfy linear constraints" and "project to rank-1" does not converge because:

1. **Non-convex feasible set**: The intersection of {Z : Z satisfies linear carry constraints} and {Z : Z is rank-1} is non-convex. Alternating projection between a convex set and a non-convex set has no convergence guarantee.

2. **Oscillation**: The algorithm oscillates between satisfying constraints (which pushes Z away from rank-1) and projecting to rank-1 (which violates constraints). The step size decays, but the iterates converge to a point that is neither constraint-satisfying nor rank-1.

3. **Lack of coupling**: The linear constraints couple z_{ij} values across different positions k, but the rank-1 projection treats Z as a whole matrix. There is no mechanism to propagate "this z_{ij} must decrease because position k's carry is wrong" through the rank-1 structure.

### What would make this work

A proper SDP solver (e.g., CVXPY with SCS/MOSEK) could solve the actual relaxation:

```
minimize    trace(Z)
subject to  M = [[1, x^T], [x, Z]]  is PSD
            sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k  for all k
            0 <= z_{ij} <= (b-1)^2
            0 <= x_i <= b-1
```

But even this would likely fail because:
- The PSD + linear constraints give a *relaxation* of rank-1, not rank-1 itself
- The integrality gap analysis shows the relaxation is loose
- Rounding from the SDP optimum to an integer rank-1 matrix has no guarantee
- For the relaxation to be tight, we'd need the SDP to have a unique optimum at the true factorization — but the feasible set is too large

## Connection to existing results

This confirms the picture from the lattice exploration:

- The carry-propagation constraints are genuinely linear and well-behaved
- The rank-1 constraint is the hard part, and **no convex relaxation captures it well**
- Approaches that work (alternating projection succeeding) do so by avoiding the rank-1 problem entirely (random trial division in disguise)
- The digit convolution formulation provides a clean decomposition of factoring into "easy linear part" + "hard nonlinear part," but doesn't make the hard part easier

## Files created

- `src/factoring_lab/algorithms/sdp_convolution.py` — Algorithm implementations (SDPConvolution, AlternatingProjection, SDPAnalysis)
- `tests/test_sdp_convolution.py` — 21 tests, all passing
- `scripts/sdp_comparison.py` — Benchmark comparison script

## Honest assessment

The SDP relaxation approach does not work as a factoring method. The integrality gap is too large, the non-convex alternating projection does not converge, and the successes are attributable to random trial division rather than any structural insight from the SDP formulation.

This is a **useful negative result**: it confirms that convex relaxation of the rank-1 constraint is insufficient, and that any approach based on the digit convolution formulation must directly tackle the nonlinear rank-1 structure rather than relaxing it away.

## Next steps to consider

1. **Non-convex optimization**: Instead of convex relaxation, try Riemannian optimization directly on the rank-1 manifold subject to the linear constraints
2. **Hybrid SDP + SAT**: Use SDP to get a warm start, then SAT/SMT to enforce integrality
3. **Tighter relaxations**: Sum-of-squares (SOS) or Lasserre hierarchy might give tighter bounds, but at much higher computational cost
