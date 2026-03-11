# Scaling Laws: SMT Convolution Factoring with Digit Constraints

**Date:** 2026-03-10
**Experiment:** Extended base sweep at bit sizes [20, 24, 28, 32, 36, 40] with power-of-2 bases, plus exponential scaling law fits.

## Setup

- **Semiprimes:** balanced, seed=42
- **Bit sizes:** 20, 24, 28, 32, 36, 40
- **Bases tested:** powers of 2: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- **Control:** SMTConvolutionRaw (no digit constraints)
- **Timeout:** 30 seconds per solve
- **Failures:** raw timed out at 40-bit; base 64 also timed out at 40-bit

## Optimal Base per Bit Size

| Bits | Optimal Base | Runtime (s) | Raw Runtime (s) | Speedup |
|------|-------------|-------------|-----------------|---------|
| 20   | 1024        | 0.019       | 0.027           | 1.5x    |
| 24   | 8           | 0.026       | 0.028           | 1.1x    |
| 28   | 16          | 0.104       | 0.080           | 0.8x    |
| 32   | 512         | 0.135       | 1.702           | 12.6x   |
| 36   | 64          | 2.376       | 0.379           | 0.2x    |
| 40   | 16          | 6.948       | timeout (30s)   | >4.3x   |

### Observations on optimal base

The optimal base does **not** follow a clean scaling law. The hypothesis that `optimal_base ~ 2^(bits/4)` is rejected (R^2 = 0.09). The optimal base varies erratically: 1024, 8, 16, 512, 64, 16 across increasing bit sizes. This is likely because:

1. The optimal base is highly instance-dependent (single semiprime per bit size).
2. Z3's internal heuristics interact unpredictably with different constraint structures.
3. The 8-digit cap in `_add_digit_constraints` creates a sharp cutoff that favors different bases at different sizes.

## Exponential Scaling Fits

Model: `time = a * 2^(b * bits)`, fit via log-linear regression.

### Key results

| Configuration | b (exponent) | R^2   | Runtime doubles every |
|--------------|-------------|-------|----------------------|
| Raw          | 0.339       | 0.688 | 2.9 bits             |
| Base 64      | 0.378       | 0.959 | 2.6 bits             |
| Base 32      | 0.458       | 0.971 | 2.2 bits             |
| Base 2       | 0.477       | 0.956 | 2.1 bits             |
| Base 8       | 0.489       | 0.962 | 2.0 bits             |
| Base 16      | 0.488       | 0.870 | 2.0 bits             |
| Base 1024    | 0.503       | 0.975 | 2.0 bits             |

### Interpretation

**The raw solver has the lowest scaling exponent (0.339) but the worst fit (R^2 = 0.688).** The 36-bit instance is an outlier for raw: it solved in 0.38s despite 32-bit taking 1.7s. This suggests the raw solver's performance is highly variable and instance-dependent. Removing the 36-bit point would likely increase the raw exponent substantially.

**Digit-constrained bases have higher exponents (0.38-0.50) but much better fits (R^2 > 0.95).** This means digit constraints make runtime more *predictable* even if they don't reduce the asymptotic scaling rate.

**The best digit-constrained exponent (base 64, b=0.378) is close to the raw exponent (0.339).** The ratio is 1.12, meaning digit constraints do NOT change the scaling class -- they are in the same exponential growth regime.

## Scaling Class: Same or Different?

**Conclusion: Digit constraints do not change the scaling class.** The exponential growth rate `b` is approximately the same (within ~15%) for raw and the best digit-constrained configuration. What digit constraints provide is:

1. **Constant-factor speedups** at specific bit sizes (up to 12.6x at 32-bit).
2. **More predictable runtime** (higher R^2 in exponential fits).
3. **Ability to solve instances the raw solver cannot** (40-bit raw timed out, but base 16 succeeded in 6.9s).

The speedups are real but reflect a smaller constant `a` in `time = a * 2^(b * bits)`, not a smaller exponent `b`. As bit sizes increase, the exponential term dominates and the constant factor advantage shrinks relative to total runtime.

## Caveats

1. **Single instance per bit size.** The results may not generalize. A proper study needs 10+ semiprimes per bit size to average out instance-dependent fluctuations.
2. **Small bit range.** 20-40 bits spans only one order of magnitude in the exponent. The fit may not extrapolate to 64+ bits.
3. **Z3 non-determinism.** Solver heuristics can cause 2-3x runtime variation on the same instance. Repeated runs with different random seeds would strengthen conclusions.
4. **The 8-digit cap** in the constraint encoder artificially limits how many constraints are active. Removing this cap or making it adaptive could change the picture.
5. **Raw 36-bit anomaly.** The raw solver at 36-bit (0.38s) is suspiciously fast compared to 32-bit (1.70s). This single point significantly affects the raw scaling fit. It may be a lucky Z3 heuristic decision for this particular semiprime.

## Assessment of Publishability

**Not yet publishable as a standalone paper.** The results are suggestive but preliminary:

- **Pro:** The observation that power-of-2 bases provide constant-factor speedups via bit-slicing is novel and mechanistically clear. The 12.6x speedup at 32-bit is significant.
- **Pro:** The finding that digit constraints make runtime more predictable (higher R^2) is interesting from a solver-engineering perspective.
- **Con:** Single instances per bit size -- this is the biggest weakness. Need statistical rigor.
- **Con:** Bit range is too small. Need 48, 56, 64-bit data to make scaling claims.
- **Con:** No comparison to other factoring approaches (trial division, Pollard rho, ECM) at the same bit sizes.
- **Con:** No Z3 internals profiling (conflict counts, propagations) to explain *why* certain bases help.

**Recommended next steps for a paper:**
1. Run 20+ semiprimes per bit size with different seeds.
2. Extend to 48, 56, 64 bits (will require longer timeouts or parallelism).
3. Remove the 8-digit cap and test adaptive constraint depth.
4. Profile Z3 internals to build a mechanistic explanation.
5. Compare to Pollard rho / ECM baselines at the same bit sizes.
6. Test non-power-of-2 bases more systematically to confirm the bit-slicing hypothesis.
