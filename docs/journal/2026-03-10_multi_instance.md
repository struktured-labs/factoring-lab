# Multi-Instance Benchmark Results

**Date:** 2026-03-10
**Setup:** 50 balanced semiprimes per bit size (seeds 0-49), bit sizes [16, 20, 24, 28, 32]

## Purpose

Previous experiments used a single seed (seed=42) per bit size. This creates single-seed bias: one lucky or unlucky instance can skew conclusions. This experiment generates 50 independent instances per bit size and computes statistical summaries to establish confidence in our results.

## Classical Algorithms

All four classical algorithms were tested: TrialDivision, PollardRho, PollardPM1(B=10000), and ECM.

### Key Findings

**TrialDivision and PollardRho achieve 100% success rate across all bit sizes.** These are reliable baselines. PollardRho is consistently faster than TrialDivision, with the gap widening at larger bit sizes (at 32-bit: PollardRho median 0.107ms vs TrialDivision median 1.131ms -- a 10x difference).

**PollardPM1 shows strong seed-dependence.** Its success rate varies dramatically:
- 16-bit: 0% (!) -- all 50 instances failed
- 20-bit: 26%
- 24-bit: 46%
- 28-bit: 60%
- 32-bit: 78%

The counterintuitive increase in success rate with bit size is because larger balanced semiprimes are more likely to have at least one factor where p-1 is 10000-smooth. At 16-bit, the factors are only 8 bits each, so p-1 is small but the smoothness bound is generous; however, the issue is that PollardPM1 hits gcd=n (both factor orders are smooth) and fails to separate them, or the factors are too small for the algorithm's structure to help.

**ECM shows high variance.** Coefficient of variation (CV) ranges from 1.17 to 2.12 across bit sizes, meaning the standard deviation is larger than the mean. This is expected: ECM's runtime depends heavily on the group order of the random curve, which varies per curve and per input. Despite this variance, ECM achieves 100% success across all 250 instances.

## SMT Algorithms

Three SMT configurations were tested: raw (no digit constraints), base-16, and base-256, all with 15-second timeouts.

### Key Findings

**All three SMT encodings achieve 100% success across all 250 instances each.** No timeouts occurred even at 32 bits. This is a strong result -- single-seed experiments were representative of success/failure behavior.

**SMT runtime variance is moderate but consistent.** At 32-bit:
- Raw: mean=1.52s, std=1.03s (CV=0.68)
- Base-16: mean=1.39s, std=1.07s (CV=0.77)
- Base-256: mean=1.71s, std=0.98s (CV=0.57)

The max/min ratio at 32-bit is roughly 40x for all encodings, showing significant instance-dependent behavior.

**Digit constraints provide negligible benefit.** Across 50 instances:
- At 32-bit, base-16 is slightly faster than raw (mean 1.39s vs 1.52s), but base-256 is slightly slower (1.71s). None of these differences are statistically significant given the high variance.
- At smaller bit sizes, all three encodings are essentially equivalent.

This confirms the single-seed finding: digit-level convolution constraints do not meaningfully help Z3's CDCL solver. The solver's internal bitvector reasoning already handles these constraints effectively.

## Cross-Paradigm Comparison

**Classical algorithms dominate at all tested bit sizes.** At 32 bits:
- PollardRho median: 0.000107s
- Best SMT median: 1.274576s (base-16)
- Gap: ~12,000x

The gap grows exponentially with bit size, consistent with SMT's exponential scaling vs PollardRho's O(n^{1/4}) complexity.

## Variance and Representativeness

**Are single-seed results representative?**

For success/failure: Yes. TrialDivision, PollardRho, ECM, and all SMT variants show 100% success across 50 instances. PollardPM1 is the exception -- a single seed could show success or failure depending on the instance, and single-seed results are NOT representative for this algorithm.

For runtime: Partially. The ranking of algorithms is stable across instances (PollardRho always beats TrialDivision, classical always beats SMT). However, absolute runtime values can vary by 40x for SMT and 10x for ECM across instances. Reporting median with interquartile range is more appropriate than reporting a single runtime.

**Outliers:**

- PollardPM1 at 16-bit: Complete failure across all 50 instances. This is a systematic issue, not an outlier.
- ECM occasionally takes 10-15x longer than median (e.g., 15.5ms vs 0.5ms median at 24-bit). These are instances where many curves must be tried before finding one with a smooth group order.
- SMT max runtimes at 32-bit reach 3.9-5.1 seconds, but no timeouts. The 15s timeout provides adequate margin.

## Statistical Confidence in Encoding Comparison

With 50 instances per configuration, we can assess whether encoding differences are significant:

At 32-bit (the most interesting case):
- Raw vs Base-16: difference in means = 0.13s, pooled std ~ 1.05s. Effect size (Cohen's d) ~ 0.12. **Not significant.**
- Raw vs Base-256: difference = -0.19s, effect size ~ 0.19. **Not significant.**
- Base-16 vs Base-256: difference = 0.32s, effect size ~ 0.31. **Marginally interesting but not significant at p<0.05 with this sample size.**

We can conclude with high confidence that none of the digit-level constraint encodings provide a meaningful speedup over raw bitvector multiplication for Z3 on balanced semiprimes at these bit sizes.

## Conclusions

1. **50-instance experiments confirm single-seed findings** for algorithm rankings and success/failure patterns.
2. **PollardPM1 is the algorithm most sensitive to instance selection** -- always report success rates, not single outcomes.
3. **SMT encoding choice does not matter** -- raw, base-16, and base-256 perform equivalently within noise.
4. **Classical algorithms outperform SMT by 4+ orders of magnitude** at 32 bits, with the gap growing exponentially.
5. **For publication, report median and IQR** rather than single-instance runtimes, especially for ECM and SMT which show high variance.
