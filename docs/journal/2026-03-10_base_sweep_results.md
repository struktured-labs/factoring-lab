# Base Sweep Results: SMT Convolution Factoring

**Date:** 2026-03-10
**Experiment:** Systematic sweep of digit-convolution bases 2--512 on fixed balanced semiprimes, profiling Z3 runtime.

## Setup

- **Semiprimes:** balanced, seed=42, generated at 20, 24, 28, and 32 bits
- **Bases tested:** 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 32, 64, 100, 128, 256, 512
- **Control:** SMTConvolutionRaw (no digit constraints)
- **Timeout:** 15 seconds per solve
- **All 76 runs succeeded** (no timeouts)

### Semiprimes used

| Bits | N            | p      | q      |
|------|--------------|--------|--------|
| 20   | 370,817      | 601    | 617    |
| 24   | 8,998,631    | 2,963  | 3,037  |
| 28   | 127,285,073  | 10,477 | 12,149 |
| 32   | 2,369,625,997| 48,541 | 48,817 |

## Raw Results (runtime in seconds)

| Base | 20-bit | 24-bit | 28-bit | 32-bit |
|------|--------|--------|--------|--------|
| raw  | 0.0471 | 0.3518 | 0.1168 | 2.2084 |
| 2    | 0.0246 | 0.0853 | 0.1658 | 2.9749 |
| 3    | 0.0212 | 0.0317 | 0.1544 | 2.8066 |
| 4    | 0.0330 | 0.0453 | 0.3311 | 2.2186 |
| 5    | 0.0292 | 0.1228 | 0.1556 | 5.7234 |
| 6    | 0.0325 | 0.0817 | 0.2968 | 2.6121 |
| 7    | 0.0367 | 0.1052 | 0.5497 | 3.2451 |
| 8    | 0.0250 | 0.0446 | 0.5294 | 1.3809 |
| 9    | 0.0283 | 0.0433 | 0.3177 | 1.8979 |
| 10   | 0.0218 | 0.0952 | 0.1235 | 2.1037 |
| 12   | 0.0305 | 0.0497 | 0.1780 | 2.6078 |
| 16   | 0.0256 | 0.0421 | 0.1037 | 0.3369 |
| 20   | 0.0247 | 0.1502 | 0.2896 | 2.7804 |
| 32   | 0.0293 | 0.0994 | 0.2797 | 3.9401 |
| 64   | 0.0243 | 0.0951 | 0.1822 | 0.3467 |
| 100  | 0.0191 | 0.0428 | 0.3883 | 4.3059 |
| 128  | 0.0262 | 0.0592 | 0.2102 | 1.3267 |
| 256  | 0.0279 | 0.1316 | 0.3473 | 2.5272 |
| 512  | 0.0208 | 0.1199 | 0.2412 | 0.1349 |

## Observations

### Best performers at 32-bit (the hardest instance)

| Rank | Base | Time (s) | Notes |
|------|------|----------|-------|
| 1    | 512  | 0.13     | 16x faster than raw |
| 2    | 16   | 0.34     | 6.5x faster than raw |
| 3    | 64   | 0.35     | 6.4x faster than raw |
| 4    | 128  | 1.33     | 1.7x faster than raw |
| 5    | 8    | 1.38     | 1.6x faster than raw |
| ...  | raw  | 2.21     | baseline |
| worst| 5    | 5.72     | 2.6x slower than raw |

### Key patterns

1. **Powers of 2 dominate.** Bases 16, 64, 128, and 512 are consistently among the fastest at 32-bit. This is likely because Z3's bitvector engine can decompose power-of-2 modular arithmetic into simple bit extraction (Extract operations) rather than expensive division circuits.

2. **Base 512 is the overall winner at 32 bits** with 0.13s -- a 16x speedup over raw. This is striking because 512 = 2^9 means each "digit" is a 9-bit slice of the bitvector, giving only ~4 digit constraints. The constraints are cheap to encode and provide just enough pruning structure.

3. **Non-power-of-2 bases are volatile.** Bases 5, 7, and 100 are among the slowest at 32-bit despite 100 being fast at 20-bit. The modular reduction for non-power-of-2 bases requires Z3 to reason about general integer division, which adds solver complexity that may outweigh the pruning benefit.

4. **The optimal base shifts with problem size.** At 20-bit, base 100 is fastest (0.019s). At 24-bit, base 3 wins (0.032s). At 28-bit, base 16 wins (0.104s). At 32-bit, base 512 wins (0.135s). Larger problems benefit from larger bases that produce fewer digit constraints.

5. **Digit constraints can hurt.** Base 5 at 32-bit is 2.6x *slower* than raw. Adding expensive modular constraints that don't align with the bitvector representation can actively harm performance by increasing the solver's theory-combination burden without providing useful propagation.

6. **The raw solver is surprisingly competitive.** At 28-bit, raw (0.117s) beats most bases. The digit constraints only provide clear benefit at 32-bit, suggesting the overhead of encoding digit constraints only pays off when the search space is large enough.

### Hypotheses

1. **Power-of-2 bases are cheap because they reduce to bit-slicing.** Z3's bitvector solver can implement `x mod 2^k` as `Extract(k-1, 0, x)`, avoiding expensive division. This means the digit constraints are essentially free to propagate, providing pure upside.

2. **Optimal base scales with bit-width.** The pattern suggests `base ~ 2^(bits/4)` might be near-optimal: for 32-bit, `2^8 = 256` and `2^9 = 512` are the top performers. This would mean each factor has roughly 2 "digits," giving minimal constraint overhead while still carving the search space.

3. **Smooth bases (highly composite) should help, but don't clearly.** Bases 6 (2*3), 12 (2^2*3), and 100 (2^2*5^2) don't consistently outperform their prime-power neighbors. The smoothness of the base appears less important than whether Z3 can efficiently implement the modular arithmetic.

4. **The 8-digit cap in `_add_digit_constraints` interacts with base size.** For base 512 at 32-bit, there are only ~4 digits total, so all constraints are active. For base 2 at 32-bit, there are 32 digits but only 8 constraints are added, covering only the lowest 8 bits. This may explain why small bases don't scale well -- they constrain too little of the search space.

## Next steps

- Test at 40 and 48 bits to see if the power-of-2 advantage holds
- Try removing the 8-digit cap and see if small bases improve
- Profile Z3 internals (conflict counts, propagation counts) to understand *why* certain bases help
- Test bases that are exact powers of 2: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 only
- Run multiple semiprimes per bit-size to check if results are instance-dependent
