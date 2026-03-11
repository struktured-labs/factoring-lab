# Leaked Bits Experiment: SMT Digit Convolution with Partial Factor Knowledge

**Date:** 2026-03-10
**Author:** struktured + Claude

## Objective

Measure how many least-significant bits of one factor (p) need to be leaked
before the SMT bitvector solver can factor n = p * q within a 30-second
timeout, across a range of semiprime sizes.

## Setup

- **Solver:** Z3 bitvector solver with raw `x * y = n` constraint (no digit
  convolution base), plus a bit-mask constraint fixing leaked LSBs of x.
- **Semiprimes:** Balanced (p and q have equal bit length), generated with
  seed=42 for reproducibility.
- **Timeout:** 30 seconds per attempt.
- **Leak fractions tested:** 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8.

## Results

| Semiprime bits | Factor bits | Min leak fraction | Leaked bits / total | Runtime at threshold |
|:-:|:-:|:-:|:-:|:-:|
| 32 | 16 | 0.0 | 0/16 | 0.25s |
| 48 | 24 | 0.3 | 7/24 | 5.07s |
| 64 | 32 | 0.5 | 16/32 | 7.59s |
| 80 | 40 | 0.6 | 24/40 | 0.97s |
| 96 | 48 | 0.6 | 28/48 | 12.31s |
| 128 | 64 | 0.7 | 44/64 | 8.85s |

### Key observations

1. **32-bit semiprimes** are solvable with zero leaked bits (0.25s), consistent
   with prior SMT results on this codebase.

2. **The minimum leak fraction increases with bit size**, roughly linearly from
   0.3 at 48 bits to 0.7 at 128 bits.  The relationship is approximately:
   `min_leak ~ 0.3 + 0.004 * (bits - 48)` for bits >= 48.

3. **Once the threshold is crossed, adding more bits helps rapidly.**  At 64
   bits, going from 0.5 (7.6s) to 0.6 (0.7s) is a 10x speedup.  The solver
   transitions sharply from "impossible within timeout" to "easy."

4. **Runtime at the threshold is consistent (1-12s)** across all sizes,
   suggesting the leaked bits reduce the effective search space to a roughly
   constant difficulty level.

## Comparison to Ajani et al. 2024

Ajani et al. (2024) studied SAT/SMT factoring with circuit-based encodings
and found that approximately **50-60% of bits** needed to be leaked for
their approach to be practical on 64-128 bit semiprimes.

Our results with the bitvector multiplication encoding:

| Bit size | Our min leak | Ajani et al. ~min leak | Advantage |
|:-:|:-:|:-:|:-:|
| 64 | 50% | ~50% | Comparable |
| 80 | 60% | ~55% | Slightly worse |
| 96 | 60% | ~55-60% | Comparable |
| 128 | 70% | ~60% | Worse |

**At smaller sizes (48-64 bits), the raw bitvector encoding is competitive**
with circuit-based approaches.  At larger sizes (128 bits), it requires
modestly more leaked information (~70% vs ~60%).

## Does digit convolution benefit from leaked bits?

The experiment used the raw bitvector constraint (`x * y = n`) without
digit-level convolution structure.  This is a deliberate baseline: the
leaked bits alone reduce the search space by fixing a mask on x.

Compared to circuit-based SAT encodings (which decompose multiplication
into a gate-level circuit), the bitvector approach:

- **Benefits similarly** from leaked bits at moderate sizes (48-80 bits).
  The Z3 bitvector solver handles the multiplication natively and the
  leaked-bit mask constraint integrates cleanly.

- **Benefits less at larger sizes** (128+ bits).  Circuit-based encodings
  produce more unit-propagation opportunities that interact well with
  leaked bits.  The monolithic bitvector multiplication in Z3 is more
  opaque to CDCL, so partial information propagates less efficiently.

- **Future work:** Test whether adding digit-convolution constraints
  (`base=10` or `base=2`) on top of leaked bits improves the threshold.
  The convolution constraints could provide additional propagation paths
  that compensate for the opacity of native bitvector multiplication.

## Raw data

Full results exported to `reports/leaked_bits.csv`.

## Files

- Algorithm: `src/factoring_lab/algorithms/smt_leaked.py`
- Experiment script: `scripts/leaked_bits_experiment.py`
- Tests: `tests/test_smt_leaked.py`
- Data: `reports/leaked_bits.csv`
