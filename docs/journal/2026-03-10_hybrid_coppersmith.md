# Hybrid Digit-Convolution + Coppersmith Lattice Factoring

*March 10, 2026*

## Motivation

Two prior explorations reached complementary conclusions:
- **SMT with leaked bits** (`2026-03-10_leaked_bits_results.md`): Z3 can factor when ~50-70% of a factor's bits are known, but needs that information handed to it.
- **Lattice convolution** (`2026-03-10_lattice_exploration.md`): LLL can solve for one factor given the other (Coppersmith's method), but cannot factor from scratch due to the rank-1 constraint.

The hybrid idea: use digit convolution constraints to *generate* partial factor knowledge, then hand it to a lattice solver. Specifically, enumerate valid low-digit assignments from the convolution structure, and for each candidate, attempt Coppersmith-style recovery of the remaining digits.

## Approach

### Digit enumeration phase

Given n in base b, enumerate (x_low, y_low) pairs satisfying:
```
x_low * y_low = n  (mod b^depth)
```

At depth 1, this is the single-digit constraint x_0 * y_0 = c_0 (mod b). At depth 2, we extend to two-digit assignments satisfying the product modulo b^2, etc.

The number of valid pairs grows with depth (roughly b^depth candidates at each level), but each candidate provides more partial information to the lattice solver.

### Lattice recovery phase

For each candidate x_low, construct a Coppersmith-type lattice:
- We know p = x_low (mod b^depth)
- Build lattice basis from [n, b^depth, x_low] relationships
- Apply LLL to find short vectors
- Check if any vector reveals a factor of n

The implementation uses multiple lattice constructions (2D, 3D) plus a brute-force sweep for small remainders as a fallback.

## Results

### 32-bit semiprimes: 100% success at all depths

| Depth | Leak equiv | Success rate | Avg runtime |
|:-----:|:----------:|:------------:|:-----------:|
| 1     | 0.21       | 10/10        | 0.00s       |
| 2     | 0.42       | 10/10        | 0.00s       |
| 3     | 0.62       | 10/10        | 0.01s       |

At 32 bits, even depth-1 enumeration (knowing just the last decimal digit of each factor) gives enough information for the lattice + brute-force sweep to recover the factor. The search space after fixing p mod 10 is only ~5000 candidates, which the sweep handles instantly.

### 48-bit semiprimes: depth 3 needed

| Depth | Leak equiv | Success rate | Avg runtime |
|:-----:|:----------:|:------------:|:-----------:|
| 1     | 0.14       | 0/10         | 0.00s       |
| 2     | 0.28       | 0/10         | 0.04s       |
| 3     | 0.42       | 6/10         | 0.28s       |

At 48 bits, depths 1-2 provide insufficient partial information. Depth 3 (knowing 3 decimal digits, equivalent to ~42% of the factor's bits) succeeds 60% of the time. This aligns with the leaked-bits result that 48-bit semiprimes need ~30% leaked bits -- the digit convolution provides slightly less effective information per bit than random bit leaking.

### 64-bit semiprimes: depth 3 insufficient

| Depth | Leak equiv | Success rate | Avg runtime |
|:-----:|:----------:|:------------:|:-----------:|
| 1     | 0.10       | 0/10         | 0.00s       |
| 2     | 0.21       | 0/10         | 0.04s       |
| 3     | 0.31       | 0/10         | 0.45s       |

At 64 bits, depth 3 provides ~31% equivalent leak fraction, which is below the ~50% threshold observed for SMT leaked bits. The lattice construction in our simplified Coppersmith implementation is not powerful enough to compensate. Depth 4+ would provide more information but the enumeration cost grows as 10^4 = 10000 candidates per sample, which may be feasible but was not tested in this run.

## Comparison with SMT leaked bits

| Bit size | Hybrid depth=3 | SMT leaked 0.3 | SMT leaked 0.5 |
|:--------:|:--------------:|:--------------:|:--------------:|
| 32       | 10/10 (0.01s)  | 10/10 (instant)| 10/10 (instant)|
| 48       | 6/10 (0.28s)   | 7/10* (5s)     | 10/10 (instant)|
| 64       | 0/10 (0.45s)   | 0/10 (timeout) | 10/10 (7.6s)   |

*Estimated from leaked bits results at similar fractions.

The hybrid approach is **faster** than SMT when it succeeds (sub-second vs multiple seconds), because the lattice recovery is polynomial-time once enough information is available. But it **succeeds less often** at the same equivalent leak fraction, because:

1. Digit convolution gives *structured* partial information (low digits), not *random* bits scattered through the factor.
2. Our simplified Coppersmith implementation uses only basic LLL on small lattices, not the full Howgrave-Graham machinery with optimal polynomial choices.
3. The enumeration explores many candidate (x_low, y_low) pairs, most of which are wrong, wasting time.

## Key insights

### Digit information vs random bit information

Knowing the low k decimal digits of p gives us ~3.32k bits of information. But this information is *contiguous* and *low-order*, which is exactly the setting where Coppersmith's method is most effective. In contrast, SMT leaked bits fixes random positions throughout the factor.

Surprisingly, the contiguous low-order information is **less** effective per bit than random leaked bits for SMT solving. This is because:
- Random leaked bits create propagation constraints throughout the bitvector, pruning the search space more uniformly.
- Contiguous low bits only constrain the bottom of the factor, leaving the top completely free.
- The lattice method needs roughly half the bits to guarantee success (Coppersmith's theorem), while SMT can sometimes exploit less structured partial information.

### The enumeration bottleneck

At depth d in base b, we enumerate up to b^(2d) candidate pairs (though convolution constraints prune this significantly). For base 10:
- Depth 1: ~10 candidates (fast)
- Depth 2: ~100 candidates (fast)
- Depth 3: ~1000 candidates (manageable)
- Depth 4: ~10000 candidates (slow)
- Depth 5: ~100000 candidates (impractical)

The enumeration grows exponentially while the information gain grows only linearly (3.32 bits per depth level in base 10). This is the fundamental scaling problem.

### Where hybrid could win

The hybrid approach would be most valuable when:
1. The number is too large for pure SMT (>128 bits)
2. Enough digits can be enumerated cheaply to cross the Coppersmith threshold (~50% of bits)
3. A proper Coppersmith implementation with Howgrave-Graham lattices is used

For base 2^16 (65536), a single depth level gives 16 bits of information, and the enumeration cost is bounded by 65536^2 candidates per position. For a 256-bit semiprime (128-bit factors), we'd need depth ~4 in base 2^16 to reach the Coppersmith threshold, with ~10^19 candidates -- still impractical.

## Files

- Algorithm: `src/factoring_lab/algorithms/hybrid_coppersmith.py`
- Tests: `tests/test_hybrid_coppersmith.py` (26 tests, all passing)
- Experiment: `scripts/hybrid_experiment.py`
- Data: `reports/hybrid_coppersmith.csv`

## Conclusion

The hybrid digit-convolution + Coppersmith approach works as a proof of concept but does not provide a practical advantage over either pure SMT or pure lattice methods. The digit convolution structure provides *valid* partial information for Coppersmith recovery, confirming the theoretical connection. However, the enumeration cost to generate enough partial information grows faster than the lattice benefit, making the approach uncompetitive at cryptographic sizes.

The most promising direction would be using a higher base (reducing depth needed) combined with a full Coppersmith implementation, but the enumeration cost remains the fundamental bottleneck.
