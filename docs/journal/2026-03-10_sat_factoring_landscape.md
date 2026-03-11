# SAT/SMT Factoring Landscape & Where Digit Convolution Sits

*March 10, 2026*

## Three encoding strategies in the literature

### 1. Circuit-based (standard approach)
Build a binary multiplication circuit, convert every gate into SAT clauses via
Tseytin transformation. This is what most papers use.

- **Performance**: Pure SAT handles ~20-30 bit semiprimes before timing out
- **Scaling**: Exponential, not competitive with even Pollard rho
- **Key paper**: [Encoding Basic Arithmetic Operations for SAT-Solvers](https://yurichev.com/mirrors/SAT_factor/Encoding%20Basic%20Arithmetic%20Operations%20for%20SAT-Solvers.pdf)
- **Tool**: [satfactor (GitHub)](https://github.com/sjneph/satfactor)

### 2. Hybrid SAT + Coppersmith/lattice (Ajani et al., 2024)
Don't make SAT do all the work. When the solver has guessed ~60% of a factor's
bits, call Coppersmith's lattice method to recover the rest.

- **Performance**: 768-bit numbers in 789s — but only with ~50% leaked bits
- **Without leaked bits**: Still exponential
- **Key result**: SAT+CAS gave 115x speedup over pure SAT at 768 bits
- **Paper**: [SAT and Lattice Reduction for Integer Factorization](https://arxiv.org/html/2406.20071v1)

### 3. Our digit convolution approach
Encode factoring as arithmetic-level constraints — digit convolutions with carry
propagation — and solve with Z3's SMT bitvector theory.

**What's different from both above:**
- No boolean circuit. Constraints are number-theoretic (`α_k ≡ c_k (mod b)`)
- Base is a tunable parameter (nobody else has this knob)
- SMT bitvector theory reasons about multiplication natively, not via bit-blasting
- Simpler encoding that preserves arithmetic meaning

## Performance comparison

| Approach | Encoding | No leaked bits | With leaked bits |
|----------|----------|---------------|-----------------|
| Circuit SAT (standard) | Boolean gates → CNF | ~20-30 bits | N/A |
| Hybrid SAT+Coppersmith | Circuit + lattice | Still hard | 768 bits (50% leaked) |
| Digit convolution (backtracking) | Arithmetic constraints | ~24 bits | N/A |
| **Digit convolution (Z3)** | **SMT bitvector + modular** | **~32 bits** | **Not tested yet** |
| Pollard rho (reference) | Not constraint-based | ~60-80 bits easily | N/A |

Z3 on our constraints already beats pure circuit-SAT by a few bits. The encoding
is simpler and preserves arithmetic structure.

## What's novel

1. **Nobody else encodes factoring as digit convolution + carry propagation into SMT.**
   Standard is circuit → CNF → SAT. We skip the circuit entirely.

2. **The base parameter is unexplored.** In circuit approaches there is no analog of
   "choose a base." Our finding that base choice affects Z3 performance is original.

3. **Digit constraints help Z3.** At 32 bits: base-10 constraints (2.1s) beat raw
   bitvector (3.4s). The carry-propagation structure gives Z3 useful intermediate lemmas.

## The big open question

Can the digit convolution constraints be reformulated as a lattice problem?

If the carry-propagation system `m_k = α_k + (m_{k-1} - c_{k-1})/b` can be
expressed as finding a short vector in some lattice, then LLL might solve it in
polynomial time — at least for certain bases or instance families. This would
connect our work to the Ajani et al. hybrid approach, but starting from a more
algebraic encoding.

## Sources

- [SAT and Lattice Reduction for Integer Factorization (Ajani et al., 2024)](https://arxiv.org/html/2406.20071v1)
- [On speeding up factoring with quantum SAT solvers (Mosca et al., 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7490379/)
- [Factoring semi-primes with (quantum) SAT-solvers](https://www.nature.com/articles/s41598-022-11687-7)
- [Encoding Basic Arithmetic Operations for SAT-Solvers](https://yurichev.com/mirrors/SAT_factor/Encoding%20Basic%20Arithmetic%20Operations%20for%20SAT-Solvers.pdf)
- [satfactor (GitHub)](https://github.com/sjneph/satfactor)
- [SAT and Lattice Reduction for Integer Factorization (ACM)](https://dl.acm.org/doi/10.1145/3666000.3669712)
