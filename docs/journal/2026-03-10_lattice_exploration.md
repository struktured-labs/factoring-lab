# Lattice Formulation of Digit Convolution Constraints

*March 10, 2026*

## Motivation

The digit convolution constraint system for factoring n = x * y is:

```
alpha_k = sum_{i=0}^{k} x_i * y_{k-i}    (digit convolution)
m_0 = alpha_0
m_k = alpha_k + (m_{k-1} - c_{k-1}) / b  (carry propagation)
m_k = c_k (mod b)                         (digit matching)
```

The modular constraints `m_k = c_k (mod b)` can be rewritten as `m_k = c_k + b*t_k` for integer carries `t_k`. This creates a system of linear equations with integer unknowns -- exactly the setting where lattice reduction (LLL) excels. The question: can LLL solve this efficiently?

## What was built

`src/factoring_lab/algorithms/lattice_convolution.py` implements:

1. **LLL basis reduction** (pure numpy, no external deps)
2. **Full linearization**: Treat products `z_{ij} = x_i * y_j` as independent variables. The carry-propagation constraints become a linear system `A @ [z, t] = c` solvable by lattice methods.
3. **Partial-knowledge solver**: Given x digits, solve the linear system for y via LLL.
4. **Diagnostic tools**: `LatticeAnalysis` class to verify constraint satisfaction and analyze lattice structure.

## Key findings

### The constraint system DOES have lattice structure -- partially

The carry propagation constraints are genuinely linear when expressed in terms of the products `z_{ij} = x_i * y_j` and carries `t_k`:

```
sum_{i+j=k} z_{ij} + t_{k-1} - b*t_k = c_k    for each k
```

This is a valid integer linear system. LLL can find short vectors in the solution lattice. The `verify_known_factorization` diagnostic confirms that true factorizations satisfy these constraints perfectly.

### The fundamental obstacle: the rank-1 constraint

For the solution to represent an actual factorization, the matrix `Z` where `Z[i,j] = z_{ij}` must have **rank 1** (i.e., `Z = x * y^T`). This is a **nonlinear** constraint that cannot be encoded in a lattice.

The lattice contains all integer vectors satisfying the linear carry constraints, but most of these do NOT correspond to valid factorizations. The valid solutions are a tiny subset (the rank-1 matrices) embedded in a much larger lattice.

This is the same fundamental barrier that separates polynomial factoring (where LLL works, because the problem is inherently linear in coefficient space) from integer factoring (where the multiplicative structure introduces nonlinearity).

### What works

- **Known-x solving**: When x is given, the system for y is fully linear. LLL can solve it. This is essentially Coppersmith's method: given partial information about one factor, lattice reduction recovers the rest. Verified working in tests.
- **Constraint verification**: The lattice correctly encodes the carry-propagation constraints. Known factorizations satisfy `A @ v = b` exactly.
- **Small cases via enumeration**: The algorithm falls back to trying small x values, which works for small semiprimes but is just trial division with extra steps.

### What doesn't work

- **Full lattice approach (no prior knowledge)**: LLL finds short vectors in the linearized lattice, but these are generally NOT rank-1. The shortest vector in the lattice does not correspond to the factorization.
- **Scaling**: Even if we could somehow enforce rank-1, the lattice dimension grows as O(d^2) in the number of digits, making LLL's O(d^6) cost problematic for large numbers.

## Analysis: why this was expected

The connection between lattices and factoring is well-studied. The key insight from the literature (and confirmed empirically here):

1. **Coppersmith's theorem** (1996): If you know approximately half the bits of a factor, LLL can recover the rest in polynomial time. This works because the remaining unknowns are few enough that the lattice has a uniquely short vector.

2. **Without partial knowledge**: The lattice dimension is too large relative to the gap between the target vector and other short vectors. LLL finds short vectors, but they're not the RIGHT short vectors.

3. **The rank-1 gap**: There is no known way to encode `z_{ij} = x_i * y_j` as a lattice constraint. This would require encoding multiplication as a linear operation, which is exactly what makes integer factoring hard.

## What would need to be true for LLL to work

For a pure lattice approach to factor n efficiently, one of these would need to hold:

1. **The rank-1 solution is the shortest vector**: This would require the factorization vector to be much shorter than all other lattice vectors. For random semiprimes, this appears false -- the lattice has many short vectors that aren't rank-1.

2. **A clever re-encoding eliminates the rank-1 constraint**: Perhaps a different variable choice or constraint formulation could make the nonlinearity disappear. We did not find such an encoding, and the polynomial-vs-integer factoring gap suggests none exists in general.

3. **Enough partial information is available**: With ~50% of bits known, Coppersmith/LLL works beautifully (as shown by Ajani et al., 2024). The digit convolution encoding might provide useful "partial information" in some bases, but our experiments did not find a base where this happens automatically.

## Connection to existing work

This exploration confirms the picture from `2026-03-10_sat_factoring_landscape.md`:

- **Hybrid approaches** (SAT + lattice) work because SAT provides the partial information that makes lattice reduction effective.
- **Pure lattice approaches** fail because they can't encode the multiplicative structure.
- **The digit convolution encoding** is a cleaner way to express the constraints, but doesn't change the fundamental hardness.

## Files created

- `src/factoring_lab/algorithms/lattice_convolution.py` -- Algorithm implementation
- `tests/test_lattice_convolution.py` -- 21 tests, all passing

## Next steps to consider

1. **Hybrid digit-convolution + LLL**: Use Z3/SAT to guess some digits, then LLL to recover the rest. The digit convolution encoding might provide a better "meeting point" than circuit-based approaches.
2. **Structured lattices**: The carry-propagation has a specific banded structure. Specialized lattice algorithms for banded/Toeplitz-like structures might exploit this.
3. **Approximate rank-1**: Instead of exact rank-1, look for lattice vectors that are "close to" rank-1. SVP approximation algorithms might help.
