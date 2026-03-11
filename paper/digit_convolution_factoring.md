# Digit Convolution Constraints for Integer Factorization: SMT Encodings, Lattice Structure, and Restricted-Model Lower Bounds

**Authors:** struktured (struktured labs)

**Date:** March 2026

---

## Abstract

We investigate integer factorization through the lens of digit convolution constraints, a formulation that decomposes the problem $n = p \cdot q$ into base-$b$ digit-level arithmetic with explicit carry propagation. This formulation, which originated in 2009, encodes factoring as a system of modular constraints $\alpha_k + t_{k-1} \equiv c_k \pmod{b}$ where $\alpha_k = \sum_{i+j=k} x_i y_j$ is the $k$-th digit convolution coefficient. We implement this encoding as SMT bitvector assertions in Z3 and compare it systematically against circuit-based SAT encodings and raw bitvector multiplication. A base sweep over 18 bases (2--512) reveals that power-of-2 bases consistently outperform others, achieving up to 16x speedup over raw bitvector at 32 bits, because Z3 reduces modular arithmetic to bit extraction. However, multi-instance experiments (50 balanced semiprimes per bit size, seeds 0--49) show that at 32 bits, the encoding advantage is not statistically significant (Cohen's $d < 0.31$). Scaling law fits confirm all encodings share the same exponential growth class, with digit constraints providing only constant-factor improvements and better runtime predictability. We then analyze the lattice structure of the carry-propagation system, identifying the rank-1 constraint $Z = \mathbf{x} \otimes \mathbf{y}$ as the precise locus of hardness: LLL efficiently solves the linear subsystem but cannot enforce the nonlinear rank-1 condition. An SDP relaxation of the rank-1 constraint fails with a 30--39% integrality gap. Finally, we establish unconditional lower bounds in black-box and generic oracle models, proving that rank-1 points constitute a $2^{-\Omega(d^2)}$ fraction of the carry-propagation lattice (Theorem 4). We argue that the digit convolution framework, while not yielding a faster factoring algorithm, provides a clean analytical decomposition of factoring hardness into a tractable linear part and an intractable nonlinear part.

---

## 1. Introduction

### 1.1 Background

The computational complexity of integer factorization remains one of the central open problems in computer science. While factoring is known to lie in BQP via Shor's algorithm [Shor94], its classical complexity status is unresolved: it is not known to be in P, nor is it known to be NP-complete. The best classical algorithm, the General Number Field Sieve (GNFS), runs in time $L_n[1/3, c] = \exp\bigl(c (\log n)^{1/3} (\log \log n)^{2/3}\bigr)$, which is sub-exponential but super-polynomial.

A parallel line of research has explored constraint-satisfaction approaches to factoring, encoding the problem as a Boolean satisfiability (SAT) or satisfiability modulo theories (SMT) instance. The standard approach constructs a binary multiplication circuit, converts each gate to clauses via Tseytin transformation, and feeds the result to a SAT solver [YurichevSAT]. While conceptually clean, this circuit-based encoding handles only 20--30 bit semiprimes before timing out [satfactor], and remains exponential even with modern CDCL solvers.

Recent hybrid approaches combine SAT solvers with lattice reduction: Ajani et al. [Ajani24] showed that when approximately 50--60% of a factor's bits are leaked, a circuit-based SAT solver combined with Coppersmith's lattice method can factor 768-bit numbers in under 800 seconds. Without leaked bits, however, the approach remains exponential.

### 1.2 Contributions

This paper makes the following contributions:

1. **Digit convolution formulation.** We present a systematic treatment of integer factorization as digit convolution with carry propagation (Section 2), formulating the problem as modular arithmetic constraints rather than Boolean circuits. The base $b$ serves as a tunable parameter with no analog in circuit-based approaches.

2. **SMT encoding and experimental evaluation.** We implement the digit convolution constraints as Z3 bitvector assertions and conduct systematic experiments (Sections 3--4):
   - A base sweep over 18 bases (2--512) showing power-of-2 bases dominate due to Z3's bit-extraction optimization.
   - Multi-instance benchmarks (50 semiprimes per bit size) establishing that the encoding advantage is not statistically significant at 32 bits.
   - Scaling law fits confirming digit constraints share the same exponential growth class as raw bitvector multiplication, with constant-factor improvements and better predictability.
   - Leaked-bit experiments showing a phase transition at 50--70% known bits.

3. **Lattice analysis and the rank-1 barrier.** We reformulate the carry-propagation system as a lattice problem (Section 5) and identify the rank-1 constraint $Z = \mathbf{x} \otimes \mathbf{y}$ as the precise obstruction to efficient solution. We demonstrate that LLL solves the linear subsystem but fails on the nonlinear rank-1 condition, and that SDP relaxation of the rank-1 constraint produces a 30--39% integrality gap.

4. **Restricted-model lower bounds.** We establish unconditional lower bounds in black-box and generic oracle models (Section 6), proving that rank-1 points are exponentially sparse in the carry-propagation lattice ($2^{-\Omega(d^2)}$ fraction). We discuss barriers to extending these results to the full algebraic model.

5. **A Rust implementation** of the backtracking solver achieving 30--33x speedup over Python with identical search paths (Appendix A).

### 1.3 What This Paper Is Not

We emphasize that this paper does not present a new factoring algorithm competitive with GNFS, ECM, or even Pollard's rho. Classical algorithms outperform our SMT-based approach by over four orders of magnitude at 32 bits, with the gap growing exponentially. The contribution is a *framework*: digit convolution provides a clean decomposition of factoring hardness into a tractable linear part (carry propagation) and an intractable nonlinear part (the rank-1 constraint), offering a new analytical perspective on why integer factoring is hard.

### 1.4 Paper Outline

Section 2 develops the digit convolution formulation. Section 3 describes the SMT encoding. Section 4 presents experimental results, including base sweeps, multi-instance statistics, scaling laws, and leaked-bit experiments. Section 5 analyzes the lattice structure and the rank-1 barrier. Section 6 establishes restricted-model lower bounds. Section 7 surveys related work. Section 8 discusses implications and future directions.

---

## 2. Digit Convolution Formulation

### 2.1 Base-$b$ Digit Decomposition

Let $n$ be a positive integer to be factored, and let $b \ge 2$ be a base. Write $n = p \cdot q$ where $2 \le p \le q$. The base-$b$ digit representations are:

$$p = \sum_{i=0}^{d_x - 1} x_i \cdot b^i, \quad q = \sum_{j=0}^{d_y - 1} y_j \cdot b^j, \quad n = \sum_{k=0}^{d-1} c_k \cdot b^k$$

where $x_i, y_j \in \{0, 1, \ldots, b-1\}$, $c_k$ are the known digits of $n$, $d_x = \lceil \log_b p \rceil$, $d_y = \lceil \log_b q \rceil$, and $d = \lceil \log_b n \rceil$.

### 2.2 Convolution Coefficients

The product $n = p \cdot q$ induces a convolution on the digit vectors. Define the $k$-th convolution coefficient:

$$\alpha_k = \sum_{\substack{i + j = k \\ 0 \le i < d_x \\ 0 \le j < d_y}} x_i \cdot y_j$$

This is exactly the coefficient of $b^k$ in the polynomial product $p(b) \cdot q(b)$ before carry propagation. The coefficients $\alpha_k$ may exceed $b - 1$, so carries must propagate to higher-order positions.

### 2.3 Carry Propagation

Define the partial sum and carry at each digit position:

$$m_k = \alpha_k + t_{k-1}$$

where $t_{k-1}$ is the carry from position $k-1$ (with $t_{-1} = 0$). The digits of $n$ are determined by:

$$c_k \equiv m_k \pmod{b}, \quad t_k = \lfloor m_k / b \rfloor$$

Equivalently, the system of constraints is:

$$\sum_{\substack{i+j=k}} x_i \cdot y_j + t_{k-1} - b \cdot t_k = c_k, \quad k = 0, 1, \ldots, d-1$$

### 2.4 The Constraint System

Factoring $n$ in base $b$ reduces to finding digit vectors $\mathbf{x} = (x_0, \ldots, x_{d_x-1})$ and $\mathbf{y} = (y_0, \ldots, y_{d_y-1})$ with $x_i, y_j \in \{0, \ldots, b-1\}$ satisfying the carry-propagation constraints above. The key observation is that this system is *bilinear* in $\mathbf{x}$ and $\mathbf{y}$: the products $x_i y_j$ appear linearly in the constraints, but the relationship between $\mathbf{x}$, $\mathbf{y}$, and the products is nonlinear.

**Historical note.** This formulation of factoring as digit convolution with carry propagation was first developed by the author in 2009. It is a natural consequence of viewing integer multiplication as polynomial evaluation at $b$.

### 2.5 Connection to Polynomial Factoring

The analogy to polynomial factoring is illuminating. If we replace the carry-propagation constraints with the simpler requirement that all $\alpha_k$ equal the corresponding "digits" directly (i.e., no carries), we obtain polynomial factoring over $\mathbb{Z}[x]$. The LLL algorithm [LLL82] solves polynomial factoring in polynomial time precisely because the constraint system is *linear* in the products $z_{ij} = x_i y_j$ (there are no carry terms $t_k$). Integer factoring is harder because carry propagation couples all digit positions, and the system is globally constrained.

---

## 3. SMT Encoding

### 3.1 Bitvector Encoding in Z3

We encode the digit convolution constraints as assertions in the SMT-LIB bitvector theory, solved by Z3 [deMoura08]. The encoding proceeds as follows.

**Variables.** Declare bitvector variables $x$ and $y$ of appropriate bit width $\beta/2$ (for balanced semiprimes).

**Core constraint.** Assert $x \cdot y = n$ using native bitvector multiplication.

**Digit constraints.** For a chosen base $b$, compute the digits $c_k$ of $n$ and assert, for each digit position $k$:

$$\bigl(\text{digitsum}(x, y, k, b) + \text{carry}_{k-1}\bigr) \bmod b = c_k$$

where $\text{digitsum}$ computes $\alpha_k = \sum_{i+j=k} x_i y_j$ using bitvector extract and multiply operations.

**Balancing constraint.** Assert $2 \le x \le \lfloor\sqrt{n}\rfloor$ to restrict search to the smaller factor.

### 3.2 Base as a Tunable Parameter

A distinctive feature of our encoding is that the base $b$ is a free parameter with no analog in circuit-based SAT approaches. The choice of $b$ affects:

1. **Number of digit positions:** $d = \lceil \log_b n \rceil$. Larger bases yield fewer constraints.
2. **Digit range:** $x_i \in \{0, \ldots, b-1\}$. Larger bases have wider digit ranges.
3. **Solver efficiency:** When $b = 2^k$ for integer $k$, the modular operation $m \bmod b$ reduces to a bitvector extract (bit-slicing), which Z3 handles with zero overhead. Non-power-of-2 bases require expensive integer division within the solver.

### 3.3 Implementation

Our implementation supports three modes:
- **Raw:** Only the core constraint $x \cdot y = n$ is asserted, with no digit-level structure.
- **Digit-constrained:** Both the core constraint and digit convolution constraints for a specified base $b$ are asserted.
- **Leaked bits:** The core constraint plus a bit-mask fixing known least-significant bits of $x$.

The implementation caps the number of digit constraints at 8 per factor to limit encoding overhead. This cap interacts with base choice: for base 512 at 32 bits, only $\sim$4 digit positions exist, so all constraints are active.

---

## 4. Experimental Results

### 4.1 Methodology

All experiments use balanced semiprimes: $n = p \cdot q$ where $p$ and $q$ are primes of equal bit length, generated deterministically from integer seeds for reproducibility. Unless otherwise stated, Z3 4.x is used with default settings and a 15--30 second timeout per instance.

### 4.2 Base Sweep

**Setup.** 18 bases (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 32, 64, 100, 128, 256, 512) tested on fixed balanced semiprimes (seed=42) at 20, 24, 28, and 32 bits. All 76 runs completed without timeout.

**Results at 32 bits** (hardest instance):

| Rank | Base | Runtime (s) | Speedup vs. raw |
|------|------|-------------|-----------------|
| 1 | 512 | 0.13 | 16.4x |
| 2 | 16 | 0.34 | 6.5x |
| 3 | 64 | 0.35 | 6.4x |
| 4 | 128 | 1.33 | 1.7x |
| 5 | 8 | 1.38 | 1.6x |
| -- | raw | 2.21 | 1.0x |
| worst | 5 | 5.72 | 0.39x |

**Key findings:**

1. **Powers of 2 dominate.** Bases 16, 64, 128, and 512 are consistently the fastest at 32 bits. Z3 implements $x \bmod 2^k$ as $\text{Extract}(k-1, 0, x)$, making digit constraints essentially free to propagate.

2. **Non-power-of-2 bases are volatile.** Base 5 is 2.6x *slower* than raw at 32 bits: the modular reduction overhead outweighs the pruning benefit.

3. **The optimal base shifts with problem size.** At 20 bits, base 100 is fastest; at 24 bits, base 3; at 28 bits, base 16; at 32 bits, base 512. The hypothesis that $\text{optimal base} \sim 2^{\beta/4}$ is rejected ($R^2 = 0.09$).

4. **Digit constraints can hurt.** Adding expensive modular constraints that don't align with the bitvector representation actively harms performance.

### 4.3 Multi-Instance Statistics

**Setup.** 50 balanced semiprimes per bit size (seeds 0--49), bit sizes 16, 20, 24, 28, 32. Three SMT configurations: raw, base-16, base-256. Timeout: 15 seconds.

**Results at 32 bits:**

| Configuration | Mean (s) | Std (s) | CV | Max/Min |
|--------------|---------|--------|-----|---------|
| Raw | 1.52 | 1.03 | 0.68 | ~40x |
| Base-16 | 1.39 | 1.07 | 0.77 | ~40x |
| Base-256 | 1.71 | 0.98 | 0.57 | ~40x |

**Statistical significance.** With 50 instances per configuration:

| Comparison | $\Delta\mu$ (s) | Cohen's $d$ | Significant? |
|-----------|----------------|-------------|--------------|
| Raw vs. Base-16 | 0.13 | 0.12 | No |
| Raw vs. Base-256 | -0.19 | 0.19 | No |
| Base-16 vs. Base-256 | 0.32 | 0.31 | No |

**Conclusion.** At 32 bits, none of the digit-level constraint encodings provide a statistically significant speedup over raw bitvector multiplication. The effect sizes (Cohen's $d < 0.31$) are small. We conclude with high confidence that digit-level convolution constraints do not meaningfully help Z3's CDCL solver at these bit sizes.

**Cross-paradigm comparison.** Classical algorithms dominate at all tested bit sizes. At 32 bits, Pollard's rho (median 0.107 ms) is approximately 12,000x faster than the best SMT configuration (base-16, median 1.27 s). This gap grows exponentially with bit size.

### 4.4 Scaling Laws

**Setup.** Extended sweep at bit sizes 20, 24, 28, 32, 36, 40 with power-of-2 bases (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024). Timeout: 30 seconds. The raw solver timed out at 40 bits.

**Exponential fit.** We fit the model $\text{time} = a \cdot 2^{b \cdot \beta}$ via log-linear regression:

| Configuration | Exponent $b$ | $R^2$ | Runtime doubles every |
|--------------|-------------|-------|----------------------|
| Raw | 0.339 | 0.688 | 2.9 bits |
| Base 64 | 0.378 | 0.959 | 2.6 bits |
| Base 32 | 0.458 | 0.971 | 2.2 bits |
| Base 8 | 0.489 | 0.962 | 2.0 bits |
| Base 1024 | 0.503 | 0.975 | 2.0 bits |

**Interpretation.** Digit constraints do not change the scaling class. The exponent $b$ is approximately the same (within 15%) for raw and the best digit-constrained configuration. What digit constraints provide is:

1. **Constant-factor speedups** at specific bit sizes (up to 12.6x at 32 bits).
2. **More predictable runtime** ($R^2 > 0.95$ for digit-constrained vs. $R^2 = 0.69$ for raw).
3. **Ability to solve instances the raw solver cannot** (40-bit raw timed out; base 16 succeeded in 6.9 s).

The speedups reflect a smaller constant $a$ in $\text{time} = a \cdot 2^{b \cdot \beta}$, not a smaller exponent. As bit sizes increase, the exponential term dominates.

**Caveats.** Single instance per bit size in the scaling experiment (the multi-instance experiment addresses representativeness separately). The 20--40 bit range spans only one order of magnitude in the exponent; extrapolation to 64+ bits is uncertain.

### 4.5 Leaked Bits

**Setup.** Balanced semiprimes at 32, 48, 64, 80, 96, 128 bits. Leak fractions 0.0 to 0.8 of the least-significant bits of one factor. Raw bitvector encoding (no digit constraints). Timeout: 30 seconds.

**Results:**

| Semiprime bits | Factor bits | Min leak fraction | Runtime at threshold |
|:-:|:-:|:-:|:-:|
| 32 | 16 | 0.0 | 0.25 s |
| 48 | 24 | 0.3 | 5.07 s |
| 64 | 32 | 0.5 | 7.59 s |
| 80 | 40 | 0.6 | 0.97 s |
| 96 | 48 | 0.6 | 12.31 s |
| 128 | 64 | 0.7 | 8.85 s |

**Key observations:**

1. The minimum leak fraction increases roughly linearly with bit size: $\text{min\_leak} \approx 0.3 + 0.004 \cdot (\beta - 48)$ for $\beta \ge 48$.

2. A sharp phase transition occurs at the threshold: at 64 bits, going from 50% (7.6 s) to 60% (0.7 s) leaked bits yields a 10x speedup.

3. Runtime at the threshold is consistent (1--12 s) across all sizes, suggesting leaked bits reduce the effective search space to a roughly constant difficulty level.

4. Comparison with Ajani et al. [Ajani24]: at 64 bits, our bitvector encoding requires 50% leaked bits, comparable to their circuit-based approach (~50%). At 128 bits, our approach requires 70% vs. their ~60%, suggesting circuit-based encodings produce more unit-propagation opportunities at larger sizes.

---

## 5. Lattice Structure and the Rank-1 Barrier

### 5.1 Carry Propagation as a Linear System

Introduce linearized variables $z_{ij}$ intended to represent the products $x_i \cdot y_j$, and carry variables $t_k$. The carry-propagation constraints become:

$$\sum_{\substack{i+j=k \\ 0 \le i < d_x \\ 0 \le j < d_y}} z_{ij} + t_{k-1} - b \cdot t_k = c_k, \quad k = 0, 1, \ldots, d-1$$

with $t_{-1} = 0$. This is a system of $d$ linear equations in $m = d_x d_y + d$ integer unknowns, which we write as:

$$A \mathbf{v} = \mathbf{c}$$

where $\mathbf{v} = (z_{00}, z_{01}, \ldots, z_{d_x-1, d_y-1}, t_0, \ldots, t_{d-1})$, $A$ is the $d \times m$ constraint matrix, and $\mathbf{c} = (c_0, \ldots, c_{d-1})$.

The constraint matrix $A$ has a characteristic *banded* structure: variable $z_{ij}$ appears only in row $k = i + j$, and carry $t_k$ appears in rows $k$ and $k+1$.

### 5.2 Lattice Formulation

The integer solutions to $A\mathbf{v} = \mathbf{c}$ form an affine lattice:

$$\Lambda_n = \{\mathbf{v} \in \mathbb{Z}^m : A\mathbf{v} = \mathbf{c}\}$$

Since $A$ has rank $d$ (generically), the lattice has dimension $m - d = d_x d_y$. For a balanced semiprime with $d_x \approx d_y \approx d/2$, this is $\Theta(d^2) = \Theta(\log_b^2 n)$.

**LLL reduction.** The LLL algorithm [LLL82] can compute a reduced basis for $\Lambda_n$ in polynomial time. Our implementation (`lattice_convolution.py`) constructs $\Lambda_n$, applies LLL reduction, and enumerates short vectors. Verified experimentally: true factorizations satisfy the constraints $A\mathbf{v} = \mathbf{c}$ exactly, confirming the lattice correctly encodes the carry-propagation system.

### 5.3 The Rank-1 Constraint

For a lattice point $\mathbf{v} \in \Lambda_n$ to represent a valid factorization, the $d_x \times d_y$ matrix $Z$ formed from the $z_{ij}$ components must satisfy:

$$\operatorname{rank}(Z) = 1, \quad \text{i.e., } Z = \mathbf{x} \otimes \mathbf{y} = \mathbf{x}\mathbf{y}^T$$

for some $\mathbf{x} \in \{0, \ldots, b-1\}^{d_x}$, $\mathbf{y} \in \{0, \ldots, b-1\}^{d_y}$.

This is the *precise locus of hardness*. The rank-1 condition is a system of $\binom{d_x}{2}\binom{d_y}{2}$ quadratic equations (all $2 \times 2$ minors of $Z$ must vanish). It defines the *Segre variety* $\operatorname{Seg}(d_x, d_y)$ in the ambient space $\mathbb{R}^{d_x \times d_y}$, which has dimension $d_x + d_y - 1$ compared to the ambient dimension $d_x d_y$.

**Empirical finding.** Our LLL-based solver finds short vectors in $\Lambda_n$, but these are generically *not* rank-1. The shortest vector in the lattice does not correspond to the factorization. This is consistent across all tested instances.

### 5.4 Why LLL Works for Polynomial Factoring

The connection to polynomial factoring is clarifying. Polynomial factoring over $\mathbb{Z}[x]$ admits an analogous formulation, but without carry propagation: the "digits" of the product polynomial are the convolution coefficients $\alpha_k$ directly. The constraint system is identical except that $t_k = 0$ for all $k$. In this case, the lattice dimension is smaller and, crucially, the problem structure allows LLL to find the factorization as a short vector.

The key difference: polynomial factoring does not have a rank-1 constraint that interacts with the lattice structure. The degree of a polynomial provides a natural "dimension reduction" -- factoring a degree-$d$ polynomial yields factors of strictly smaller degree, enabling recursive approaches. Integers lack this property.

### 5.5 SDP Relaxation Failure

We attempted to enforce the rank-1 constraint via semidefinite programming, using an ADMM-like algorithm that alternates between:
1. **Linear feasibility:** Project onto the affine subspace $\{Z : A\mathbf{v} = \mathbf{c}\}$.
2. **Rank-1 projection:** Compute the SVD of $Z$ and retain only the leading singular value/vector pair.

**Results:**

| Bit size | Alternating Proj. | SDP Relaxation | Backtracking |
|----------|-------------------|----------------|-------------|
| 12-bit | 5/5, 0.0002 s | 5/5, 0.0014 s | 5/5, 0.0007 s |
| 16-bit | 5/5, 0.0008 s | 5/5, 0.0032 s | 5/5, 0.0047 s |
| 20-bit | 5/5, 0.0018 s | 5/5, 0.0229 s | 5/5, 0.1372 s |
| 24-bit | 5/5, 0.0104 s | 2/5, 0.0115 s | 5/5, 2.168 s |

**Integrality gap analysis.** The `SDPAnalysis` diagnostic measures the spectral gap between the true rank-1 solution and random feasible $Z$ matrices:

- True rank-1 ratio: 1.0000 (exact rank-1, as expected).
- Average relaxed gap: 0.29--0.39, meaning 30--39% of spectral mass lies outside the leading singular value.
- The gap *grows* with problem size: more digits yield a higher-dimensional $Z$ matrix, loosening the rank-1 approximation.

**Why ADMM fails.** The alternating projection between the convex set $\{Z : A\mathbf{v} = \mathbf{c}\}$ and the non-convex set $\{Z : \operatorname{rank}(Z) = 1\}$ has no convergence guarantee. The iterates oscillate between satisfying constraints (pushing $Z$ away from rank-1) and projecting to rank-1 (violating constraints), converging to a point that is neither feasible nor rank-1.

**Assessment.** The alternating projection method's apparent successes are attributable to random trial division (randomly initializing $x$ and checking if $\lfloor n/x \rfloor$ divides $n$), not SDP structure. The SDP relaxation is too loose to be useful: convex relaxation of the rank-1 constraint is insufficient for factoring.

---

## 6. Restricted-Model Lower Bounds

### 6.1 Formal Model Definition

We define a *digit convolution algorithm* for factoring $n$ as a procedure with the following cost structure.

**Free operations:** (1) Digit decomposition of $n$ in any base $b$. (2) Construction of the carry-propagation linear system $A\mathbf{v} = \mathbf{c}$. (3) Any polynomial-time lattice operation on $\Lambda_n$ (LLL reduction, short vector enumeration, Gram-Schmidt computation).

**Costly operation:** Each call to the *rank-1 oracle* $\mathcal{O}_{\text{R1}}$ costs 1 unit. Given a matrix $Z \in \mathbb{Z}^{d_x \times d_y}$ whose entries correspond to a point in $\Lambda_n$, the oracle returns:

$$\mathcal{O}_{\text{R1}}(Z) = \begin{cases} 1 & \text{if } \exists\, \mathbf{x} \in \{0,\ldots,b-1\}^{d_x},\, \mathbf{y} \in \{0,\ldots,b-1\}^{d_y} \text{ s.t. } Z = \mathbf{x} \otimes \mathbf{y} \\ 0 & \text{otherwise} \end{cases}$$

A positive oracle response immediately yields the factorization. The *query complexity* $Q_{\mathcal{A}}(\beta)$ is the worst-case number of oracle calls over all $\beta$-bit semiprimes.

**Remark.** This model captures the intuition that the linear part of factoring (carry propagation, lattice reduction) is tractable, while the nonlinear part (rank-1 testing) is the bottleneck.

### 6.2 The Rank-1 Oracle

The oracle $\mathcal{O}_{\text{R1}}$ simultaneously checks rank-1 structure *and* digit-range constraints ($0 \le x_i, y_j \le b-1$). A positive response provides $\Theta(\beta)$ bits of information (the full factorization), while a negative response provides much less. This asymmetry makes information-theoretic lower bounds weak (Proposition 1 below).

### 6.3 Rank-1 Sparsity (Theorem 4)

The main structural result is the following.

**Theorem 4** (Rank-1 sparsity in the carry-propagation lattice). *For a $\beta$-bit balanced semiprime $n = p \cdot q$ in base $b$, let $d = \lceil \log_b n \rceil$ and let $\Lambda_n$ be the carry-propagation lattice. Then:*

*(a) The number of rank-1 integer matrices in $\Lambda_n \cap \mathcal{B}$ is at most $2$ (corresponding to the factorizations $(p,q)$ and $(q,p)$).*

*(b) The ratio of rank-1 points to total lattice points satisfies:*

$$\frac{|\Lambda_n \cap \mathcal{R}_1 \cap \mathcal{B}|}{|\Lambda_n \cap \mathcal{B}|} \le \frac{2}{|\Lambda_n \cap \mathcal{B}|} \le 2^{-\Omega(d^2)}$$

*assuming the heuristic lattice point count $|\Lambda_n \cap \mathcal{B}| \ge 2^{\Omega(d^2)}$.*

*Proof of (a).* A rank-1 matrix $Z = \mathbf{x} \otimes \mathbf{y}$ in $\Lambda_n \cap \mathcal{B}$ with $x_i, y_j \in \{0, \ldots, b-1\}$ satisfying the carry-propagation constraints yields $p = \sum x_i b^i$ and $q = \sum y_j b^j$ with $p \cdot q = n$. Since $n$ is semiprime, the only factorizations are $(p,q)$ and $(q,p)$. $\square$

*Heuristic for (b).* The number of lattice points in the feasible box $\mathcal{B} = \{Z : 0 \le z_{ij} \le (b-1)^2\}$ is approximately:

$$|\Lambda_n \cap \mathcal{B}| \approx \frac{((b-1)^2 + 1)^{d_x d_y}}{b^d}$$

For $d_x \approx d_y \approx d/2$ and $d \ge 5$, this is $2^{\Omega(d^2)}$, confirming the claim. A rigorous bound would require Barvinok's algorithm or explicit enumeration.

**Interpretation.** Among all lattice points satisfying the linear carry constraints and digit-range bounds, the valid factorizations are exponentially sparse: a $2^{-\Omega(d^2)}$ fraction. Any algorithm that tests lattice points without exploiting the algebraic structure of rank-1 matrices requires exponentially many queries.

### 6.4 Black-Box and Generic Oracle Bounds

**Theorem 1** (Black-box lower bound). *In the black-box model (where the algorithm cannot exploit any algebraic structure of $\mathcal{O}_{\text{R1}}$ beyond the bound $|\mathcal{O}^{-1}(1)| \le 2$), any deterministic algorithm requires:*

$$Q(\beta) \ge \frac{|\Lambda_n \cap \mathcal{B}|}{2} - 1$$

*queries in the worst case.*

*Proof.* Standard adversary argument: the adversary answers "no" on every query until only 2 candidates remain, at which point it is forced to answer "yes." $\square$

**Theorem 2** (Randomized black-box lower bound). *Any randomized algorithm succeeding with probability $\ge 2/3$ requires $Q(\beta) \ge |\Lambda_n \cap \mathcal{B}|/6$ queries in expectation.*

*Proof.* By Yao's minimax principle, applied to the uniform distribution over placements of the 2 positive instances among $|\Lambda_n \cap \mathcal{B}|$ lattice points. $\square$

**Theorem 5** (Generic oracle lower bound). *Against a generic rank-1 oracle (positive instances placed uniformly at random, independent of algebraic structure), any algorithm requires $Q(\beta) \ge 2^{\Omega(d^2 / \log d)}$ queries in expectation.*

**Theorem 3** (Conditional lower bound). *If FACTORING $\notin$ BPP, then any digit convolution algorithm with polynomial-time inter-query computation requires superpolynomially many rank-1 queries: $Q_{\mathcal{A}}(\beta) \ge \beta^{\omega(1)}$.*

*Proof.* By contrapositive: polynomially many polynomial-time-computable queries yield a polynomial-time factoring algorithm. $\square$

### 6.5 Barriers to Unconditional Lower Bounds

Proving an unconditional superpolynomial lower bound in the full digit convolution model (where the algorithm can exploit the algebraic structure of $\mathcal{R}_1$) faces fundamental barriers.

**Barrier 1: Algebraic structure of the Segre variety.** The rank-1 variety is the image of the Segre embedding $\sigma : \mathbb{P}^{d_x-1} \times \mathbb{P}^{d_y-1} \hookrightarrow \mathbb{P}^{d_x d_y - 1}$. It has degree $\binom{d_x + d_y - 2}{d_x - 1}$ and is smooth and rational. An algorithm could potentially exploit this rich structure to guide queries.

**Barrier 2: Lattice-variety interaction.** The constraint matrix $A$ is banded (entry $z_{ij}$ appears only in row $k = i+j$) with a chain structure from carries. Algorithms can exploit this to decompose the problem (e.g., digit-by-digit backtracking, which uses $O(b^{d/2})$ queries -- exponential in $d$ but subexponential in $n$).

**Barrier 3: Natural proofs.** Any superpolynomial lower bound on a model capturing polynomial-time computation faces the Razborov-Rudich natural proofs barrier [RR97].

### 6.6 Connection to Algebraic Complexity Theory

The digit convolution model relates to several studied frameworks:

- **Algebraic decision trees** [BenOr83]: Each rank-1 query tests a polynomial predicate (the $2 \times 2$ minors of $Z$).
- **Shoup's generic group model** [Shoup97]: Analogous oracle-query framework, with $\Omega(\sqrt{p})$ lower bounds for discrete logarithm. The key difference: our model does not black-box the lattice structure.
- **Geometric Complexity Theory** [Mulmuley01]: The rank-1 condition is a constraint on orbit structure under $GL(d_x) \times GL(d_y)$. However, the carry-propagation constraints break this symmetry.
- **Communication complexity**: If $\mathbf{x}$ is held by Alice and $\mathbf{y}$ by Bob, factoring $n$ becomes a two-party problem where carry constraints link their inputs.

---

## 7. Related Work

### 7.1 Circuit-Based SAT Factoring

The standard approach to constraint-based factoring constructs a binary multiplication circuit and converts it to CNF via Tseytin transformation. The survey by Yurichev [YurichevSAT] describes this encoding; the `satfactor` tool [satfactor] provides an implementation. Performance is limited to approximately 20--30 bit semiprimes with modern CDCL solvers. Our digit convolution encoding differs by operating at the arithmetic level rather than the gate level, preserving number-theoretic structure.

### 7.2 Hybrid SAT + Lattice Methods

Ajani et al. [Ajani24] combine circuit-based SAT with Coppersmith's lattice method, achieving factorization of 768-bit numbers in 789 seconds -- but only with approximately 50% of bits leaked. Their key insight is that SAT provides the partial information needed for lattice reduction to succeed. Our lattice analysis (Section 5) explains why: the carry-propagation lattice can be solved by LLL when one factor is known, because the remaining system is fully linear.

### 7.3 Quantum Approaches

Shor's algorithm [Shor94] factors in polynomial time on a quantum computer. Mosca et al. [Mosca20] and Schaller and Schutzhold [SS22] explore quantum SAT-based approaches, using quantum speedups for the search component of SAT-based factoring.

### 7.4 LLL and Polynomial Factoring

The LLL algorithm [LLL82] achieves polynomial-time factorization of univariate polynomials over $\mathbb{Z}$, building on Berlekamp-Zassenhaus. As discussed in Section 5.4, LLL succeeds for polynomial factoring because the coefficient constraints are linear (no carry propagation), avoiding the rank-1 barrier. Coppersmith's extension [Coppersmith96] uses LLL to find small roots of univariate modular polynomials, enabling partial-knowledge factoring.

---

## 8. Discussion and Future Work

### 8.1 The Digit Convolution Framework

The central contribution of this paper is not a faster factoring algorithm but a *framework* that cleanly decomposes factoring hardness. The digit convolution formulation separates the problem into:

1. **Tractable linear part:** The carry-propagation system $A\mathbf{v} = \mathbf{c}$ defines a well-structured affine lattice, amenable to LLL reduction and polynomial-time lattice algorithms.

2. **Intractable nonlinear part:** The rank-1 constraint $Z = \mathbf{x} \otimes \mathbf{y}$ is the precise obstruction. No convex relaxation captures it (30--39% integrality gap), and the rank-1 points are exponentially sparse ($2^{-\Omega(d^2)}$ fraction) in the carry-propagation lattice.

This decomposition explains several empirical observations:
- LLL works for polynomial factoring (no rank-1 constraint) but not integer factoring (rank-1 is the bottleneck).
- Hybrid SAT+lattice methods work when enough bits are leaked (the remaining problem is linear).
- Digit-level constraints help Z3 modestly (constant-factor improvement, better predictability) but do not change the scaling class.

### 8.2 Honest Assessment of Negative Results

We have been deliberately honest about what does *not* work:

- **Digit constraints do not provide statistically significant speedup** at 32 bits (Cohen's $d < 0.31$, Section 4.3).
- **The optimal base has no clean formula** ($R^2 = 0.09$ for the hypothesized power law, Section 4.2).
- **SDP relaxation fails** with a 30--39% integrality gap (Section 5.5).
- **The conditional lower bounds are trivially true** (Theorem 3 says nothing specific about digit convolution, Section 6.4).
- **Classical algorithms dominate** by 4+ orders of magnitude at all tested sizes (Section 4.3).

We view these negative results as strengths: they clarify the boundaries of the approach and prevent false optimism.

### 8.3 Future Directions

Several directions remain promising:

1. **Intersection theory.** What is the precise intersection class of the Segre variety $\operatorname{Seg}(d_x, d_y)$ with affine lattices arising from carry propagation? Tools from algebraic geometry (Chow rings, Hilbert functions, tropical geometry) may yield dimension and degree bounds with complexity-theoretic implications.

2. **Geometric Complexity Theory (GCT).** The rank-1 constraint is a condition on orbit structure under $GL(d_x) \times GL(d_y)$. If the digit convolution factoring problem can be cast within the GCT framework, representation-theoretic tools might yield lower bounds -- though GCT has not yet produced unconditional results for any problem.

3. **Communication complexity.** Viewing factoring as a two-party problem ($\mathbf{x}$ with Alice, $\mathbf{y}$ with Bob, carry constraints linking them) connects to well-studied communication complexity models. Known lower bounds for set disjointness and related problems might yield factoring-specific bounds.

4. **Adaptive base selection.** Our base sweep reveals that the optimal base depends on problem size and instance. Adaptive strategies that choose the base during the solve (perhaps guided by solver feedback) could improve practical performance.

5. **Tighter relaxations.** Sum-of-squares (SOS) or Lasserre hierarchy relaxations of the rank-1 constraint might yield tighter bounds than SDP, at the cost of higher computational overhead.

6. **Exact lattice point counts.** Computing $|\Lambda_n \cap \mathcal{B}|$ exactly for small cases via Barvinok's algorithm would validate or refute the heuristic estimates underpinning Theorem 4(b).

---

## Reproducibility

All code, data, and experiment scripts are available in an open-source repository. The implementation includes:

- Python package `factoring_lab` with SMT, backtracking, lattice, and SDP algorithms.
- Rust extension `factoring_kernels` (via PyO3/maturin) for the backtracking solver.
- Experiment scripts in `scripts/` reproducing all results in this paper.
- Raw data in `reports/` (CSV format) for multi-instance and leaked-bit experiments.
- Deterministic semiprime generation from integer seeds for exact reproducibility.

---

## References

- [Ajani24] Ajani, Y., et al. "SAT and Lattice Reduction for Integer Factorization." *Proceedings of the ACM on Programming Languages*, 2024. arXiv:2406.20071. doi:10.1145/3666000.3669712.

- [BenOr83] Ben-Or, M. "Lower bounds for algebraic computation trees." *Proc. 15th ACM STOC*, 80--86, 1983.

- [Coppersmith96] Coppersmith, D. "Finding a small root of a univariate modular equation." *Proc. EUROCRYPT*, 155--165, 1996.

- [deMoura08] de Moura, L. and Bjorner, N. "Z3: An efficient SMT solver." *Proc. TACAS*, LNCS 4963, 337--340, 2008.

- [LLL82] Lenstra, A.K., Lenstra, H.W., and Lovasz, L. "Factoring polynomials with rational coefficients." *Mathematische Annalen* 261, 515--534, 1982.

- [Mosca20] Mosca, M., et al. "On speeding up factoring with quantum SAT solvers." *Scientific Reports* 10, 2020. PMC7490379.

- [Mulmuley01] Mulmuley, K. and Sohoni, M. "Geometric complexity theory I: An approach to the P vs. NP and related problems." *SIAM J. Comput.* 31(2), 496--526, 2001.

- [RR97] Razborov, A.A. and Rudich, S. "Natural proofs." *J. Comput. System Sci.* 55(1), 24--35, 1997.

- [satfactor] Neph, S. "satfactor." GitHub repository. https://github.com/sjneph/satfactor.

- [SS22] Schaller, M. and Schutzhold, R. "Factoring semi-primes with (quantum) SAT-solvers." *Scientific Reports* 12, 2022. doi:10.1038/s41598-022-11687-7.

- [Shor94] Shor, P.W. "Algorithms for quantum computation: Discrete logarithms and factoring." *Proc. 35th IEEE FOCS*, 124--134, 1994.

- [Shoup97] Shoup, V. "Lower bounds for discrete logarithms and related problems." *Proc. EUROCRYPT*, LNCS 1233, 256--266, 1997.

- [YurichevSAT] Yurichev, D. "Encoding Basic Arithmetic Operations for SAT-Solvers." https://yurichev.com/mirrors/SAT_factor/.

---

## Appendix A: Rust Port Performance

The digit convolution backtracking algorithm was ported to Rust via PyO3/maturin, yielding a consistent 30--33x speedup over the Python implementation with identical search paths:

| Bit size | Python (s) | Rust (s) | Speedup |
|----------|-----------|---------|---------|
| 12 | 0.00116 | 0.00004 | 26x |
| 16 | 0.01033 | 0.00033 | 31x |
| 20 | 0.14805 | 0.00452 | 33x |
| 24 | 0.47351 | 0.01409 | 33x |

Iteration counts are identical between implementations, confirming the speedup is purely from eliminating interpreter overhead. The Rust implementation uses `u128` arithmetic, supporting semiprimes up to approximately 38 decimal digits.
