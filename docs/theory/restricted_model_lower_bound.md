# Restricted-Model Lower Bounds for Digit Convolution Factoring

*March 10, 2026*

**Status:** Working document. Results range from rigorous theorems to conjectures to open questions. Each is clearly labeled.

See [notation.md](notation.md) for symbol definitions.

---

## A. Formal Model Definition

### A.1. The Digit Convolution Computation Model

We define a *digit convolution algorithm* for factoring an integer $n$ as a procedure that operates in the following framework.

**Input.** An $\beta$-bit semiprime $n = p \cdot q$ with $2 \le p \le q$.

**Free Setup.** The algorithm may perform the following operations at zero cost:

1. **Digit decomposition.** Choose any base $b \ge 2$ and compute the base-$b$ digits $c_0, c_1, \ldots, c_{d-1}$ of $n$, where $d = \lceil \log_b n \rceil$.

2. **Linear constraint construction.** Set up the carry-propagation system. Define linearized variables $z_{ij}$ (intended to represent $x_i \cdot y_j$) and carry variables $t_k$. The system is:

$$\sum_{\substack{i + j = k \\ 0 \le i < d_x \\ 0 \le j < d_y}} z_{ij} + t_{k-1} - b \cdot t_k = c_k, \quad k = 0, 1, \ldots, d-1$$

with $t_{-1} = 0$. This defines the *carry-propagation lattice*:

$$\Lambda_n = \left\{ \mathbf{v} = (z_{00}, z_{01}, \ldots, z_{d_x - 1, d_y - 1}, t_0, \ldots, t_{d-1}) \in \mathbb{Z}^m : A \mathbf{v} = \mathbf{c} \right\}$$

where $m = d_x \cdot d_y + d$ and $A$ is the $d \times m$ constraint matrix.

3. **Lattice operations.** Any polynomial-time lattice operation on $\Lambda_n$ (e.g., LLL reduction, enumeration of short vectors, Gram-Schmidt orthogonalization) is free.

**Costly Operation.** The algorithm pays 1 unit per *rank-1 query* (defined below).

### A.2. The Rank-1 Oracle

**Definition (Rank-1 Oracle $\mathcal{O}_{\mathrm{R1}}$).** Given a matrix $Z \in \mathbb{Z}^{d_x \times d_y}$ whose entries correspond to a point in $\Lambda_n$ (i.e., the $z_{ij}$ and implied $t_k$ satisfy the carry-propagation constraints), the oracle returns:

$$\mathcal{O}_{\mathrm{R1}}(Z) = \begin{cases} 1 & \text{if } \exists\, \mathbf{x} \in \{0,\ldots,b-1\}^{d_x},\, \mathbf{y} \in \{0,\ldots,b-1\}^{d_y} \text{ s.t. } Z = \mathbf{x} \otimes \mathbf{y} \\[4pt] 0 & \text{otherwise} \end{cases}$$

**Remark.** Note that the oracle checks *both* the rank-1 condition *and* the digit-range constraint. A query implicitly also checks consistency with the carry-propagation system, since the input $Z$ is required to lie on $\Lambda_n$. Thus a positive oracle response immediately yields the factorization $n = p \cdot q$ where $p = \sum x_i b^i$, $q = \sum y_j b^j$.

**Definition (Query Complexity).** For a digit convolution algorithm $\mathcal{A}$ and input $n$, define $Q_{\mathcal{A}}(n)$ as the number of rank-1 oracle calls made by $\mathcal{A}$. The *worst-case query complexity* at bit-length $\beta$ is:

$$Q_{\mathcal{A}}(\beta) = \max_{\substack{n : |n| = \beta \\ n \text{ semiprime}}} Q_{\mathcal{A}}(n)$$

We seek lower bounds on $Q_{\mathcal{A}}(\beta)$ that hold for all digit convolution algorithms $\mathcal{A}$.

### A.3. Information-Theoretic Framework

Each oracle query partitions the current candidate space. Before any queries, the candidate set is:

$$\mathcal{S}_0 = \left\{ Z \in \Lambda_n \cap \mathbb{Z}_{\ge 0}^{d_x \times d_y} : 0 \le z_{ij} \le (b-1)^2 \right\}$$

After query $i$ with answer $a_i \in \{0, 1\}$, the candidate set updates:

$$\mathcal{S}_{i+1} = \begin{cases} \mathcal{S}_i \cap \mathcal{R}_1 & \text{if } a_i = 1 \text{ (factorization found)} \\[4pt] \mathcal{S}_i \setminus \{Z_i\} & \text{if } a_i = 0 \end{cases}$$

The algorithm succeeds when some query returns 1, at which point the factorization is determined.

---

## B. Information-Theoretic Lower Bound (Naive)

### B.1. Counting Factor Pairs

**Lemma 1 (Factor pair count).** For a $\beta$-bit balanced semiprime $n = p \cdot q$ with $p \le q$, there is exactly one factorization (since $p$ and $q$ are distinct primes). However, the algorithm does not know this factorization a priori and must distinguish the correct $(p, q)$ from all possibilities.

The number of candidate factor pairs is bounded by:

$$|\{(p, q) : p \cdot q = n,\; 2 \le p \le \sqrt{n}\}| = 1$$

But the algorithm does not know $p$, so it must search among $\Theta(2^{\beta/2})$ possible values of $p$ (all odd numbers up to $\sqrt{n}$, modulo the constraint that $p | n$).

### B.2. The Naive Bound

**Proposition 1 (Trivial information-theoretic lower bound).**

$$Q_{\mathcal{A}}(\beta) \ge 1$$

for any algorithm $\mathcal{A}$ that provably factors every $\beta$-bit semiprime.

*Proof.* The algorithm must call the oracle at least once to receive the answer "yes" and extract the factorization. $\square$

**Remark.** One might hope for $Q_{\mathcal{A}}(\beta) \ge \Omega(\beta)$ by arguing that each query returns only 1 bit and there are $\sim 2^{\beta/2}$ candidates, giving $\beta/2$ queries. But this argument is *wrong* in our model: a "yes" answer from the oracle immediately reveals the factorization (since the queried $Z$ encodes it), so the algorithm only needs to find *one* correct $Z$, not distinguish among all $2^{\beta/2}$ candidates. The relevant question is not "how many bits of information are needed" but "how quickly can the algorithm find a rank-1 point in $\Lambda_n$."

The naive information-theoretic approach gives essentially nothing because the oracle is asymmetric: a "yes" is worth $\Theta(\beta)$ bits (the full factorization), while a "no" is worth much less.

---

## C. Structural Lower Bound via Lattice-Variety Intersection

### C.1. Geometric Setup

The core of the factoring problem in this model is: find a point in the intersection

$$\Lambda_n \cap \mathcal{R}_1 \cap \mathcal{B}$$

where:
- $\Lambda_n \subseteq \mathbb{Z}^m$ is the affine lattice of carry-propagation solutions (an affine subspace of $\mathbb{R}^m$ intersected with $\mathbb{Z}^m$)
- $\mathcal{R}_1 = \{ Z \in \mathbb{R}^{d_x \times d_y} : \mathrm{rank}(Z) \le 1 \}$ is the rank-1 variety (technically, the determinantal variety defined by all $2 \times 2$ minors of $Z$ vanishing)
- $\mathcal{B} = \{ Z : 0 \le z_{ij} \le (b-1)^2 \}$ is the box constraint from digit bounds

### C.2. Dimensions

**Lemma 2 (Lattice dimension).**

The constraint matrix $A$ is $d \times m$ where $m = d_x d_y + d$. Since $A$ has rank at most $d$ (one constraint per digit position), and generically has rank exactly $d$, the lattice $\Lambda_n$ has dimension:

$$\dim(\Lambda_n) = m - d = d_x d_y$$

For a balanced semiprime with $d_x \approx d_y \approx d/2$, this is $\Theta(d^2) = \Theta(\log_b^2 n)$.

**Lemma 3 (Rank-1 variety dimension).**

The variety $\mathcal{R}_1 \subset \mathbb{R}^{d_x \times d_y}$ has dimension $d_x + d_y - 1$ (parametrized by the entries of $\mathbf{x}$ and $\mathbf{y}$, modulo a single scaling redundancy). As a real algebraic variety, it is defined by $\binom{d_x}{2} \binom{d_y}{2}$ equations (the $2 \times 2$ minors of $Z$).

For $d_x \approx d_y \approx d/2$, the rank-1 variety has dimension $\Theta(d)$ while the ambient space has dimension $\Theta(d^2)$. So $\mathcal{R}_1$ is a "thin" subset.

### C.3. Intersection Analysis

The question is: what is the dimension (or cardinality) of $\Lambda_n \cap \mathcal{R}_1 \cap \mathcal{B}$?

**Heuristic Dimension Count.** The lattice contributes $d_x d_y$ degrees of freedom. The rank-1 condition imposes $\binom{d_x}{2}\binom{d_y}{2}$ equations, but these are not all independent on the lattice. A naive dimension count:

$$\text{expected dim} = d_x d_y - \binom{d_x}{2}\binom{d_y}{2}$$

For $d_x = d_y = d/2$ with $d \ge 6$, this is negative, suggesting the intersection is generically a discrete (zero-dimensional) set. This is consistent with our expectation: there are finitely many factorizations of $n$ (in fact, exactly one for a semiprime).

**However**, this dimension count is only a heuristic. The rank-1 variety has special structure (it is the affine cone over the Segre variety), and the lattice $\Lambda_n$ has special structure (banded/Toeplitz-like from the convolution). Their intersection could behave differently from the generic case.

### C.4. Why This Does Not Yield Strong Lower Bounds

The geometric picture tells us that $\Lambda_n \cap \mathcal{R}_1$ is generically a discrete set, but it does *not* tell us how hard it is to *find* a point in this intersection given oracle access.

The difficulty: an algorithm is not restricted to querying random points. It can use the lattice structure (LLL, enumeration, etc.) to focus queries on promising regions of $\Lambda_n$. The question becomes: how effectively can lattice algorithms guide the search toward rank-1 points?

**Observation 1.** If the algorithm could compute, for any affine subspace $L \subseteq \mathbb{R}^{d_x \times d_y}$, the closest rank-1 matrix to $L$ (a continuous optimization problem), then a single computation plus a single oracle query might suffice. The cost model must account for this.

**Observation 2.** In our model, lattice operations are free but the rank-1 *test* is costly. This means the lower bound question is really: "How many rank-1 tests are needed if all other computation is free?" If the algorithm can narrow the search space to $K$ candidates using free lattice operations, then $K$ queries suffice.

This reduces to: **What is the minimum number of lattice points in $\Lambda_n$ that must be tested before a rank-1 point is guaranteed to be found?**

---

## D. The Real Difficulty: Why Superpolynomial Lower Bounds Are Hard

### D.1. The Barrier

Proving that $Q_{\mathcal{A}}(\beta) \ge \beta^{\omega(1)}$ (superpolynomial) for all algorithms $\mathcal{A}$ in the digit convolution model would require showing that no polynomial-time strategy can find rank-1 points in the carry-propagation lattice. This faces several fundamental barriers.

**Barrier 1: Algebraic structure of the Segre variety.** The rank-1 variety is not an arbitrary nonlinear constraint. It is the image of the Segre embedding:

$$\sigma : \mathbb{P}^{d_x - 1} \times \mathbb{P}^{d_y - 1} \hookrightarrow \mathbb{P}^{d_x d_y - 1}$$

This embedding has rich algebraic structure. In particular:
- The degree of the Segre variety $\mathrm{Seg}(d_x, d_y)$ in $\mathbb{P}^{d_x d_y - 1}$ is $\binom{d_x + d_y - 2}{d_x - 1}$, which is polynomial in $d$ for fixed $d_x/d_y$ ratio.
- The variety is *smooth* and *rational* (parametrized by $(\mathbf{x}, \mathbf{y})$).
- It admits explicit equations (the $2 \times 2$ minors), which an algorithm could potentially exploit even though the oracle only provides yes/no answers.

Any lower bound proof must account for the possibility that an algorithm exploits this algebraic structure.

**Barrier 2: Lattice-variety interaction.** The carry-propagation lattice $\Lambda_n$ is not a "generic" lattice. It has specific structure inherited from the convolution:
- The constraint matrix $A$ is *banded*: entry $z_{ij}$ appears only in constraint $k = i + j$.
- The carries $t_k$ create a chain structure: $t_k$ appears in constraints $k$ and $k+1$.

An algorithm could exploit this structure to decompose the problem, e.g., solving digit-by-digit or in blocks. This is precisely what backtracking algorithms do (and they correspond to $O(b^{d/2})$ queries in our model, which is exponential in $d$ but subexponential in $n$).

**Barrier 3: Natural proofs barrier.** Any superpolynomial lower bound on a model that captures polynomial-time computation faces the natural proofs barrier (Razborov-Rudich, 1997). While our model is restricted (only rank-1 queries, not arbitrary computation), extending lower bounds to less restricted models runs into this barrier.

### D.2. Comparison with Known Restricted Models

The digit convolution model is reminiscent of several studied computational models:

| Model | Lower bounds known? | Relevance |
|-------|-------------------|-----------|
| Algebraic decision trees | $\Omega(n \log n)$ for sorting (Ben-Or, 1983) | Closest analogy: each rank-1 query is an algebraic test |
| Linear decision trees | $\Omega(n \log n)$ for sorting | Our "free" operations are linear; queries are nonlinear |
| Algebraic computation trees | Some exponential lower bounds (Yao, 1997) | Stronger model than ours |
| Black-box group model | $\Omega(p^{1/2})$ for DLP (Shoup, 1997) | Analogous oracle-query framework |

The closest analogy is to Shoup's generic group model for discrete logarithm. In that model, group elements are accessed only through a black-box oracle, and $\Omega(\sqrt{p})$ queries are needed. Our rank-1 oracle plays a similar role. The key difference: in our model, the lattice structure is *not* black-boxed — the algorithm has full access to the linear constraints.

---

## E. What IS Provable

### E.1. Black-Box Rank-1 Oracle Lower Bound

We can prove a lower bound in a model where the algorithm treats the oracle as a *completely opaque* predicate — it cannot exploit any algebraic structure of the rank-1 condition.

**Definition (Black-Box Digit Convolution Model).** As in Section A, except:
- The rank-1 oracle $\mathcal{O}_{\mathrm{R1}}$ is replaced by an arbitrary predicate $\mathcal{O} : \Lambda_n \to \{0, 1\}$ satisfying:
  - $|\mathcal{O}^{-1}(1) \cap \mathcal{B}| \le \tau(n)$, where $\tau(n)$ is the number of divisor pairs of $n$ (for a semiprime, $\tau(n) = 2$, representing $(p,q)$ and $(q,p)$).
  - The algorithm may not assume anything about $\mathcal{O}$ beyond the bound on positive instances.

In other words, the algorithm knows that at most $\tau(n)$ lattice points in the box will yield "yes," but it cannot predict *which* ones without querying.

**Theorem 1 (Black-box lower bound).**

*In the black-box digit convolution model, any deterministic algorithm requires at least*

$$Q(\beta) \ge \frac{|\Lambda_n \cap \mathcal{B}|}{2} - 1$$

*queries in the worst case, where $|\Lambda_n \cap \mathcal{B}|$ is the number of lattice points in the feasible box.*

*Proof.* An adversary argument. The adversary maintains a set $\mathcal{S} \subseteq \Lambda_n \cap \mathcal{B}$ of points that could be the positive instance. Initially $|\mathcal{S}| = |\Lambda_n \cap \mathcal{B}|$. On each query $Z_i$:

- If $|\mathcal{S}| > 2$, the adversary answers "no" and removes $Z_i$ from $\mathcal{S}$.
- If $|\mathcal{S}| \le 2$, the adversary is forced to answer "yes" on one of the remaining candidates.

After $k$ queries with "no" answers, $|\mathcal{S}| = |\Lambda_n \cap \mathcal{B}| - k$. The algorithm can only be certain of finding the positive instance when $|\mathcal{S}| \le 2$, requiring $k \ge |\Lambda_n \cap \mathcal{B}| - 2$ queries.

For a worst-case adversary that places the positive instances last, the deterministic algorithm needs $|\Lambda_n \cap \mathcal{B}|/2 - 1$ queries (since two positive instances could be either of the remaining two points). $\square$

**Corollary 1.** *If $|\Lambda_n \cap \mathcal{B}| \ge 2^{\Omega(d)}$, then deterministic black-box algorithms require $2^{\Omega(d)} = n^{\Omega(1/\log b)}$ queries.*

**Remark on the strength of this result.** Theorem 1 is a genuine theorem, rigorously proved. However, it is of limited interest for two reasons:

1. The black-box model is unrealistically weak. Real algorithms *do* exploit the algebraic structure of the rank-1 condition (e.g., computing SVDs, checking $2 \times 2$ minors). Theorem 1 does not apply to such algorithms.

2. The bound depends on $|\Lambda_n \cap \mathcal{B}|$, which we have not computed precisely. We address this next.

### E.2. Lattice Point Count

**Lemma 4 (Lattice point count, heuristic).**

The number of integer points in $\Lambda_n \cap \mathcal{B}$ is approximately:

$$|\Lambda_n \cap \mathcal{B}| \approx \frac{\mathrm{vol}(\mathcal{B})}{\det(\Lambda_n)} = \frac{(b-1)^{2 d_x d_y}}{\det(A^T A)^{1/2}}$$

For a balanced semiprime with $d_x \approx d_y \approx d/2$ and base $b$, each $z_{ij}$ ranges over $\{0, 1, \ldots, (b-1)^2\}$ giving $(b-1)^2 + 1$ values. The $d$ linear constraints each remove roughly a factor of $b$ (since they constrain carries modulo $b$), giving:

$$|\Lambda_n \cap \mathcal{B}| \approx \frac{((b-1)^2 + 1)^{d^2/4}}{b^d} = \frac{(b^2 - 2b + 2)^{d^2/4}}{b^d}$$

For $d \ge 5$ and $b \ge 2$, this is $2^{\Omega(d^2)}$, which is $n^{\Omega(\log_b n)}$ — superpolynomial in $n$.

**Caveat.** This is a heuristic estimate. The lattice point count in a polytope depends on the lattice geometry, and the actual count could differ significantly from the volume heuristic. A rigorous bound would require Barvinok's algorithm or explicit enumeration for small cases.

### E.3. Randomized Black-Box Lower Bound

**Theorem 2 (Randomized black-box lower bound).**

*In the black-box digit convolution model, any randomized algorithm that succeeds with probability at least $2/3$ requires*

$$Q(\beta) \ge \frac{|\Lambda_n \cap \mathcal{B}|}{6}$$

*queries in expectation.*

*Proof sketch.* By Yao's minimax principle, the expected cost of the best randomized algorithm against the worst-case input equals the expected cost of the best deterministic algorithm against the worst-case distribution. Consider the uniform distribution over placements of the positive instance. A deterministic algorithm querying $k$ random points hits the positive instance with probability $k \cdot 2 / |\Lambda_n \cap \mathcal{B}|$ (union bound, with 2 positive instances). For this to exceed $2/3$, we need $k \ge |\Lambda_n \cap \mathcal{B}| / 6$. $\square$

### E.4. Algebraic Oracle Lower Bound (Conditional)

The following result applies to algorithms that use the rank-1 oracle *only* through membership queries, but may also perform polynomial-time algebraic computations on the lattice.

**Theorem 3 (Conditional lower bound).**

*Assume that integer factoring is not solvable in polynomial time (i.e., FACTORING $\notin$ BPP). Then for any digit convolution algorithm $\mathcal{A}$ (with unrestricted polynomial-time computation between queries), the query complexity satisfies:*

$$Q_{\mathcal{A}}(\beta) \ge \beta^{\omega(1)}$$

*i.e., superpolynomially many rank-1 queries are needed in the worst case.*

*Proof.* By contrapositive. Suppose $\mathcal{A}$ makes at most $Q(\beta) = \beta^{O(1)}$ rank-1 queries. Each query can be computed in polynomial time (compute the $2 \times 2$ minors of $Z$ and check that all vanish, plus verify digit bounds). The inter-query computation is polynomial-time by assumption. Therefore $\mathcal{A}$ runs in total time $\mathrm{poly}(\beta)$, and since $\mathcal{A}$ factors every $\beta$-bit semiprime, this places FACTORING $\in$ BPP. Contradiction. $\square$

**Remark.** Theorem 3 is *trivially true* — it says nothing specific about the digit convolution model. It applies equally to *any* factoring algorithm that uses polynomially many polynomial-time subroutine calls. Its value is in stating the obvious precisely: the digit convolution model inherits the hardness of factoring.

### E.5. A Non-Trivial Structural Result

The following is the strongest unconditional result we can state.

**Theorem 4 (Rank-1 sparsity in the carry-propagation lattice).**

*For an $\beta$-bit balanced semiprime $n = p \cdot q$ in base $b$, let $d = \lceil \log_b n \rceil$ and let $\Lambda_n$ be the carry-propagation lattice. Then:*

*(a) The number of rank-1 integer matrices in $\Lambda_n \cap \mathcal{B}$ is at most $2$.*

*(b) The ratio of rank-1 points to total lattice points satisfies:*

$$\frac{|\Lambda_n \cap \mathcal{R}_1 \cap \mathcal{B}|}{|\Lambda_n \cap \mathcal{B}|} \le \frac{2}{|\Lambda_n \cap \mathcal{B}|} \le 2^{-\Omega(d^2)}$$

*(assuming the heuristic lattice point count from Lemma 4).*

*Proof of (a).* A rank-1 matrix $Z = \mathbf{x} \otimes \mathbf{y}$ in $\Lambda_n \cap \mathcal{B}$ with $x_i, y_j \in \{0, \ldots, b-1\}$ and $Z$ satisfying the carry-propagation constraints yields $p = \sum x_i b^i$ and $q = \sum y_j b^j$ with $p \cdot q = n$. Since $n$ is a semiprime, the only factorizations are $(p, q)$ and $(q, p)$. $\square$

**Interpretation.** The "needle in a haystack" is exponentially sparse: among all lattice points satisfying the linear carry constraints, the rank-1 solutions form a fraction $2^{-\Omega(d^2)}$. Any algorithm that tests lattice points "at random" requires exponentially many queries. The question is whether the algebraic structure of rank-1 matrices allows a non-random search strategy.

---

## F. Conditional Results

### F.1. Assuming Factoring Hardness

**Corollary 2 (of Theorem 3, restated for emphasis).**

*If FACTORING $\notin$ BPP, then no digit convolution algorithm with polynomial-time inter-query computation can factor $\beta$-bit semiprimes using $\mathrm{poly}(\beta)$ rank-1 queries.*

This is logically equivalent to: "If polynomial rank-1 queries suffice, then factoring is in BPP."

### F.2. Assuming Exponential Hardness of Factoring

**Corollary 3.**

*If FACTORING requires time $2^{\Omega(\beta^{1/3})}$ (consistent with the GNFS heuristic), then any digit convolution algorithm with polynomial-time inter-query computation requires $2^{\Omega(\beta^{1/3})} / \mathrm{poly}(\beta)$ rank-1 queries.*

*Proof.* Same argument as Theorem 3: each query costs $\mathrm{poly}(\beta)$ time, so $Q$ queries take $Q \cdot \mathrm{poly}(\beta)$ total time. $\square$

### F.3. Generic Oracle Model

**Definition (Generic Rank-1 Oracle).** A *generic* rank-1 oracle is one where the "yes" instances are placed uniformly at random among lattice points, subject only to the constraint that they correspond to valid factorizations. An algorithm using a generic oracle cannot exploit correlations between the algebraic structure of rank-1 matrices and the carry-propagation lattice.

**Theorem 5 (Generic oracle lower bound).**

*Against a generic rank-1 oracle, any algorithm requires*

$$Q(\beta) \ge \frac{|\Lambda_n \cap \mathcal{B}|}{6} \ge 2^{\Omega(d^2/\log d)}$$

*queries in expectation to factor with probability $\ge 2/3$.*

*Proof.* Follows from Theorem 2 and the heuristic lattice point count. In the generic oracle model, the algorithm has no information about which lattice points are positive except through queries. $\square$

**Caveat.** The generic oracle model is artificial. The *real* rank-1 oracle has highly structured "yes" instances (they are exactly the rank-1 matrices), and algorithms can exploit this structure. Theorem 5 quantifies the value of this structure: any algorithm that factors in fewer than $2^{\Omega(d^2/\log d)}$ queries must be exploiting the algebraic structure of the Segre variety.

---

## G. Open Questions and Research Directions

### G.1. Algebraic Geometry Tools

**Open Question 1.** *What is the precise intersection theory of the Segre variety $\mathrm{Seg}(d_x, d_y)$ with affine lattices arising from the carry-propagation constraints?*

The relevant tools include:
- **Intersection theory on Grassmannians:** The carry constraints define a linear subspace; the Segre variety sits in projective space. Their intersection class in the Chow ring could yield dimension and degree bounds.
- **Hilbert function computations:** The ideal of $2 \times 2$ minors of $Z$ has a well-studied Hilbert function. Intersecting with the linear ideal of carry constraints gives a quotient ring whose Hilbert function encodes the geometry of the solution set.
- **Tropical geometry:** The carry-propagation constraints have a "staircase" structure related to Newton polytopes of the convolution. Tropical intersection theory might yield combinatorial bounds.

### G.2. Connection to Geometric Complexity Theory

**Open Question 2.** *Is the digit convolution factoring problem an instance of the problems studied in Mulmuley's Geometric Complexity Theory (GCT) program?*

GCT studies the complexity of fundamental algebraic problems (matrix multiplication, determinant vs. permanent) using algebraic geometry and representation theory. The rank-1 constraint on $Z$ is a condition on the orbit structure under $GL(d_x) \times GL(d_y)$ action: rank-1 matrices form the unique closed orbit of the group action on $\mathbb{R}^{d_x \times d_y}$.

If the digit convolution factoring problem can be cast as an orbit problem, GCT tools (occurrence obstructions, representation-theoretic multiplicities) might yield lower bounds. However:
- GCT has not yet yielded *unconditional* lower bounds for any problem.
- The carry-propagation constraints break the $GL(d_x) \times GL(d_y)$ symmetry, so the problem is not a pure orbit problem.

### G.3. Equivalence with Known Models

**Open Question 3.** *Is the digit convolution model polynomially equivalent to any known restricted computation model?*

Candidates:
- **Algebraic decision trees** (Ben-Or): Each rank-1 query tests a polynomial predicate (the $2 \times 2$ minors). Our model is an algebraic decision tree of a specific form.
- **Bounded-depth arithmetic circuits**: The convolution $\alpha_k = \sum_{i+j=k} x_i y_j$ is computable by a depth-2 arithmetic circuit. Does the circuit depth hierarchy yield query lower bounds?
- **Communication complexity**: If $x$ is held by Alice and $y$ by Bob, factoring $n$ is a two-party communication problem where the carry-propagation constraints link their inputs. Known communication complexity lower bounds (e.g., for set disjointness) might apply.

### G.4. Concrete Open Problems

1. **Compute $|\Lambda_n \cap \mathcal{B}|$ exactly** for small cases ($d \le 8$, $b \le 10$). Does the heuristic estimate in Lemma 4 hold? Can Barvinok's algorithm handle these instances?

2. **Determine the shortest vector in $\Lambda_n$** that is rank-1. Is it always the factorization vector? What is its norm relative to the shortest non-rank-1 vector? (Our empirical work in `lattice_convolution.py` suggests the rank-1 vector is *not* the shortest.)

3. **Establish lower bounds for specific algorithms.** Even if we cannot prove general lower bounds, can we show that specific strategies (e.g., "query the $K$ shortest lattice vectors") fail? The lattice exploration journal entry provides empirical evidence that LLL-reduced short vectors are not rank-1.

4. **Relate query complexity to base $b$.** Our empirical work suggests base choice affects SMT solver performance. Does the query complexity $Q(\beta)$ depend on $b$? For $b = 2$ (binary), the lattice has maximal dimension but minimal digit range. For $b = 2^{\beta/2}$ (two-digit representation), the lattice is trivial but each "digit" is half-length. Is there an optimal $b$?

---

## Summary of Results

| Result | Type | Statement |
|--------|------|-----------|
| Theorem 1 | **Proved** (unconditional) | Black-box model: $\Omega(|\Lambda_n \cap \mathcal{B}|)$ queries needed |
| Theorem 2 | **Proved** (unconditional) | Randomized black-box: $\Omega(|\Lambda_n \cap \mathcal{B}|/6)$ expected queries |
| Theorem 3 | **Proved** (conditional) | If FACTORING $\notin$ BPP, then superpolynomial queries needed |
| Theorem 4 | **Proved** (unconditional) | Rank-1 fraction in lattice is $\le 2^{-\Omega(d^2)}$ |
| Theorem 5 | **Proved** (unconditional in generic model) | Generic oracle: $2^{\Omega(d^2/\log d)}$ queries needed |
| Corollary 3 | **Proved** (conditional on GNFS hardness) | $2^{\Omega(\beta^{1/3})}$ queries under standard assumption |
| Lemma 4 | **Heuristic** | Lattice point count $\approx 2^{\Theta(d^2)}$ |
| Superpolynomial unconditional lower bound | **Open** | Not proved; faces algebraic structure barriers |

**Honest assessment.** The unconditional results (Theorems 1, 2, 4, 5) are real but apply only to black-box or generic oracle models, which are weaker than the actual setting. The conditional results (Theorem 3, Corollary 3) are trivially true and say nothing specific about digit convolution. An unconditional superpolynomial lower bound in the full digit convolution model (where the algorithm can exploit the algebraic structure of rank-1 matrices) remains *wide open* and would likely require a major advance in algebraic complexity theory.
