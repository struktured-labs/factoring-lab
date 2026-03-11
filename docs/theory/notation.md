# Mathematical Notation

This document defines notation used throughout the theory documents in this project.

## Numbers and Factoring

| Symbol | Meaning |
|--------|---------|
| $n$ | The integer to be factored (a semiprime, $n = p \cdot q$) |
| $\beta$ | The number of bits of $n$, i.e., $\beta = \lceil \log_2 n \rceil$ |
| $p, q$ | The two prime factors of $n$, with $2 \le p \le q$ |
| $b$ | The radix (base) used for digit decomposition |
| $d$ | The number of base-$b$ digits of $n$, i.e., $d = \lceil \log_b n \rceil$ |

## Digit Representations

| Symbol | Meaning |
|--------|---------|
| $c_k$ | The $k$-th base-$b$ digit of $n$, so $n = \sum_{k=0}^{d-1} c_k \cdot b^k$ |
| $x_i$ | The $i$-th base-$b$ digit of $p$, so $p = \sum_{i=0}^{d_x - 1} x_i \cdot b^i$ |
| $y_j$ | The $j$-th base-$b$ digit of $q$, so $q = \sum_{j=0}^{d_y - 1} y_j \cdot b^j$ |
| $d_x$ | Number of base-$b$ digits of $p$ |
| $d_y$ | Number of base-$b$ digits of $q$ |
| $\mathbf{x}$ | The digit vector $(x_0, x_1, \ldots, x_{d_x - 1}) \in \{0, \ldots, b-1\}^{d_x}$ |
| $\mathbf{y}$ | The digit vector $(y_0, y_1, \ldots, y_{d_y - 1}) \in \{0, \ldots, b-1\}^{d_y}$ |

## Convolution and Carry Propagation

| Symbol | Meaning |
|--------|---------|
| $\alpha_k$ | The $k$-th digit convolution: $\alpha_k = \sum_{i+j=k} x_i \cdot y_j$ |
| $t_k$ | The carry at position $k$, defined by $\alpha_k + t_{k-1} = c_k + b \cdot t_k$ |
| $m_k$ | The partial sum at position $k$: $m_k = \alpha_k + t_{k-1}$ |

Convention: $t_{-1} = 0$ (no carry into the least significant position).

## Matrices and Rank

| Symbol | Meaning |
|--------|---------|
| $Z$ | The $d_x \times d_y$ product matrix with $Z_{ij} = x_i \cdot y_j$ |
| $z_{ij}$ | Entry $(i,j)$ of $Z$, used as a linearized variable |
| $\mathrm{rank}(Z)$ | The matrix rank of $Z$ |
| $\mathbf{x} \otimes \mathbf{y}$ | The outer product $\mathbf{x} \mathbf{y}^T$, so $Z = \mathbf{x} \otimes \mathbf{y}$ iff $\mathrm{rank}(Z) = 1$ |

## Lattice and Constraint System

| Symbol | Meaning |
|--------|---------|
| $A$ | The constraint matrix encoding carry-propagation: $A \mathbf{v} = \mathbf{c}$ |
| $\mathbf{v}$ | The variable vector $\mathbf{v} = (z_{00}, z_{01}, \ldots, t_0, t_1, \ldots)$ |
| $\mathbf{c}$ | The digit vector $(c_0, c_1, \ldots, c_{d-1})$ of $n$ |
| $\Lambda$ | A lattice, typically $\Lambda = \{ \mathbf{v} \in \mathbb{Z}^m : A\mathbf{v} = \mathbf{c} \}$ |
| $\mathcal{R}_1$ | The rank-1 variety: $\mathcal{R}_1 = \{ Z \in \mathbb{R}^{d_x \times d_y} : \mathrm{rank}(Z) = 1 \}$ |

## Algebraic Geometry

| Symbol | Meaning |
|--------|---------|
| $\mathbb{P}^{n}$ | Projective space of dimension $n$ |
| $\sigma$ | The Segre embedding $\sigma : \mathbb{P}^{d_x - 1} \times \mathbb{P}^{d_y - 1} \hookrightarrow \mathbb{P}^{d_x d_y - 1}$ |
| $\mathrm{Seg}(d_x, d_y)$ | The Segre variety, image of $\sigma$; parametrizes rank-1 matrices |
| $\dim(V)$ | The algebraic dimension of a variety $V$ |
| $\deg(V)$ | The degree of a projective variety $V$ |

## Complexity and Computation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{O}(\cdot)$ | Big-O (asymptotic upper bound) |
| $\Omega(\cdot)$ | Big-Omega (asymptotic lower bound) |
| $\Theta(\cdot)$ | Big-Theta (asymptotic tight bound) |
| $\mathrm{poly}(\cdot)$ | Some fixed polynomial |
| $\mathrm{P}$ | The complexity class of polynomial-time decidable languages |
| $\mathrm{NP}$ | The complexity class of nondeterministic polynomial-time decidable languages |
| $\mathrm{BPP}$ | Bounded-error probabilistic polynomial time |

## Oracle and Query Model

| Symbol | Meaning |
|--------|---------|
| $\mathcal{O}_{\mathrm{R1}}$ | The rank-1 oracle |
| $Q(n)$ | The number of rank-1 queries used by an algorithm to factor $n$ |
| $\mathcal{S}$ | The candidate solution space (set of feasible $Z$ matrices) |
| $\mathcal{S}_i$ | The candidate space after $i$ oracle queries |
| $\|\mathcal{S}\|$ | The cardinality (or measure) of a candidate space |
