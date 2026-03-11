# Reading List

## SAT/SMT Solvers
- [Satisfiability Modulo Theories (Wikipedia)](https://en.wikipedia.org/wiki/Satisfiability_modulo_theories) — what SMT is and why it matters
- [Programming Z3 (official tutorial)](https://theory.stanford.edu/~nikolaj/programmingz3.html) — free, excellent, covers bitvector theory (exactly what we use)
- [Conflict-Driven Clause Learning (Wikipedia)](https://en.wikipedia.org/wiki/Conflict-driven_clause_learning) — the "learn from failures" trick that makes modern SAT/SMT solvers fast

## Factoring as Constraint Satisfaction
- [Integer Factorization (Wikipedia)](https://en.wikipedia.org/wiki/Integer_factorization) — overview of where factoring sits in complexity
- [Encoding Integer Factorization as a SAT Problem (Heule, 2018)](https://arxiv.org/abs/1804.02313) — circuit-based SAT encoding of multiplication
- [Factoring with SAT solvers (Warren Smith notes)](https://www.rangevoting.org/FactSAT.html) — practical experiments with SAT-based factoring

## Digit Convolution Background
- [Long Multiplication (Wikipedia)](https://en.wikipedia.org/wiki/Multiplication_algorithm#Long_multiplication) — the operation our α_k convolution formalizes
- [Convolution (Wikipedia)](https://en.wikipedia.org/wiki/Convolution) — the general concept; factoring via digit convolution is deconvolution
- [Lattice Basis Reduction / LLL (Wikipedia)](https://en.wikipedia.org/wiki/Lattice_basis_reduction) — solved polynomial factoring in P; could something analogous work on digit constraints?

## Complexity Theory
- [BQP (Wikipedia)](https://en.wikipedia.org/wiki/BQP) — where quantum factoring (Shor) lives
- [Integer Factorization: Difficulty and Complexity (Wikipedia)](https://en.wikipedia.org/wiki/Integer_factorization#Difficulty_and_complexity) — the "limbo" status of factoring
- [Complexity Zoo](https://complexityzoo.net/Complexity_Zoo) — the full zoo of complexity classes

## Prime Gaps
- [Prime Gap (Wikipedia)](https://en.wikipedia.org/wiki/Prime_gap) — average gaps, extreme gaps, twin primes
- [Cramér's Conjecture (Wikipedia)](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_conjecture) — max gap ~ (ln N)²

## Classical Factoring Algorithms
- [Pollard's Rho Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm) — birthday paradox on residues
- [Pollard's p−1 Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Pollard%27s_p_%E2%88%92_1_algorithm) — exploits smoothness of p−1
- [Elliptic Curve Method (Wikipedia)](https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization) — next algorithm to implement
- [Quadratic Sieve (Wikipedia)](https://en.wikipedia.org/wiki/Quadratic_sieve) — congruence of squares
- [General Number Field Sieve (Wikipedia)](https://en.wikipedia.org/wiki/General_number_field_sieve) — state of the art for large integers
