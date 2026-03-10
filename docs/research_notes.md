# Factoring Research Notes

## 1. Polynomial vs Integer Factorization

**Polynomial factorization over $\mathbb{Z}[x]$ is in P.** Proven by Lenstra-Lenstra-Lovasz (LLL algorithm, 1982), building on Berlekamp-Zassenhaus. The LLL lattice basis reduction algorithm provides polynomial-time factorization of univariate polynomials over $\mathbb{Z}$.

**Integer factorization complexity is unknown.** Not known to be in P, not known to be NP-complete. Best classical algorithm (GNFS) runs in $L_n[1/3, c]$ — sub-exponential but super-polynomial.

**ChatGPT's flawed "proof by contradiction":** Attempted to show polynomial factoring is in P via contradiction, but the argument was circular — it assumed the tractability it aimed to prove. The conclusion happens to be correct (by LLL), but the reasoning was invalid.

**Open structural question:** What property of $\mathbb{Z}[x]$ makes factoring tractable that $\mathbb{Z}$ lacks?
- Polynomials have degree as a natural complexity measure; degree drops strictly under factorization.
- Polynomials admit reduction modulo primes $p$, enabling lift-based strategies (Hensel lifting).
- Lattice methods (LLL) exploit the linear-algebraic structure of polynomial coefficient spaces.
- Integers lack analogous "dimension reduction" — factoring $n$ doesn't obviously decompose into smaller structured subproblems.

## 2. Empirical Factoring Boundaries (Trial Division)

From ChatGPT experiments:

| Number | Bits | Result | Time |
|--------|------|--------|------|
| $2383737833$ | ~31 | $5741 \times 415213$ | instant |
| $373833388272710007$ | ~59 | $3^2 \times 83 \times 4777789 \times 104744329$ | instant |
| $1127373837281627281110474874743227221029$ | ~130 | timeout | $> 60$s |

This aligns with trial division's $O(\sqrt{n})$ complexity:
- $\sqrt{2^{60}} = 2^{30} \approx 10^9$ — feasible in seconds.
- $\sqrt{2^{130}} = 2^{65} \approx 3.7 \times 10^{19}$ — infeasible by enumeration.

**Practical threshold for trial division: ~60-70 bits.** Beyond this, need Pollard's rho, ECM, quadratic sieve, or GNFS.

## 3. Goldbach-Related Question

**Conjecture (variant):** For every prime $p > 2$, there exist primes $q, r$ such that $p = q + r$ or $p = q + r + 1$.

- For even $p$: not applicable (only $p = 2$).
- For odd primes $p$: $p - 1$ is even, so by Goldbach's conjecture (verified up to $4 \times 10^{18}$), $p - 1 = q + r$ for primes $q, r$, giving $p = q + r + 1$.
- Alternatively, if $p - 2$ is prime, then $p = (p-2) + 2 = q + r$.

This is essentially a corollary of Goldbach's conjecture. No counterexample is known. ChatGPT could not find one but also could not execute verification code.

**Connection to factoring:** Additive decompositions of primes relate to the distribution of primes in arithmetic progressions, which connects to sieve methods also used in factoring algorithms (e.g., quadratic sieve).

## 4. Prime Gap Statistics

Let $g_n = p_{n+1} - p_n$ denote the $n$-th prime gap.

**Average gap:** By the prime number theorem, the average gap near $N$ is $\sim \ln N$.

**Normalized gaps:** Define $D_n = \frac{p_{n+1} - p_n}{\ln p_n}$. Conjectured (and supported empirically) that $D_n$ follows an $\text{Exponential}(1)$ distribution as $n \to \infty$.

**Cramer's conjecture (1936):**
$$\limsup_{n \to \infty} \frac{p_{n+1} - p_n}{(\ln p_n)^2} = 1$$

Maximum prime gap near $N$ is $\sim (\ln N)^2$. Proven lower bound (Rankin): gaps can be as large as $c \cdot \frac{\ln N \cdot \ln \ln N \cdot \ln \ln \ln \ln N}{(\ln \ln \ln N)^2}$ infinitely often. The conjecture remains open.

**Relevance to factoring:** Prime density governs the expected number of trial divisions needed and the probability that a random number near $N$ is prime (important for RSA key generation and probabilistic factoring methods).
