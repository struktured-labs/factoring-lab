
# Factoring Research Conversation Notes

## Objective
Explore whether prime factorization is outside P (or otherwise classify its complexity) and identify productive research directions.

## Key Points Discussed
1. **Current Knowledge**
   - Factoring is in BQP (via Shor's algorithm).
   - It is not known whether factoring is in P or not.
   - Proving factoring ∉ P would require major breakthroughs in lower-bound techniques.

2. **Two Viable Research Paths**
   - **Empirical analysis of classical factoring bottlenecks**
     - Study classical algorithms and identify structural limitations.
     - Hypothesis: classical methods rely heavily on smoothness events or congruence-of-squares constructions.
   - **Restricted-model hardness**
     - Define computational models capturing common classical algorithm structures.
     - Attempt to prove limitations within that model.

3. **Recommended Strategy**
   Phase 1: Build an experimental harness to study algorithm behavior.
   Phase 2: Analyze empirical bottlenecks.
   Phase 3: Use those observations to propose restricted models and potential theorems.

4. **Architecture Recommendation**
   - Use Python for orchestration, experimentation, and analysis.
   - Potentially move heavy arithmetic kernels to Rust later.
   - Maintain a stable algorithm interface so implementations can be swapped.

5. **Initial Algorithms**
   - Trial division
   - Pollard rho
   - Pollard p-1

6. **Instance Families to Generate**
   - Balanced semiprimes
   - Unbalanced semiprimes
   - p−1 smooth semiprimes
   - Random semiprimes

7. **Metrics to Record**
   - Runtime
   - Iterations
   - GCD calls
   - Modular multiplications
   - Success/failure

The purpose of the repository is to provide a systematic experimental environment to investigate structural properties of classical factoring algorithms.
