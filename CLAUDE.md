# Factoring Lab - Project Instructions

## What this is
Research repo studying classical integer factorization bottlenecks, aiming toward
restricted-model hardness results (not a direct proof that factoring ∉ P, but
meaningful structural insights about why classical algorithms plateau).

## Language policy
- Python is the default for orchestration, analysis, generators, CLI
- Rust for performance-critical arithmetic kernels (later phases)
- OCaml for formal/inductive reasoning about algorithm structure (later phases)
- Use `uv` for Python package management

## Key conventions
- All algorithms implement `FactoringAlgorithm` base class from `algorithms/base.py`
- All generators return `SemiprimeSpec` with known factors
- Experiments must be reproducible (seedable randomness)
- Track instrumentation: iterations, gcd calls, modular multiplies
- Export results as CSV/Parquet for analysis

## Testing
- Run `uv run pytest -v` before claiming anything works
- Every algorithm needs correctness tests on known semiprimes
- Every generator needs structure validation tests
