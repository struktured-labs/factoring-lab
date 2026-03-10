# Factoring Lab

Research harness for studying structural bottlenecks of classical integer factorization algorithms.

## Research Questions

1. What structural properties make integers easy or hard to factor classically?
2. Do all competitive classical methods fundamentally reduce to smoothness/congruence-of-squares?
3. Can we formalize restricted computational models that capture these bottlenecks?
4. What makes quantum factoring (Shor) structurally different from classical approaches?

## Quick Start

```bash
uv venv && uv pip install -e ".[dev]"

# Factor a single number
uv run factoring-lab factor 1000036000099

# Run a benchmark
uv run factoring-lab benchmark --family balanced --bits 32 --count 20 --json

# Export results
uv run factoring-lab benchmark --family smooth_pm1 --bits 48 --count 50 --output reports/smooth_pm1_48bit.csv

# Run tests
uv run pytest -v
```

## Algorithms (Phase 1)

| Algorithm | Type | Exploits |
|-----------|------|----------|
| Trial division | Deterministic | Small factors |
| Pollard rho | Probabilistic | Birthday paradox on residues |
| Pollard p-1 | Probabilistic | Smoothness of p-1 |

## Instance Families

| Family | Structure | Purpose |
|--------|-----------|---------|
| `balanced` | p,q ~ same size | Hardest general case |
| `unbalanced` | one small factor | Easy for trial division |
| `smooth_pm1` | p-1 is B-smooth | Easy for Pollard p-1 |
| `random` | random factor split | Baseline |

## Project Structure

```
src/factoring_lab/
  algorithms/    # Factoring implementations
  generators/    # Structured semiprime generators
  benchmarks/    # Experiment orchestration
  metrics/       # Result schemas and export
  cli/           # Command-line interface
tests/           # Mirror of source structure
reports/         # Benchmark outputs
docs/            # Research notes
```

## Roadmap

- **Phase 1** (current): Baseline algorithms + harness
- **Phase 2**: ECM, quadratic sieve, deeper instrumentation
- **Phase 3**: Restricted-model hardness formalization
- **Later**: Rust kernels for heavy arithmetic, OCaml for formal/inductive reasoning
