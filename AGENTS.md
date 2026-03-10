
# Factoring Lab Agent Instructions

## Mission
Study structural bottlenecks of classical integer factorization algorithms.

## Priorities
1. Correctness
2. Reproducibility
3. Instrumentation
4. Readability
5. Extensibility

## Coding Standards
- Python 3.12+
- Type hints
- Dataclasses where appropriate
- Small clear functions
- Explicit naming
- Seedable randomness

## Directory Rules
algorithms/ : factoring implementations
generators/ : integer instance generators
benchmarks/ : experiment orchestration
metrics/ : result schemas and logging

## Experiment Expectations
Every experiment must:
- define integer family
- define parameters
- be reproducible
- export machine-readable results

## Research Hypothesis
Many classical factoring methods rely on:
- smoothness events
- congruence-of-squares constructions

The code should help evaluate these hypotheses empirically.
