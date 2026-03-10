"""Tests for benchmark runner."""

from factoring_lab.algorithms import TrialDivision, PollardRho
from factoring_lab.benchmarks.runner import ExperimentConfig, run_experiment


def test_smoke_benchmark():
    config = ExperimentConfig(
        name="smoke_test",
        family="balanced",
        bits=24,
        count=3,
        algorithms=[TrialDivision(), PollardRho()],
        seed=42,
    )
    result = run_experiment(config)
    assert len(result.rows) == 6  # 3 instances * 2 algorithms
    assert "trial_division" in result.summary["algorithms"]
    assert "pollard_rho" in result.summary["algorithms"]

    for row in result.rows:
        assert "algorithm_name" in row
        assert "n" in row
        assert "success" in row
        assert "runtime_seconds" in row
