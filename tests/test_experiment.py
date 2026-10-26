"""Experiment configuration."""

import pytest

from src.experiment import ExperimentConfig, run_experiment


def test_unknown_component_is_rejected_before_any_work():
    with pytest.raises(ValueError, match="rebalancing"):
        run_experiment(ExperimentConfig(rebalancing="psychic"))
