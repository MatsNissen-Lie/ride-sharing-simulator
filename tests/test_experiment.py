"""Experiment configuration."""

import pytest

from src.experiment import ExperimentConfig, run_experiment


def test_unknown_component_is_rejected_before_any_work():
    with pytest.raises(ValueError, match="rebalancing"):
        run_experiment(ExperimentConfig(rebalancing="psychic"))


def test_demand_model_excludes_every_simulated_day():
    from datetime import date, datetime

    from src.experiment import simulated_days

    t0 = datetime(2025, 1, 1)
    day = 24 * 3600
    evening = simulated_days(t0, 15 * day + 17.75 * 3600, 15 * day + 19 * 3600)
    night = simulated_days(t0, 17 * day + 23.75 * 3600, 18 * day + 3 * 3600)
    assert evening == {date(2025, 1, 16)}
    assert night == {date(2025, 1, 18), date(2025, 1, 19)}


def test_old_rebalancer_names_still_work():
    from src.experiment import REBALANCERS

    assert REBALANCERS["forecast-greedy"] is REBALANCERS["demand"]
    assert REBALANCERS["forecast-optimal"] is REBALANCERS["transport"]
