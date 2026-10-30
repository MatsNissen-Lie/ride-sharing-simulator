"""Experiment configuration: pick pipeline components by name and run a simulation.

    from src.experiment import ExperimentConfig, run_experiment
    run_experiment(ExperimentConfig(fleet=750, rebalancing="oracle", hours=1))

- `config.py`: ExperimentConfig, the names and settings of one run;
- `registry.py`: every component by name (one registry per interface) and
  the Context their factories receive; add a component there;
- `run.py`: builds the simulator and runs it.
"""

from src.experiment.config import ExperimentConfig
from src.experiment.registry import (
    ASSIGNMENT_SOLVERS,
    CONTROLLERS,
    DEMAND_MODELS,
    FARE_MODELS,
    FLEET_REPLAYS,
    OBJECTIVES,
    REBALANCERS,
    REQUEST_SAMPLERS,
    RTV_BUILDERS,
    RV_BUILDERS,
    TRIP_SOLVERS,
    VEHICLE_VALUES,
    ZONINGS,
    Context,
    simulated_days,
)
from src.experiment.run import build_simulator, run_experiment, simulation_window

__all__ = [
    "ExperimentConfig",
    "ASSIGNMENT_SOLVERS",
    "CONTROLLERS",
    "DEMAND_MODELS",
    "FARE_MODELS",
    "FLEET_REPLAYS",
    "OBJECTIVES",
    "REBALANCERS",
    "REQUEST_SAMPLERS",
    "RTV_BUILDERS",
    "RV_BUILDERS",
    "TRIP_SOLVERS",
    "VEHICLE_VALUES",
    "ZONINGS",
    "Context",
    "simulated_days",
    "build_simulator",
    "run_experiment",
    "simulation_window",
]
