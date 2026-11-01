"""What one experiment runs: the component names and their settings."""

from dataclasses import dataclass
from datetime import datetime

import config.config as config


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "sim_standard"
    fleet: int = config.NUMBER_OF_VEHICLES
    start: datetime | None = None  # local time; None uses config.MIN_START_TIME
    hours: float | None = None  # measured hours after warm-up; None uses config.END_TIME
    max_batches: int | None = None
    controller: str = "batch"
    fleet_replay: str = "arrays"
    zoning: str = "taxi"
    demand_model: str = "nowcast-past-day-of-week"  # simple and past-only
    objective: str = "delay"
    cost_per_km: float = config.COST_PER_KM  # used by the profit objective
    fare_model: str = "linear"
    trip_solver: str = "astar"
    rv_builder: str = "standard"
    rtv_builder: str = "trip"
    assignment_solver: str = "gurobi"
    rebalancing: str = "reactive"
    oracle_horizon_min: float = 10.0
    demand_horizon_min: float = 15.0  # look-ahead of the demand rebalancer
    history_weeks: int = 8  # weeks of past TLC history the past-* demand models average
    nowcast_window_min: float = 60.0  # past window the nowcast demand model compares with its base forecast
    vehicle_value: str = "zero"  # V(node, time) used by the joint solver
    request_sampler: str = "historical"  # virtual future requests for the sampled solver
    sample_horizon_min: float = 5.0  # look-ahead of the sampled solver
    virtual_weight: float = 0.05  # weight of a virtual rider relative to a real one
    fleet_plan_interval_s: float = 60.0  # how often the fleet MPC re-solves its plan
    fleet_futures: int = 10  # sampled demand futures of the fleet plan (vehicle value mpc_sampled)
    fleet_workers: int = 4  # processes solving them
    vehicle_value_weight: float = 0.003  # scales the fleet-MPC values (0 = paper's ILP with MPC moves)
    virtual_per_free: float = 1.0  # active virtual riders per free vehicle (*-matched solvers)
    scenarios: int = 5  # sampled futures of the two-stage solver
    scenario_horizon_min: float = 10.0  # look-ahead of the two-stage solver
    scenario_weight: float = 0.5  # weight of a future rider relative to a real one (two-stage)
    travel_time_factor: float = config.TRAVEL_TIME_FACTOR
    read_only: bool = False
    profile: bool = False  # cProfile the batches into profile.prof (slows the run)

