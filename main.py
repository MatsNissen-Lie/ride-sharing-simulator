import argparse
import os
import sys
from datetime import datetime

from src.experiment import (
    ASSIGNMENT_SOLVERS,
    OBJECTIVES,
    REBALANCERS,
    REQUEST_SAMPLERS,
    RTV_BUILDERS,
    TRIP_SOLVERS,
    VEHICLE_VALUES,
    ExperimentConfig,
    run_experiment,
)

DEFAULTS = ExperimentConfig()


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run the ride-sharing simulation.")
    parser.add_argument("--name", default=DEFAULTS.name, help="Suffix for the run folder in reports/.")
    parser.add_argument("--fleet", type=int, default=DEFAULTS.fleet, help="Fleet size.")
    parser.add_argument(
        "--start",
        type=datetime.fromisoformat,
        default=None,
        help="Start of warm-up, local time, e.g. 2025-01-16T17:45 (default: MIN_START_TIME).",
    )
    parser.add_argument(
        "--hours", type=float, default=None, help="Simulated hours after warm-up (default: END_TIME)."
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many batches.")
    parser.add_argument("--objective", choices=OBJECTIVES, default=DEFAULTS.objective)
    parser.add_argument(
        "--cost-per-km",
        type=float,
        default=DEFAULTS.cost_per_km,
        help="Variable cost per vehicle-km in USD (profit objective).",
    )
    parser.add_argument("--trip-solver", choices=TRIP_SOLVERS, default=DEFAULTS.trip_solver)
    parser.add_argument(
        "--rtv",
        choices=RTV_BUILDERS,
        default=DEFAULTS.rtv_builder,
        help="RTV graph builder: trip-centric or vehicle-centric (paper's Algorithm 1).",
    )
    parser.add_argument(
        "--solver",
        choices=ASSIGNMENT_SOLVERS,
        default=DEFAULTS.assignment_solver,
        help="Assignment ILP solver (HiGHS needs no licence; joint = HiGHS with rebalancing moves).",
    )
    parser.add_argument(
        "--vehicle-value",
        choices=VEHICLE_VALUES,
        default=DEFAULTS.vehicle_value,
        help="Value of where vehicles end up, used by --solver joint.",
    )
    parser.add_argument(
        "--request-sampler",
        choices=REQUEST_SAMPLERS,
        default=DEFAULTS.request_sampler,
        help="Virtual future requests, used by --solver sampled.",
    )
    parser.add_argument(
        "--sample-horizon",
        type=float,
        default=DEFAULTS.sample_horizon_min,
        help="Look-ahead of --solver sampled in minutes.",
    )
    parser.add_argument(
        "--virtual-weight",
        type=float,
        default=DEFAULTS.virtual_weight,
        help="Weight of a virtual rider relative to a real one (--solver sampled).",
    )
    parser.add_argument(
        "--virtual-per-free",
        type=float,
        default=DEFAULTS.virtual_per_free,
        help="Active virtual riders per free vehicle (--solver sampled-matched, sampled-chain-matched).",
    )
    parser.add_argument(
        "--rebalancing",
        choices=REBALANCERS,
        default=DEFAULTS.rebalancing,
        help="reactive (paper), forecast-greedy, forecast-optimal (fewest moves), mpc, oracle (perfect information) or none; demand/transport are old names.",
    )
    parser.add_argument(
        "--horizon", type=float, default=DEFAULTS.oracle_horizon_min, help="Oracle look-ahead in minutes."
    )
    parser.add_argument(
        "--travel-time-factor",
        type=float,
        default=DEFAULTS.travel_time_factor,
        help="Override config.TRAVEL_TIME_FACTOR (1.0 = free-flow speeds).",
    )
    parser.add_argument("--read-only", action="store_true", help="Do not write to the route database.")
    a = parser.parse_args()
    return ExperimentConfig(
        name=a.name,
        fleet=a.fleet,
        start=a.start,
        hours=a.hours,
        max_batches=a.max_batches,
        objective=a.objective,
        cost_per_km=a.cost_per_km,
        trip_solver=a.trip_solver,
        rtv_builder=a.rtv,
        assignment_solver=a.solver,
        vehicle_value=a.vehicle_value,
        request_sampler=a.request_sampler,
        sample_horizon_min=a.sample_horizon,
        virtual_weight=a.virtual_weight,
        virtual_per_free=a.virtual_per_free,
        rebalancing=a.rebalancing,
        oracle_horizon_min=a.horizon,
        travel_time_factor=a.travel_time_factor,
        read_only=a.read_only,
    )


if __name__ == "__main__":
    # Sets of requests and vehicles iterate in hash order, which decides solver
    # tie-breaks. Python randomises string hashing per process, so fix the seed
    # to make runs reproducible.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    run_experiment(parse_args())
