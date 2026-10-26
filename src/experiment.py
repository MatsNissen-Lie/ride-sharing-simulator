"""Experiment configuration: pick pipeline components by name and run a simulation.

    from src.experiment import ExperimentConfig, run_experiment
    run_experiment(ExperimentConfig(fleet=750, rebalancing="oracle", hours=1))

To add a component, implement its interface and register a factory below.
"""

import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import config.config as config
from config import get_logger
from src.components.assignment_solver import GurobiAssignmentSolver, HighsAssignmentSolver
from src.components.batch_controller import BatchController
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rtv_vehicle_centric import VehicleCentricRTVGraphBuilder
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.network_service import NetworkXService
from src.components.objectives import DelayObjective
from src.components.oracle_rebalance_solver import OracleRebalancingSolver
from src.components.rebalance_solver import ReactiveRebalancingSolver
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.trip_solver import AStarTripSolver
from src.database.simple_db import DuckDBPathCache
from src.interfaces import (
    IAssignmentSolver,
    IObjective,
    IRebalancingSolver,
    IRTVGraphBuilder,
    ITripSolver,
)
from src.interfaces.stats import BatchStatistics
from src.simulator import Simulator
from src.utils.generate_plots import generate_plots
from src.utils.output_manager import SimulationOutputManager
from src.utils.stats_logger import SimulationStatsLogger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "sim_standard"
    fleet: int = config.NUMBER_OF_VEHICLES
    start: datetime | None = None  # local time; None uses config.MIN_START_TIME
    hours: float | None = None  # measured hours after warm-up; None uses config.END_TIME
    max_batches: int | None = None
    objective: str = "delay"
    trip_solver: str = "astar"
    rtv_builder: str = "trip"
    assignment_solver: str = "gurobi"
    rebalancing: str = "reactive"
    oracle_horizon_min: float = 10.0
    read_only: bool = False


@dataclass
class Context:
    """What component factories may use."""

    config: ExperimentConfig
    provider: SimulationDataProvider
    network: NetworkXService
    start_time: float
    end_time: float


OBJECTIVES: dict[str, Callable[[Context], IObjective]] = {
    "delay": lambda ctx: DelayObjective(),
}
TRIP_SOLVERS: dict[str, Callable[[Context], ITripSolver]] = {
    "astar": lambda ctx: AStarTripSolver(ctx.network.get_l1_cache()),
}
RTV_BUILDERS: dict[str, Callable[[ITripSolver], IRTVGraphBuilder]] = {
    "trip": TripCentricRTVGraphBuilder,
    "vehicle": VehicleCentricRTVGraphBuilder,
}
ASSIGNMENT_SOLVERS: dict[str, Callable[[IObjective], IAssignmentSolver]] = {
    "gurobi": GurobiAssignmentSolver,
    "highs": HighsAssignmentSolver,
}


def _oracle(ctx: Context) -> IRebalancingSolver:
    horizon = ctx.config.oracle_horizon_min * 60
    future = ctx.provider.request_origins_between(ctx.start_time, ctx.end_time + horizon)
    return OracleRebalancingSolver(
        ctx.network, future["t"].to_numpy(), future["origin"].to_numpy(), horizon
    )


REBALANCERS: dict[str, Callable[[Context], IRebalancingSolver]] = {
    "reactive": lambda ctx: ReactiveRebalancingSolver(ctx.network),
    "oracle": _oracle,
}


def simulation_window(cfg: ExperimentConfig, provider: SimulationDataProvider) -> tuple[float, float]:
    """Start and end in simulation seconds."""
    start = config.MIN_START_TIME
    if cfg.start is not None:
        start = (cfg.start - provider.first_request_time()).total_seconds()
    if cfg.hours is None:
        return start, start + config.END_TIME - config.MIN_START_TIME
    return start, start + config.WARMUP_DURATION + cfg.hours * 3600


def build_simulator(ctx: Context, db_cache: DuckDBPathCache) -> Simulator:
    cfg = ctx.config
    objective = OBJECTIVES[cfg.objective](ctx)
    trip_solver = TRIP_SOLVERS[cfg.trip_solver](ctx)
    return Simulator(
        provider=ctx.provider,
        controller=BatchController(ctx.network, ctx.provider.iter_vehicles(cfg.fleet)),
        rv_builder=RVGraphBuilder(ctx.network, trip_solver),
        rtv_builder=RTV_BUILDERS[cfg.rtv_builder](trip_solver),
        assignment_solver=ASSIGNMENT_SOLVERS[cfg.assignment_solver](objective),
        rebalancing_solver=REBALANCERS[cfg.rebalancing](ctx),
        network_service=ctx.network,
        db_cache=db_cache,
        read_only=cfg.read_only,
    )


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run one simulation; returns the run folder with stats, plots and metadata."""
    for key, registry in [
        ("objective", OBJECTIVES),
        ("trip_solver", TRIP_SOLVERS),
        ("rtv_builder", RTV_BUILDERS),
        ("assignment_solver", ASSIGNMENT_SOLVERS),
        ("rebalancing", REBALANCERS),
    ]:
        if getattr(cfg, key) not in registry:
            raise ValueError(f"Unknown {key} {getattr(cfg, key)!r}; choose from {list(registry)}")

    provider = SimulationDataProvider()
    start_time, end_time = simulation_window(cfg, provider)

    output = SimulationOutputManager(base_dir="reports", run_name=cfg.name)
    metadata = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    metadata |= {"experiment": asdict(replace(cfg, start=str(cfg.start))), "start_time": start_time,
                 "end_time": end_time}
    output.save_metadata(metadata)
    csv_path = os.path.join(output.get_run_dir(), "simulation_stats.csv")

    logger.info("Initializing components...")
    db_cache = DuckDBPathCache(read_only=cfg.read_only)
    network = NetworkXService(provider.graph, db=db_cache)
    simulator = build_simulator(Context(cfg, provider, network, start_time, end_time), db_cache)

    stats_logger = SimulationStatsLogger()
    try:
        for stats in simulator.run(
            request_iterator=provider.iter_requests(),
            batch_duration=config.BATCH_DURATION,
            min_start_time=start_time,
            warmup_duration=config.WARMUP_DURATION,
            max_batches=cfg.max_batches,
            end_time=end_time,
        ):
            if not isinstance(stats, BatchStatistics):
                continue
            if stats.is_warmup:
                logger.info(f"🔥 Warm-up batch {stats.batch_id}. Stats NOT logged.")
                continue
            stats_logger.log_batch(stats)
            if stats.batch_id % 20 == 0:
                stats_logger.flush_to_csv(csv_path)
        logger.info("🏁 Simulation complete.")
    except KeyboardInterrupt:
        logger.info("🛑 Simulation stopped by user (Ctrl+C).")
    finally:
        stats_logger.flush_to_csv(csv_path)
        stats_logger.print_summary()

    if os.path.exists(csv_path):
        logger.info("📊 Generating plots...")
        generate_plots(csv_path)
    else:
        logger.warning("No stats CSV found, skipping plot generation.")
    return Path(output.get_run_dir())
