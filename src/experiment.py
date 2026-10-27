"""Experiment configuration: pick pipeline components by name and run a simulation.

    from src.experiment import ExperimentConfig, run_experiment
    run_experiment(ExperimentConfig(fleet=750, rebalancing="oracle", hours=1))

To add a component, implement its interface and register a factory below.
"""

import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Callable

import config.config as config
from config import get_logger
from src.components.assignment_solver import GurobiAssignmentSolver, HighsAssignmentSolver
from src.components.batch_controller import BatchController
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rtv_vehicle_centric import VehicleCentricRTVGraphBuilder
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.joint_assignment_solver import JointAssignmentSolver, ZeroValue
from src.components.network_service import NetworkXService
from src.components.objectives import DelayObjective, ProfitObjective
from src.components.demand import HistoricalDemandModel
from src.components.demand_rebalance_solver import DemandRebalancingSolver
from src.components.oracle_rebalance_solver import OracleRebalancingSolver
from src.components.zones import ZoneIndex
from src.components.pricing import LinearFareModel
from src.components.rebalance_solver import NoRebalancingSolver, ReactiveRebalancingSolver
from src.components.request_sampler import HistoricalRequestSampler
from src.components.sampled_assignment_solver import SampledRequestsAssignmentSolver
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.trip_solver import AStarTripSolver
from src.database.simple_db import DuckDBPathCache
from src.interfaces import (
    IAssignmentSolver,
    IFareModel,
    IObjective,
    IRebalancingSolver,
    IRequestSampler,
    IRTVGraphBuilder,
    ITripSolver,
    IVehicleValue,
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
    cost_per_km: float = config.COST_PER_KM  # used by the profit objective
    fare_model: str = "linear"
    trip_solver: str = "astar"
    rtv_builder: str = "trip"
    assignment_solver: str = "gurobi"
    rebalancing: str = "reactive"
    oracle_horizon_min: float = 10.0
    demand_horizon_min: float = 15.0  # look-ahead of the demand rebalancer
    vehicle_value: str = "zero"  # V(node, time) used by the joint solver
    request_sampler: str = "historical"  # virtual future requests for the sampled solver
    sample_horizon_min: float = 5.0  # look-ahead of the sampled solver
    virtual_weight: float = 0.05  # weight of a virtual rider relative to a real one
    travel_time_factor: float = config.TRAVEL_TIME_FACTOR
    read_only: bool = False


@dataclass
class Context:
    """What component factories may use."""

    config: ExperimentConfig
    provider: SimulationDataProvider
    network: NetworkXService
    start_time: float
    end_time: float

    @cached_property
    def zones(self) -> ZoneIndex:
        return ZoneIndex(self.provider.graph)

    @cached_property
    def demand(self) -> HistoricalDemandModel:
        """Demand forecast fitted on every day except the simulated one."""
        t0 = self.provider.first_request_time()
        simulated_day = (t0 + timedelta(seconds=self.start_time)).date()
        return HistoricalDemandModel.fit(
            self.provider.request_table(), self.zones, t0, exclude={simulated_day}
        )

    @cached_property
    def simulated_day(self):
        return (self.provider.first_request_time() + timedelta(seconds=self.start_time)).date()


OBJECTIVES: dict[str, Callable[[Context], IObjective]] = {
    "delay": lambda ctx: DelayObjective(),
    "profit": lambda ctx: ProfitObjective(ctx.network, ctx.config.cost_per_km),
}
FARE_MODELS: dict[str, Callable[[Context], IFareModel]] = {
    "linear": lambda ctx: LinearFareModel(ctx.network),
}
TRIP_SOLVERS: dict[str, Callable[[Context], ITripSolver]] = {
    "astar": lambda ctx: AStarTripSolver(ctx.network.get_l1_cache()),
}
RTV_BUILDERS: dict[str, Callable[[ITripSolver], IRTVGraphBuilder]] = {
    "trip": TripCentricRTVGraphBuilder,
    "vehicle": VehicleCentricRTVGraphBuilder,
}
VEHICLE_VALUES: dict[str, Callable[[Context], IVehicleValue]] = {
    "zero": lambda ctx: ZeroValue(),
}
REQUEST_SAMPLERS: dict[str, Callable[[Context], IRequestSampler]] = {
    "historical": lambda ctx: HistoricalRequestSampler(
        ctx.demand,
        ctx.provider.trip_table(),
        ctx.zones,
        ctx.provider.first_request_time(),
        ctx.network,
        exclude={ctx.simulated_day},
    ),
}
ASSIGNMENT_SOLVERS: dict[str, Callable[[Context, IObjective], IAssignmentSolver]] = {
    "gurobi": lambda ctx, objective: GurobiAssignmentSolver(objective),
    "highs": lambda ctx, objective: HighsAssignmentSolver(objective),
    "joint": lambda ctx, objective: JointAssignmentSolver(
        objective, ctx.network, ctx.zones, VEHICLE_VALUES[ctx.config.vehicle_value](ctx)
    ),
    "sampled": lambda ctx, objective: SampledRequestsAssignmentSolver(
        objective,
        ctx.network,
        ctx.zones,
        REQUEST_SAMPLERS[ctx.config.request_sampler](ctx),
        horizon=ctx.config.sample_horizon_min * 60,
        weight=ctx.config.virtual_weight,
    ),
}


def _oracle(ctx: Context) -> IRebalancingSolver:
    horizon = ctx.config.oracle_horizon_min * 60
    future = ctx.provider.request_origins_between(ctx.start_time, ctx.end_time + horizon)
    return OracleRebalancingSolver(
        ctx.network, future["t"].to_numpy(), future["origin"].to_numpy(), horizon
    )


def _demand(ctx: Context) -> IRebalancingSolver:
    return DemandRebalancingSolver(
        ctx.network, ctx.zones, ctx.demand, horizon=ctx.config.demand_horizon_min * 60
    )


REBALANCERS: dict[str, Callable[[Context], IRebalancingSolver]] = {
    "reactive": lambda ctx: ReactiveRebalancingSolver(ctx.network),
    "oracle": _oracle,
    "demand": _demand,
    "none": lambda ctx: NoRebalancingSolver(),
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
        controller=BatchController(
            ctx.network,
            ctx.provider.iter_vehicles(cfg.fleet),
            fare_model=FARE_MODELS[cfg.fare_model](ctx),
        ),
        rv_builder=RVGraphBuilder(ctx.network, trip_solver),
        rtv_builder=RTV_BUILDERS[cfg.rtv_builder](trip_solver),
        assignment_solver=ASSIGNMENT_SOLVERS[cfg.assignment_solver](ctx, objective),
        rebalancing_solver=REBALANCERS[cfg.rebalancing](ctx),
        network_service=ctx.network,
        db_cache=db_cache,
        read_only=cfg.read_only,
    )


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run one simulation; returns the run folder with stats, plots and metadata."""
    for key, registry in [
        ("objective", OBJECTIVES),
        ("fare_model", FARE_MODELS),
        ("trip_solver", TRIP_SOLVERS),
        ("rtv_builder", RTV_BUILDERS),
        ("assignment_solver", ASSIGNMENT_SOLVERS),
        ("rebalancing", REBALANCERS),
        ("vehicle_value", VEHICLE_VALUES),
        ("request_sampler", REQUEST_SAMPLERS),
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
    network = NetworkXService(
        provider.graph, db=db_cache, travel_time_factor=cfg.travel_time_factor
    )
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
