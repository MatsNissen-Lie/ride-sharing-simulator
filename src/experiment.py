"""Experiment configuration: pick pipeline components by name and run a simulation.

    from src.experiment import ExperimentConfig, run_experiment
    run_experiment(ExperimentConfig(fleet=750, rebalancing="oracle", hours=1))

To add a component, implement its interface and register a factory below.
"""

import os
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np

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
from src.components.demand import HOLIDAYS, HistoricalDemandModel
from src.components.fleet_mpc import FleetMpcValue
from src.components.flat_value import ZoneAveragedValue
from src.components.mpc_rebalance_solver import MpcRebalancingSolver
from src.components.zone_trips import ZoneTrips
from src.components.demand_rebalance_solver import DemandRebalancingSolver
from src.components.oracle_rebalance_solver import OracleRebalancingSolver
from src.components.zones import ZoneIndex
from src.components.pricing import LinearFareModel
from src.components.rebalance_solver import NoRebalancingSolver, ReactiveRebalancingSolver
from src.components.request_sampler import HistoricalRequestSampler
from src.components.sampled_assignment_solver import (
    BidirectionalChainSolver,
    BidirectionalSampledSolver,
    FullMatchingChainSolver,
    FullMatchingSampledSolver,
    SampledChainAssignmentSolver,
    SampledRequestsAssignmentSolver,
    SupplyMatchedChainSolver,
    SupplyMatchedSampledSolver,
)
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.transport_rebalance_solver import TransportRebalancingSolver
from src.components.trip_solver import AStarTripSolver
from src.database.simple_db import DuckDBPathCache
from src.interfaces import (
    IAssignmentSolver,
    IDemandModel,
    IRVGraphBuilder,
    IZoneIndex,
    IFareModel,
    IObjective,
    IRebalancingSolver,
    IRequestSampler,
    IRTVGraphBuilder,
    ITripSolver,
    IVehicleValue,
    Request,
)
from src.interfaces.batch_control import IBatchController
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
    controller: str = "batch"
    zoning: str = "taxi"
    demand_model: str = "historical"
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
    vehicle_value: str = "zero"  # V(node, time) used by the joint solver
    request_sampler: str = "historical"  # virtual future requests for the sampled solver
    sample_horizon_min: float = 5.0  # look-ahead of the sampled solver
    virtual_weight: float = 0.05  # weight of a virtual rider relative to a real one
    virtual_per_free: float = 1.0  # active virtual riders per free vehicle (*-matched solvers)
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

    # Built once and shared by every component that needs them.
    @cached_property
    def zones(self) -> IZoneIndex:
        return ZONINGS[self.config.zoning](self)

    @cached_property
    def demand(self) -> IDemandModel:
        return DEMAND_MODELS[self.config.demand_model](self)

    @cached_property
    def zone_trips(self) -> ZoneTrips:
        """Destination shares and trip times per pair of zones, without the simulated day."""
        return ZoneTrips.fit(
            self.provider.request_trips(),
            self.zones,
            self.network.get_l1_cache(),
            self.provider.first_request_time(),
            exclude=self.excluded_days,
        )

    @cached_property
    def vehicle_value(self) -> IVehicleValue:
        """One instance, shared by the joint solver and rebalancers that follow its plan."""
        return VEHICLE_VALUES[self.config.vehicle_value](self)

    @cached_property
    def excluded_days(self) -> set[date]:
        """Every calendar day the run touches; models fitted on history leave them out."""
        return simulated_days(self.provider.first_request_time(), self.start_time, self.end_time)


def _historical_demand(ctx: Context) -> IDemandModel:
    """Fitted on every day except the simulated ones."""
    return HistoricalDemandModel.fit(
        ctx.provider.request_table(),
        ctx.zones,
        ctx.provider.first_request_time(),
        exclude=ctx.excluded_days,
    )


def simulated_days(t0: datetime, start_time: float, end_time: float) -> set[date]:
    """Every calendar day the simulation window touches (a night run spans two)."""
    first = (t0 + timedelta(seconds=start_time)).date()
    last = (t0 + timedelta(seconds=end_time)).date()
    return {first + timedelta(days=k) for k in range((last - first).days + 1)}


CONTROLLERS: dict[str, Callable[[Context], IBatchController]] = {
    "batch": lambda ctx: BatchController(
        ctx.network,
        ctx.provider.iter_vehicles(ctx.config.fleet),
        fare_model=FARE_MODELS[ctx.config.fare_model](ctx),
    ),
}
ZONINGS: dict[str, Callable[[Context], IZoneIndex]] = {
    "taxi": lambda ctx: ZoneIndex(ctx.provider.graph),
}
DEMAND_MODELS: dict[str, Callable[[Context], IDemandModel]] = {
    "historical": _historical_demand,
}
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
RV_BUILDERS: dict[str, Callable[[Context, ITripSolver], IRVGraphBuilder]] = {
    "standard": lambda ctx, trip_solver: RVGraphBuilder(ctx.network, trip_solver),
}
RTV_BUILDERS: dict[str, Callable[[ITripSolver], IRTVGraphBuilder]] = {
    "trip": TripCentricRTVGraphBuilder,
    "vehicle": VehicleCentricRTVGraphBuilder,
}


# Riders served per vehicle per average trip time in highs + none on the tuning day
# Tue 2025-01-14 (scripts/analysis/pooling_factor.py): 1.66.
MPC_POOLING = 1.6


def _mpc(ctx: Context, **kwargs) -> IVehicleValue:
    """
    Fleet MPC values. A rider is worth what leaving a new rider unserved costs
    (delay objective: its base penalty); where that is nothing (profit), the
    mean fare of training trips.
    """
    objective = OBJECTIVES[ctx.config.objective](ctx)
    fresh = Request("mpc_reward", 0, 0, 0.0, latest_dropoff_time=0.0, latest_pickup_time=0.0)
    reward = objective.unserved_penalty(fresh, 0.0)
    if reward <= 0:
        fares = FARE_MODELS[ctx.config.fare_model](ctx)
        sample = ctx.provider.request_trips().sample(2000, seed=0)
        reward = sum(fares.price(o, d) for o, d in zip(sample["orig_node"], sample["dest_node"])) / 2000
    return FleetMpcValue(ctx.network, ctx.zones, ctx.demand, ctx.zone_trips, objective, reward, **kwargs)


VEHICLE_VALUES: dict[str, Callable[[Context], IVehicleValue]] = {
    "zero": lambda ctx: ZeroValue(),
    "mpc": lambda ctx: _mpc(ctx, pooling=MPC_POOLING),
    "mpc_p1": lambda ctx: _mpc(ctx, pooling=1.0),
    "mpc_half": lambda ctx: _mpc(ctx, pooling=MPC_POOLING, weight=0.5),
    "mpc_flat": lambda ctx: ZoneAveragedValue(
        _mpc(ctx, pooling=MPC_POOLING), np.array(list(ctx.zones.center_node.values()))
    ),
}
REQUEST_SAMPLERS: dict[str, Callable[[Context], IRequestSampler]] = {
    "historical": lambda ctx: HistoricalRequestSampler(
        ctx.demand,
        HistoricalRequestSampler.read_trips(ctx.provider.iter_request_files()),
        ctx.zones,
        ctx.provider.first_request_time(),
        ctx.network,
        exclude={*ctx.excluded_days, *HOLIDAYS},
    ),
}
def _sampled(cls, ctx: Context, objective: IObjective, **kwargs) -> IAssignmentSolver:
    return cls(
        objective,
        ctx.network,
        ctx.zones,
        ctx.vehicle_value,
        REQUEST_SAMPLERS[ctx.config.request_sampler](ctx),
        horizon=ctx.config.sample_horizon_min * 60,
        weight=ctx.config.virtual_weight,
        **kwargs,
    )


ASSIGNMENT_SOLVERS: dict[str, Callable[[Context, IObjective], IAssignmentSolver]] = {
    "gurobi": lambda ctx, objective: GurobiAssignmentSolver(objective),
    "highs": lambda ctx, objective: HighsAssignmentSolver(objective),
    "joint": lambda ctx, objective: JointAssignmentSolver(
        objective, ctx.network, ctx.zones, ctx.vehicle_value
    ),
    "sampled": lambda ctx, objective: _sampled(SampledRequestsAssignmentSolver, ctx, objective),
    "sampled-chain": lambda ctx, objective: _sampled(SampledChainAssignmentSolver, ctx, objective),
    "sampled-matched": lambda ctx, objective: _sampled(
        SupplyMatchedSampledSolver, ctx, objective, virtual_per_free=ctx.config.virtual_per_free
    ),
    "sampled-chain-matched": lambda ctx, objective: _sampled(
        SupplyMatchedChainSolver, ctx, objective, virtual_per_free=ctx.config.virtual_per_free
    ),
    "sampled-bidir": lambda ctx, objective: _sampled(
        BidirectionalSampledSolver, ctx, objective, virtual_per_free=ctx.config.virtual_per_free
    ),
    "sampled-chain-bidir": lambda ctx, objective: _sampled(
        BidirectionalChainSolver, ctx, objective, virtual_per_free=ctx.config.virtual_per_free
    ),
    "sampled-lsa": lambda ctx, objective: _sampled(
        FullMatchingSampledSolver, ctx, objective, virtual_per_free=ctx.config.virtual_per_free
    ),
    "sampled-chain-lsa": lambda ctx, objective: _sampled(
        FullMatchingChainSolver, ctx, objective, virtual_per_free=ctx.config.virtual_per_free
    ),
}


def _oracle(ctx: Context) -> IRebalancingSolver:
    horizon = ctx.config.oracle_horizon_min * 60
    future = ctx.provider.request_origins_between(ctx.start_time, ctx.end_time + horizon)
    return OracleRebalancingSolver(
        ctx.network, future["t"].to_numpy(), future["origin"].to_numpy(), horizon
    )


def _transport(ctx: Context) -> IRebalancingSolver:
    return TransportRebalancingSolver(
        ctx.network, ctx.zones, ctx.demand, horizon=ctx.config.demand_horizon_min * 60
    )


def _demand(ctx: Context) -> IRebalancingSolver:
    return DemandRebalancingSolver(
        ctx.network, ctx.zones, ctx.demand, horizon=ctx.config.demand_horizon_min * 60
    )


REBALANCERS: dict[str, Callable[[Context], IRebalancingSolver]] = {
    "reactive": lambda ctx: ReactiveRebalancingSolver(ctx.network),
    "oracle": _oracle,
    "forecast-greedy": _demand,  # chase every waiting rider, then fill zone targets greedily
    "forecast-optimal": _transport,  # reachable riders only, then fewest-moves transport
    "demand": _demand,  # old name of forecast-greedy, kept so earlier runs reproduce
    "transport": _transport,  # old name of forecast-optimal
    "none": lambda ctx: NoRebalancingSolver(),
    "mpc": lambda ctx: MpcRebalancingSolver(ctx.network, ctx.zones, ctx.vehicle_value),
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
        controller=CONTROLLERS[cfg.controller](ctx),
        rv_builder=RV_BUILDERS[cfg.rv_builder](ctx, trip_solver),
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
        ("controller", CONTROLLERS),
        ("zoning", ZONINGS),
        ("demand_model", DEMAND_MODELS),
        ("rv_builder", RV_BUILDERS),
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
