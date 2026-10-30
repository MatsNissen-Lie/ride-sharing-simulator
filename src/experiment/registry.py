"""
Every pipeline component by name: one registry per interface. A factory
gets the experiment `Context` (and sometimes collaborators) and returns the
component. To add a component, implement its interface and register it here.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cached_property
from typing import Callable

import numpy as np

from config import get_logger
from src.components.assignment.assignment_solver import GurobiAssignmentSolver, HighsAssignmentSolver
from src.components.simulation.batch_controller import BatchController
from src.components.simulation.fleet_replay import ArrayFleetReplay, PathFleetReplay
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rtv_vehicle_centric import VehicleCentricRTVGraphBuilder
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.assignment.joint_assignment_solver import JointAssignmentSolver, ZeroValue
from src.components.network.network_service import NetworkXService
from src.components.economics.objectives import DelayObjective, ProfitObjective
from src.components.demand.demand_model import HOLIDAYS, HistoricalDemandModel
from src.components.fleet_plan.fleet_mpc import FleetMpcValue
from src.components.fleet_plan.sampled_fleet_mpc import SampledFleetMpcValue
from src.components.fleet_plan.flat_value import ZoneAveragedValue
from src.components.fleet_plan.placebo_values import ZoneAveragedPlan, ZoneShuffledPlan
from src.components.rebalancing.mpc_rebalance_solver import MpcRebalancingSolver
from src.components.fleet_plan.zone_trips import ZoneTrips
from src.components.rebalancing.demand_rebalance_solver import DemandRebalancingSolver
from src.components.rebalancing.oracle_rebalance_solver import OracleRebalancingSolver
from src.components.network.zones import ZoneIndex
from src.components.economics.pricing import LinearFareModel
from src.components.rebalancing.rebalance_solver import NoRebalancingSolver, ReactiveRebalancingSolver
from src.components.demand.request_sampler import HistoricalRequestSampler
from src.components.assignment.two_stage_assignment_solver import (
    ChainedTwoStageAssignmentSolver,
    TwoStageAssignmentSolver,
)
from src.components.assignment.sampled_assignment_solver import (
    BidirectionalChainSolver,
    BidirectionalSampledSolver,
    FullMatchingChainSolver,
    FullMatchingSampledSolver,
    SampledChainAssignmentSolver,
    SampledRequestsAssignmentSolver,
    SupplyMatchedChainSolver,
    SupplyMatchedSampledSolver,
)
from src.components.simulation.simulation_data_provider import SimulationDataProvider
from src.components.rebalancing.transport_rebalance_solver import TransportRebalancingSolver
from src.components.network.trip_solver import AStarTripSolver
from src.interfaces import (
    IAssignmentSolver,
    IDemandModel,
    IFleetReplay,
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
from src.experiment.config import ExperimentConfig





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
        replay=FLEET_REPLAYS[ctx.config.fleet_replay](ctx),
        fare_model=FARE_MODELS[ctx.config.fare_model](ctx),
    ),
}
FLEET_REPLAYS: dict[str, Callable[[Context], IFleetReplay]] = {
    "arrays": lambda ctx: ArrayFleetReplay(ctx.network),  # leg totals from per-graph arrays
    "paths": lambda ctx: PathFleetReplay(ctx.network),  # a path object per leg (original)
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


def _mpc(ctx: Context, cls=FleetMpcValue, **kwargs) -> IVehicleValue:
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
    return cls(
        ctx.network, ctx.zones, ctx.demand, ctx.zone_trips, objective, reward,
        resolve_every=ctx.config.fleet_plan_interval_s, **kwargs,
    )


VEHICLE_VALUES: dict[str, Callable[[Context], IVehicleValue]] = {
    "zero": lambda ctx: ZeroValue(),
    "mpc": lambda ctx: _mpc(ctx, pooling=MPC_POOLING, weight=ctx.config.vehicle_value_weight),
    "mpc_sampled": lambda ctx: _mpc(
        ctx, SampledFleetMpcValue, pooling=MPC_POOLING, weight=ctx.config.vehicle_value_weight,
        n_futures=ctx.config.fleet_futures, workers=ctx.config.fleet_workers,
    ),
    "mpc_p1": lambda ctx: _mpc(ctx, pooling=1.0),
    "mpc_half": lambda ctx: _mpc(ctx, pooling=MPC_POOLING, weight=0.5),
    "mpc_flat": lambda ctx: ZoneAveragedValue(
        _mpc(ctx, pooling=MPC_POOLING), np.array(list(ctx.zones.center_node.values()))
    ),    # Placebo controls: the scaled MPC plan and its moves, values without location meaning.
    "mpc_shuffled": lambda ctx: ZoneShuffledPlan(
        _mpc(ctx, pooling=MPC_POOLING, weight=ctx.config.vehicle_value_weight), ctx.zones
    ),
    "mpc_flat_plan": lambda ctx: ZoneAveragedPlan(
        _mpc(ctx, pooling=MPC_POOLING, weight=ctx.config.vehicle_value_weight),
        np.array(list(ctx.zones.center_node.values())),
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


def _two_stage(cls, ctx: Context, objective: IObjective) -> IAssignmentSolver:
    return cls(
        objective,
        ctx.network,
        ctx.zones,
        ctx.vehicle_value,
        REQUEST_SAMPLERS[ctx.config.request_sampler](ctx),
        n_scenarios=ctx.config.scenarios,
        horizon=ctx.config.scenario_horizon_min * 60,
        weight=ctx.config.scenario_weight,
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
    "two-stage": lambda ctx, objective: _two_stage(TwoStageAssignmentSolver, ctx, objective),
    "two-stage-chain": lambda ctx, objective: _two_stage(ChainedTwoStageAssignmentSolver, ctx, objective),
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
