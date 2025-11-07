from dataclasses import dataclass
from typing import (
    Mapping,
    Sequence,
    Protocol,
    runtime_checkable,
    Optional,
    NewType,
)

# --------- Domain IDs & Units ----------
RequestId = NewType("RequestId", int)
VehicleId = NewType("VehicleId", int)
TripId = NewType("TripId", int)

# All times in seconds; distances in meters; costs in seconds-equivalent.


# --------- Core Entities ----------
@dataclass(slots=True, frozen=True)
class NodeRef:
    """Opaque reference to a graph node (road network)."""

    node_id: int


@dataclass(slots=True, frozen=True)
class Request:
    """Incoming ride request (becomes a Passenger after pickup)."""

    rid: RequestId
    t_request: float
    origin: NodeRef
    destination: NodeRef


@dataclass(slots=True)
class Passenger:
    """A picked-up request currently aboard a vehicle."""

    rid: RequestId
    t_pickup: float
    origin: NodeRef
    destination: NodeRef


@dataclass(slots=True)
class VehicleState:
    """Vehicle snapshot needed for routing/feasibility."""

    vid: VehicleId
    capacity: int
    position: NodeRef  # current graph node
    onboard: list[
        Passenger
    ]  # in FIFO of actual planned route order or unordered (planner decides)
    # Optional: planned route to honor current passengers (kept by the router)
    planned_route: Optional["RoutePlan"] = None


# --------- Constraints & Cost ----------
@dataclass(slots=True, frozen=True)
class QoSConstraints:
    """Z in the paper: admissible waiting and total delay."""

    max_wait: float  # Ω
    max_total_delay: float  # Δ (includes wait + in-vehicle detour)
    hard_capacity: bool = True  # enforce n_passengers ≤ capacity


@dataclass(slots=True, frozen=True)
class CostWeights:
    """Objective weights; paper minimizes sum of delays + big penalty for unserved."""

    unserved_penalty: float  # c_ko (large)
    # Optional knobs for extensions (keep zero to match paper):
    wait_weight: float = 1.0
    in_vehicle_delay_weight: float = 1.0


@dataclass(slots=True, frozen=True)
class AssignmentCost:
    """Cost terms for one (trip, vehicle) pairing."""

    total_delay: float  # ∑ δ_r over passengers+requests if executed by v
    num_requests: int


# --------- Routing & Plans ----------
@dataclass(slots=True, frozen=True)
class Leg:
    """One contiguous graph path segment with pre-aggregated metrics."""

    nodes: Sequence[int]
    distance_m: float
    travel_time_s: float


@dataclass(slots=True, frozen=True)
class RoutePlan:
    """Full ordered plan a vehicle will execute if assigned a trip."""

    legs: Sequence[Leg]
    total_distance_m: float
    total_travel_time_s: float


@dataclass(slots=True, frozen=True)
class Trip:
    """A feasible bundle of requests to serve with one vehicle."""

    tid: TripId
    requests: tuple[RequestId, ...]  # clique in RV-graph
    # Optional: store best route metrics per-vehicle lazily in RTV edges.


# --------- Graph Structures (RV / RTV) ----------
@dataclass(slots=True, frozen=True)
class RVEdge:
    """Edges between requests and requests/vehicles (pairwise shareability)."""

    # If r1-r2: both are feasible together in an empty virtual vehicle respecting QoS.
    r1: Optional[RequestId] = None
    r2: Optional[RequestId] = None
    # If r-v: request r can be feasibly inserted into vehicle v (with its current passengers).
    r: Optional[RequestId] = None
    v: Optional[VehicleId] = None
    pair_cost_delay_sum: Optional[float] = None  # δ_r1 + δ_r2 (for r1-r2)


@dataclass(slots=True, frozen=True)
class RVGraph:
    """Pairwise shareability as in the paper."""

    request_ids: tuple[RequestId, ...]
    vehicle_ids: tuple[VehicleId, ...]
    edges: tuple[RVEdge, ...]


@dataclass(slots=True, frozen=True)
class RTVEdge:
    """Trip ↔ Vehicle feasibility edges with cost attached."""

    trip: TripId
    vehicle: VehicleId
    cost: AssignmentCost


@dataclass(slots=True, frozen=True)
class RTVGraph:
    """Bipartite graph between Trips and Vehicles (plus implicit r∈T membership)."""

    trips: tuple[Trip, ...]
    vehicles: tuple[VehicleId, ...]
    # Membership: for each trip, the member requests:
    trip_membership: Mapping[TripId, tuple[RequestId, ...]]
    tv_edges: tuple[RTVEdge, ...]  # feasible (T, v) with costs


# --------- Results ----------
@dataclass(slots=True, frozen=True)
class VehicleAssignment:
    """Result of ILP: one vehicle to at most one trip (or idle)."""

    vehicle: VehicleId
    trip: Optional[TripId]  # None if vehicle idle
    plan: Optional[RoutePlan]  # Best plan for (vehicle, trip)


@dataclass(slots=True, frozen=True)
class BatchAssignment:
    """Full assignment outcome for a batch window."""

    assignments: tuple[VehicleAssignment, ...]
    served_requests: tuple[RequestId, ...]
    unserved_requests: tuple[RequestId, ...]


# --------- Service Loop Context ----------
@dataclass(slots=True)
class BatchState:
    """Single batch input snapshot (immutable to solvers)."""

    t_now: float
    constraints: QoSConstraints
    costs: CostWeights
    vehicles: tuple[VehicleState, ...]
    pool_requests: tuple[
        Request, ...
    ]  # includes unpicked past requests within Ω window


# --------- Abstract Oracles & Engines ----------
@runtime_checkable
class TravelTimeOracle(Protocol):
    """Fast travel-time/distance queries on the road network."""

    def time_and_distance(self, u: NodeRef, v: NodeRef) -> tuple[float, float]: ...
    def path(self, u: NodeRef, v: NodeRef) -> Leg: ...

    # Optional: batched versions for speed.


@runtime_checkable
class SingleVehicleRouter(Protocol):
    """
    Given vehicle state and a set of new requests, produce an optimal (or
    bounded-suboptimal) RoutePlan minimizing ∑ delays subject to QoS.
    Used by feasibility checks and cost computation.
    """

    def plan(
        self,
        vehicle: VehicleState,
        add_requests: Sequence[Request],
        constraints: QoSConstraints,
        t_now: float,
        oracle: TravelTimeOracle,
    ) -> Optional[RoutePlan]: ...

    # Return None if infeasible (violates Ω/Δ/capacity).


@runtime_checkable
class RVGraphBuilder(Protocol):
    """Build the pairwise shareability graph (requests↔requests, requests↔vehicles)."""

    def build(
        self, batch: BatchState, oracle: TravelTimeOracle, router: SingleVehicleRouter
    ) -> RVGraph: ...


@runtime_checkable
class TripEnumerator(Protocol):
    """
    Enumerate feasible trips (cliques of requests) using the RV-graph,
    expanding by size with pruning as in the paper (SI Algorithm 1).
    """

    def enumerate(
        self,
        rv: RVGraph,
        batch: BatchState,
        oracle: TravelTimeOracle,
        router: SingleVehicleRouter,
    ) -> tuple[Trip, ...]: ...


@runtime_checkable
class RTVGraphBuilder(Protocol):
    """
    For each trip T and vehicle v that could execute T, compute an edge with cost.
    Cost is ∑ delays (over onboard + T) induced by the best plan from router.
    """

    def build(
        self,
        trips: Sequence[Trip],
        batch: BatchState,
        oracle: TravelTimeOracle,
        router: SingleVehicleRouter,
    ) -> RTVGraph: ...


@runtime_checkable
class GreedyInitializer(Protocol):
    """
    Produce Σ_greedy by iterating trips in decreasing size, increasing cost,
    assigning each at most once per vehicle and once per request.
    """

    def seed(self, rtv: RTVGraph, costs: CostWeights) -> BatchAssignment: ...


@runtime_checkable
class AssignmentOptimizer(Protocol):
    """
    ILP: binary ε_{i,j} for (trip_i, vehicle_j) and χ_k for unserved requests.
    Minimize ∑ c_{i,j} ε_{i,j} + ∑ c_ko χ_k with
      - each vehicle ≤ 1 trip
      - each request served by exactly one chosen trip or marked unserved
    Supports anytime behavior via time/optimality-gap budgets.
    """

    def solve(
        self,
        rtv: RTVGraph,
        seed: BatchAssignment,
        costs: CostWeights,
        constraints: QoSConstraints,
        time_limit_s: Optional[float] = None,
        mip_gap: Optional[float] = None,
    ) -> BatchAssignment: ...


@runtime_checkable
class Rebalancer(Protocol):
    """
    Linear assignment of idle vehicles to ignored requests (Algorithm 4 in SI),
    minimizing dispatch times; may leave some vehicles idle.
    """

    def rebalance(
        self, batch: BatchState, result: BatchAssignment, oracle: TravelTimeOracle
    ) -> Mapping[VehicleId, NodeRef]: ...

    # Returns target positions for idle vehicles.


# --------- Top-level Coordinator ----------
@runtime_checkable
class BatchPlanner(Protocol):
    """
    Orchestrates the four phases per batch window:
      1) RV graph
      2) Trip enumeration & RTV graph
      3) ILP assignment (seeded by greedy)
      4) Rebalancing of idle vehicles
    """

    def plan_batch(
        self,
        batch: BatchState,
        oracle: TravelTimeOracle,
        router: SingleVehicleRouter,
        rv_builder: RVGraphBuilder,
        trip_enum: TripEnumerator,
        rtv_builder: RTVGraphBuilder,
        greedy: GreedyInitializer,
        optimizer: AssignmentOptimizer,
        rebalancer: Rebalancer,
        *,
        time_limit_s: Optional[float] = None,
        mip_gap: Optional[float] = None,
    ) -> tuple[BatchAssignment, Mapping[VehicleId, NodeRef]]: ...
