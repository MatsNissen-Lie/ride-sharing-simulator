# ---------------------------------------------------------------------------
#
#  Minimalist Architectural Framework for Alonso-Mora et al. (2017)
#  This file contains all abstract interfaces and core data structures.
#
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple


# ---------------------------------------------------------------------------
#  I. Core Data Entities and Services
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QoSConstraints:
    """Quality of Service (QoS) constraints for a request. Times are in seconds."""

    max_wait_time: float  # Maximum allowable wait time (Omega)
    max_total_delay: float  # Maximum allowable total delay (Delta)


@dataclass(frozen=True)
class MetaRequest:
    """A request submitted by a user"""

    id: str
    origin_node: int  # Graph node ID of pickup
    destination_node: int  # Graph node ID of drop-off
    request_time: float  # Timestamp of request (t_r^r)


@dataclass(frozen=True)
class Request(MetaRequest):
    """A single, unassigned travel request."""

    # id: str
    # origin_node: int  # Graph node ID of pickup
    # destination_node: int  # Graph node ID of drop-off
    # request_time: float  # Timestamp of request (t_r^r)
    # Pre-calculated earliest drop-off time
    # t_r^* = t_r^r + tau(o_r, d_r) aka request_time + shortest_path_travel_time
    earliest_dropoff_time: float

    promised_pickup_time: float | None = (
        None  # Assigned pickup time (t_r^p), but not committed (aka not picked up yet)
    )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Request) and self.id == other.id


@dataclass(frozen=True)
class Passenger:
    """A request that is currently in a vehicle's possession."""

    original_request: Request
    pickup_time: float  # The actual time t_r^p

    # The drop-off location (node) is in original_request.destination_node
    # The drop-off constraints (max_total_delay) are in original_request


# In src/interfaces/data_models.py


@dataclass
class Vehicle:
    """
    The main, mutable 'source of truth' for a vehicle in the fleet.
    This object is ONLY managed by the main controller and simulation state.
    It should NEVER be passed to planners or graph builders.
    """

    id: str
    capacity: int

    # --- Live State ---

    # Passengers currently onboard
    passengers_onboard: set[Passenger]

    # The last *stop* node the vehicle visited.
    # If idle, this is its last known location.
    last_visited_node: int

    # The simulation time it arrived at/left 'last_visited_node'
    last_visited_time: float

    # --- Intent / Future Plan ---

    # The vehicle's committed plan
    # This is 'None' if the vehicle is idle.
    current_route: Optional["Route"] = None

    # The simulation time this route was assigned
    route_assignment_time: Optional[float] = None

    @property
    def current_occupancy(self) -> int:
        return len(self.passengers_onboard)

    @property
    def is_idle(self) -> bool:
        # A vehicle is idle if it has no route AND no passengers
        return self.current_route is None and self.current_occupancy == 0


@dataclass(frozen=True)
class VehicleAvailability:
    """
    An immutable, read-only snapshot of a vehicle's future availability.
    This is the *only* vehicle data passed to the graph builders
    and route planners.
    """

    # ID to link back to the real vehicle
    vehicle_id: str

    # The vehicle's total capacity
    capacity: int

    # --- Availability State ---

    # The node from which a *new* route would begin
    available_at_node: int

    # The future simulation time when the vehicle arrives at that node
    available_from_time: float

    # --- Passenger State at Availability Time ---

    # The set of passengers who will *still* be on board
    # at 'available_from_time'
    passengers_onboard: frozenset[Passenger]

    # The list of *remaining* stops the vehicle is *still* required to visit from its old route.
    remaining_stops_to_serve: tuple["Stop", ...]

    # make this from a Vehicle
    @staticmethod
    def from_vehicle(vehicle: Vehicle, sim_time: float) -> "VehicleAvailability":
        if vehicle.is_idle:
            return VehicleAvailability(
                vehicle_id=vehicle.id,
                capacity=vehicle.capacity,
                available_at_node=vehicle.last_visited_node,
                available_from_time=sim_time,
                passengers_onboard=frozenset(),
                remaining_stops_to_serve=(),
            )
        else:
            raise NotImplementedError(
                "Conversion from non-idle Vehicle not implemented."
            )


@dataclass(frozen=True)
class Stop:
    """A single pickup or drop-off action in a route."""

    node: int  # The graph node of the stop
    is_pickup: bool  # True for pickup, False for drop-off
    request_id: str  # Identifies which request this stop serves

    # TODO perhaps the stops that have been calculated pickup time should be saved. However we also want to have stops that are not yet assigned.
    # But a route could maybe contain the scheduled times for each stop?


@dataclass(frozen=True)
class Route:
    """An ordered sequence of stops for a single vehicle."""

    # vehicle_id: str
    # route_assignment_time: float
    # This sequence defines the vehicle's path.
    # stop_times: tuple[float, ...]  # Scheduled arrival time at each stop
    # lets keep it simple for now

    stops: tuple[Stop, ...]

    # Pre-computed metrics for this specific route
    total_travel_time: float
    total_travel_distance: float

    # The final cost (sum of delays, sum(delta_r)) for this route
    # This is the 'c_ij' value for the ILP
    total_delay_cost: float


@dataclass(frozen=True)
class NetworkPath:
    """The result of a shortest-path query from the network service."""

    # travel_times: tuple[float, ...]  # time between each consecutive node

    travel_time: float  # Travel time in seconds
    travel_distance: float  # Travel distance in meters


@dataclass(frozen=True)
class NetworkPathExtended(NetworkPath):
    """The result of a shortest-path query from the network service."""

    nodes: tuple[int, ...]
    travel_times: tuple[float, ...]  # time between each consecutive node


@dataclass(frozen=True)
class FeasibilityResult:
    """Base class for the route planner's response."""

    is_feasible: bool = field(init=False)


@dataclass(frozen=True)
class FeasibleRoute(FeasibilityResult):
    """Returned when a valid route is found."""

    route: Route  # The optimal route (sequence of stops)
    cost: float  # The resulting sum of delays (sum(delta_r))

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_feasible", True)


@dataclass(frozen=True)
class Infeasible(FeasibilityResult):
    """Returned when no valid route exists."""

    reason: str  # e.g., "Violates max_wait", "Violates capacity"

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_feasible", False)


TripId = FrozenSet[str]


@dataclass(frozen=True)
class Assignment:
    vehicle_id: str
    trip_id: "TripId"
    route: Route
    cost: float


@dataclass(frozen=True)
class AssignmentResult:
    """The output of the main assignment ILP."""

    assignments: set[Assignment]
    unassigned_requests: set[Request]
    total_cost: float
    is_optimal: bool


@dataclass(frozen=True)
class RebalancingResult:
    """The output of the rebalancing LP."""

    assignments: list[Tuple[VehicleAvailability, int]]  # (vehicle, target_node)
