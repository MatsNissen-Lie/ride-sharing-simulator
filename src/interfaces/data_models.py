# ---------------------------------------------------------------------------
#
#  Minimalist Architectural Framework for Alonso-Mora et al. (2017)
#  This file contains all abstract interfaces and core data structures.
#
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
#  I. Core Data Entities and Services
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QoSConstraints:
    """Quality of Service (QoS) constraints for a request. Times are in seconds."""

    max_wait_time: float  # Maximum allowable wait time (Omega)
    max_total_delay: float  # Maximum allowable total delay (Delta)


@dataclass(frozen=True)
class Request:
    """A single, unassigned travel request."""

    id: str
    origin_node: int  # Graph node ID of pickup
    destination_node: int  # Graph node ID of drop-off
    request_time: float  # Timestamp of request (t_r^r)

    # Pre-calculated earliest drop-off time
    # t_r^* = t_r^r + tau(o_r, d_r)
    earliest_dropoff_time: float


@dataclass
class Passenger:
    """A request that is currently in a vehicle's possession."""

    original_request: Request
    pickup_time: float  # The actual time t_r^p

    # The drop-off location (node) is in original_request.destination_node
    # The drop-off constraints (max_total_delay) are in original_request


@dataclass
class Vehicle:
    """Represents the current state of a fleet vehicle."""

    id: str
    current_node: int  # Last known graph node ID
    capacity: int  # Max passengers (u)

    # Passengers currently onboard (P_v in the paper)
    passengers_onboard: set[Passenger]

    # The vehicle's current committed plan
    # This is 'None' if the vehicle is idle.
    current_route: Optional["Route"] = None

    @property
    def current_occupancy(self) -> int:
        return len(self.passengers_onboard)

    @property
    def is_idle(self) -> bool:
        return self.current_route is None and self.current_occupancy == 0


@dataclass(frozen=True)
class Stop:
    """A single pickup or drop-off action in a route."""

    node: int  # The graph node of the stop
    is_pickup: bool  # True for pickup, False for drop-off
    request_id: str  # Identifies which request this stop serves


@dataclass(frozen=True)
class Route:
    """An ordered sequence of stops for a single vehicle."""

    # This sequence defines the vehicle's path.
    stops: list[Stop]

    # Pre-computed metrics for this specific route
    total_travel_time: float
    total_travel_distance: float

    # The final cost (sum of delays, sum(delta_r)) for this route
    # This is the 'c_ij' value for the ILP
    total_delay_cost: float


@dataclass(frozen=True)
class Path:
    """The result of a shortest-path query."""

    nodes: list[int]
    travel_time: float  # Travel time in seconds
    travel_distance: float  # Travel distance in meters


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


@dataclass(frozen=True)
class AssignmentResult:
    """The output of the main assignment ILP."""

    assignments: list[Tuple[Vehicle, set[Request], Route]]
    unassigned_requests: set[Request]
    total_cost: float


@dataclass(frozen=True)
class RebalancingResult:
    """The output of the rebalancing LP."""

    assignments: list[Tuple[Vehicle, Request]]
