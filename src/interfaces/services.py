from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator
import numpy as np

from src.interfaces.data_models import (
    AssignmentResult,
    FeasibleRoute,
    FeasibilityResult,
    Route,
    NetworkPath,
    NetworkPathExtended,
    ReplayResult,
    Request,
    Vehicle,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph, RVGraph


class INetworkService(ABC):
    """Abstract interface for all road network and travel time queries."""

    is_nuclear: bool = False
    """Indicates if the cache is 'nuclear' (i.e., all-pairs precomputed)."""

    @abstractmethod
    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:
        """
        Returns the pre-computed fastest path between two nodes.
        This provides the baseline travel time tau(o_r, d_r).
        """
        pass

    @abstractmethod
    def get_shortest_path_with_travel_time(
        self, origin_node: int, destination_node: int
    ) -> NetworkPathExtended | None:
        """
        Returns the pre-computed fastest path between two nodes,
        along with travel times between each consecutive node.
        """
        pass

    def path_time_and_distance(
        self, origin_node: int, destination_node: int
    ) -> tuple[float, float] | None:
        """travel_time and travel_distance of get_shortest_path_with_travel_time,
        for callers that do not need the nodes. Implementations may answer
        without building the path; the numbers must be the same."""
        path = self.get_shortest_path_with_travel_time(origin_node, destination_node)
        return None if path is None else (path.travel_time, path.travel_distance)

    def path_steps(self, origin_node: int, destination_node: int) -> Iterator[tuple[int, float]]:
        """(node, travel time of the edge into it) along the path of
        get_shortest_path_with_travel_time after its first node, lazily."""
        path = self.get_shortest_path_with_travel_time(origin_node, destination_node)
        if path is not None:
            yield from zip(path.nodes[1:], path.travel_times)

    @abstractmethod
    def get_travel_time(self, origin: int, destination: int) -> float:
        """
        Returns the travel time between two nodes.
        Returns float('inf') if no path exists.
        """
        pass

    @abstractmethod
    def get_travel_time_matrix(
        self, origins: list[int], destinations: list[int]
    ) -> np.ndarray:
        """
        Returns a matrix of travel times between all origin-destination pairs.
        """
        raise NotImplementedError("get_travel_matrix not implemented in base class")

    @abstractmethod
    def prefetch_all_pairs(self, nodes: list[int]) -> None:
        """
        Eagerly prefetches all-pairs shortest paths for the given nodes.
        """
        raise NotImplementedError("prefetch_all_pairs not implemented in base class")

    @abstractmethod
    def clear_l1_cache(self) -> None:
        """
        Clears the manual cache of pre-fetched paths, if any.
        """
        pass

    @abstractmethod
    def get_travel_times_l1_cache_subset(
        self, nodes: Iterable[int]
    ) -> dict[tuple[int, int], float]:
        """
        Returns a dense dictionary of travel times for the given subset of nodes.
        Raises KeyError if any pair is missing from the L1 cache.
        """
        pass

    def update_cache(self, cache: Any) -> None:
        """
        Updates the L1 cache with a given time matrix.
        """
        raise NotImplementedError("update_cache not implemented in base class")

    @abstractmethod
    def get_matrix_and_mapping(self) -> tuple[np.ndarray, dict[int, int]]:
        """
        Returns the raw travel time matrix and a mapping from Node ID -> Matrix Index.
        Used for vectorized operations that require direct index access.
        """
        pass

    def get_distance_matrix(self) -> np.ndarray:
        """All-pairs shortest-path distances in metres, indexed by node id."""
        raise NotImplementedError("get_distance_matrix not implemented in base class")

    def get_l1_cache(self) -> Any:
        """
        Returns the internal L1 cache object, if any.
        """
        raise NotImplementedError("get_l1_cache not implemented in base class")


# ---------------------------------------------------------------------------
#  III. The assignment pipeline: swappable components
#
#  Each batch:  RV graph -> RTV graph -> assignment ILP -> rebalancing.
#  An IObjective decides what an assignment costs; an ITripSolver implements
#  travel(v, T). Implementations are chosen in src/experiment.py.
# ---------------------------------------------------------------------------


class IFleetReplay(ABC):
    """Moves a fleet along its routes to a batch time (the controller's projection)."""

    @abstractmethod
    def replay(
        self, fleet: list[Vehicle], t_now: float, requests: dict[str, Request]
    ) -> ReplayResult:
        """
        Serve the stops each vehicle reaches by t_now, place moving vehicles at
        the next node of their leg, and return every vehicle's availability,
        the promised pickup time of each request still on a route, and what
        was driven and delivered on the way. Picked-up requests leave
        `requests`; vehicles are changed in place. Statistics are the
        controller's to keep.
        """


class ITripSolver(ABC):
    """travel(v, T): the best feasible route for a vehicle serving a set of requests."""

    # True only if the solver serves a stop order exactly when the RV builder's
    # shortcuts say it can: hard pickup and drop-off deadlines, no waiting, and
    # leg times added in stop order from the l1 matrix. The RV builder then
    # computes edges of empty and one-passenger vehicles in closed form and
    # skips pairs no stop order can serve; otherwise it asks the solver for
    # every edge.
    matches_rv_closed_forms: bool = False

    @abstractmethod
    def solve(
        self, requests: Iterable[Request], vehicle: VehicleAvailability
    ) -> FeasibilityResult:
        """
        FeasibleRoute if every onboard passenger and new request can be served
        within their pickup and drop-off deadlines, else Infeasible. The route's
        cost is the solver's own ranking criterion; the assignment uses
        IObjective.trip_cost.
        """


class IZoneIndex(ABC):
    """A partition of the street graph into zones, with one representative node per zone."""

    zone_ids: np.ndarray  # all zone ids
    center_node: dict[int, int]  # zone id -> node used as a rebalancing destination

    @abstractmethod
    def zone(self, node: int) -> int:
        """Zone id of a graph node."""

    @abstractmethod
    def zones_of(self, nodes) -> np.ndarray:
        """Zone ids of many graph nodes."""


class IDemandModel(ABC):
    """Expected requests per zone: a forecast of where and when riders appear."""

    @abstractmethod
    def expected_requests(self, t_start: float, t_end: float) -> dict[int, float]:
        """Expected number of new requests per zone id in [t_start, t_end) (simulation seconds)."""


class IFareModel(ABC):
    """The price a rider pays for a trip, quoted when the request arrives."""

    @abstractmethod
    def price(self, origin: int, destination: int) -> float:
        """Fare in USD for a ride from `origin` to `destination` (graph node ids)."""


class IObjective(ABC):
    """What the assignment ILP minimises: trip costs and penalties for unserved requests."""

    @abstractmethod
    def trip_cost(
        self,
        route: FeasibleRoute,
        vehicle: VehicleAvailability,
        requests: list[Request],
    ) -> float:
        """Cost c_{T,v} of vehicle v serving trip T along `route`."""

    @abstractmethod
    def unserved_penalty(self, request: Request, current_time: float) -> float:
        """Cost chi_r of leaving request r unassigned in this batch."""

    @abstractmethod
    def rebalancing_cost(
        self, travel_time: float | np.ndarray, distance_m: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Cost of an empty move (used by solvers that rebalance in the assignment).
        Must work element-wise on arrays as well as on floats, with the same
        result per element: solvers score many moves at once.
        """


class IRVGraphBuilder(ABC):
    """Pairwise shareability: request-request and request-vehicle edges."""

    @abstractmethod
    def build_graph(
        self, requests: set[Request], vehicles: set[VehicleAvailability]
    ) -> RVGraph:
        """
        1. Request-Request (r1, r2): whether an ideal virtual vehicle at o_r1
           or o_r2 could serve both requests within their constraints.
        2. Request-Vehicle (r, v): whether travel(v, {r}) is feasible given
           v's current state (passengers, location).
        """


class IRTVGraphBuilder(ABC):
    """Feasible trips (request sets) and the vehicles that can serve them."""

    @abstractmethod
    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: set[VehicleAvailability],
        requests: set[Request],
    ) -> RTVGraph:
        """
        Every trip T with an edge e(T, v) for each vehicle v where travel(v, T)
        is feasible. Candidates come from cliques of the RV graph; by the
        sub-trip lemma a trip is only checked if all its sub-trips are feasible.
        """


class IAssignmentSolver(ABC):
    """The assignment ILP (SI Appendix, Section 4.4)."""

    @abstractmethod
    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,
    ) -> AssignmentResult:
        """
        min  sum c_{T,v} x_{T,v} + sum chi_r y_r
        s.t. sum_{T contains r} x_{T,v} + y_r = 1   for every request r
             sum_T x_{T,v} <= 1                     for every vehicle v
        with c and chi from the solver's IObjective, x and y binary, stopping at
        `solver_timeout` seconds.
        """


class IRebalancingSolver(ABC):
    """Where idle vehicles drive after the assignment."""

    @abstractmethod
    def solve(
        self,
        idle_vehicles: list[VehicleAvailability],
        unassigned_requests: list[Request],
        t_now: float,
        en_route: list[VehicleAvailability] = (),
    ) -> dict[str, Route]:
        """
        Rebalancing route per idle vehicle id; vehicles left out stay where
        they are. `en_route` are vehicles still driving to an earlier
        rebalancing destination (their remaining stops); they keep going
        unless a route is returned for them (an empty route stops them).
        """


class IVehicleValue(ABC):
    """
    Future value of a vehicle that becomes free at `node` at time `t`, in the
    objective's units as a saving (higher is better). Joint assignment solvers
    weigh every option of a vehicle by where and when it leaves the vehicle,
    which makes the assignment look past the current batch.
    """

    def prepare(
        self,
        vehicles: Iterable[VehicleAvailability],
        requests: Iterable[Request],
        t_now: float,
    ) -> None:
        """Called once per batch before `values` (to plan or learn); default: nothing."""

    @abstractmethod
    def values(self, nodes: np.ndarray, times: np.ndarray, t_now: float) -> np.ndarray:
        """Value of a free vehicle for each (node, time) pair."""


class IFleetPlan(IVehicleValue):
    """A vehicle value that also plans empty moves between zones (e.g. a fleet MPC)."""

    @property
    @abstractmethod
    def t_solved(self) -> float | None:
        """Simulation time of the plan in `moves` (None before the first plan)."""

    @property
    @abstractmethod
    def moves(self) -> np.ndarray:
        """Planned empty moves now: vehicles from zone i to zone j, in zone_ids order."""
