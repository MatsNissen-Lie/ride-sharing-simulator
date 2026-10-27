from abc import ABC, abstractmethod
from typing import Any, Iterable
import numpy as np

from src.interfaces.data_models import (
    AssignmentResult,
    FeasibleRoute,
    FeasibilityResult,
    Route,
    NetworkPath,
    NetworkPathExtended,
    Request,
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


class ITripSolver(ABC):
    """travel(v, T): the best feasible route for a vehicle serving a set of requests."""

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
    ) -> dict[str, Route]:
        """Rebalancing route per vehicle id; vehicles left out stay where they are."""
