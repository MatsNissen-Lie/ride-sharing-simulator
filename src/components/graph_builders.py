"""
rv_graph_builder
=================

This module provides an implementation of the pairwise request-vehicle (RV)
graph described in the supplemental material of Alonso-Mora et al. (2017).
The RV-graph captures which pairs of requests may be served together by a
single vehicle and which individual requests may be served by a given
vehicle. Each edge in the graph is annotated with the cost returned from
the single vehicle route planner, allowing subsequent steps (such as trip
construction and assignment) to reason about the feasibility and cost of
possible matches.

The implementation here does not impose any particular limit on the number
of edges per node; however, users may choose to prune the graph to speed up
subsequent computations. See the paper for details on how such pruning can
trade off optimality for runtime.

The central class exported by this module is :class:`RVGraphBuilder`, which
implements the :class:`IRVGraphBuilder` interface from ``src.interfaces``.
It relies on a provided :class:`RoutePlanner` to evaluate feasibility of
pairings. The resulting graph is returned as an instance of
:class:`RVGraph` defined below.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

from config import get_logger
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    IRVGraphBuilder,
    FeasibleRoute,
    Infeasible,
    Passenger,
    Request,
)


logger = get_logger(__name__)


@dataclass
class RVGraph:
    """Simple container for the pairwise request-vehicle graph.

    The graph consists of two kinds of edges, stored separately:

    * **Request-Request (R-R):** An undirected edge exists if two requests
      are pairwise shareable.
      - `request_request` stores the adjacency list (a dict of sets).
      - `rr_details` stores the full FeasibleRoute object for the pair.

    * **Request-Vehicle (R-V):** A directed edge exists if a vehicle
      can feasibly serve a request.
      - `request_vehicle` stores the adjacency list (a dict of sets).
      - `rv_details` stores the full FeasibleRoute object for the (r, v) pair.

    Getters are provided to access graph information.
    """

    # Adjacency lists (optimized for graph traversal / clique finding)
    request_request: dict[str, set[str]] = field(default_factory=dict)
    request_vehicle: dict[str, set[str]] = field(default_factory=dict)

    # Route details (single source of truth for cost and route)
    rr_details: dict[tuple[str, str], FeasibleRoute] = field(default_factory=dict)
    rv_details: dict[tuple[str, str], FeasibleRoute] = field(default_factory=dict)

    def add_rr_edge(
        self,
        r1_id: str,
        r2_id: str,
        route: FeasibleRoute,
    ) -> None:
        """Add an undirected request-request edge to the graph."""
        self.request_request.setdefault(r1_id, set()).add(r2_id)
        self.request_request.setdefault(r2_id, set()).add(r1_id)

        # Store route object for both directions
        self.rr_details[(r1_id, r2_id)] = route
        self.rr_details[(r2_id, r1_id)] = route

    def add_rv_edge(
        self,
        r_id: str,
        v_id: str,
        route: FeasibleRoute,
    ) -> None:
        """Add a request-vehicle edge to the graph."""
        self.request_vehicle.setdefault(r_id, set()).add(v_id)
        self.rv_details[(r_id, v_id)] = route

    # --- Getters ---

    def get_all_request_nodes(self) -> set[str]:
        """Returns a set of all request IDs in the graph."""
        # Combine keys from both adjacency lists
        return set(self.request_request.keys()) | set(self.request_vehicle.keys())

    def get_rr_neighbors(self, r_id: str) -> set[str]:
        """Get all requests that are pairwise shareable with r_id."""
        return self.request_request.get(r_id, set())

    def get_rv_neighbors(self, r_id: str) -> set[str]:
        """Get all vehicles that can feasibly serve r_id."""
        return self.request_vehicle.get(r_id, set())

    def get_rr_route(self, r1_id: str, r2_id: str) -> FeasibleRoute | None:
        """Get the FeasibleRoute object for an R-R pair."""
        return self.rr_details.get((r1_id, r2_id))

    def get_rv_route(self, r_id: str, v_id: str) -> FeasibleRoute | None:
        """Get the FeasibleRoute object for an R-V pair."""
        return self.rv_details.get((r_id, v_id))

    def get_rr_cost(self, r1_id: str, r2_id: str) -> float:
        """Get the cost for an R-R pair."""
        route = self.get_rr_route(r1_id, r2_id)
        return route.cost if route else float("inf")

    def get_rv_cost(self, r_id: str, v_id: str) -> float:
        """Get the cost for an R-V pair."""
        route = self.get_rv_route(r_id, v_id)
        return route.cost if route else float("inf")


# --- The Builder class is now simpler ---


class RVGraphBuilder(IRVGraphBuilder):
    """Concrete implementation of :class:`IRVGraphBuilder`.
    (Docstring unchanged)
    """

    def __init__(
        self,
        route_planner: ISingleVehicleRoutePlanner,
        *,
        max_virtual_capacity: int | None = None,
    ) -> None:
        super().__init__(route_planner)
        self._max_virtual_capacity = max_virtual_capacity

    def build_graph(
        self,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
    ) -> RVGraph:
        """Construct the request-vehicle graph for the given requests and vehicles.
        (Docstring unchanged)
        """
        graph = RVGraph()

        # Pre-populate the adjacency lists with empty sets for all requests
        # This ensures get_all_request_nodes() is accurate even if a request
        # has no feasible edges.
        for req in requests:
            graph.request_request.setdefault(req.id, set())
            graph.request_vehicle.setdefault(req.id, set())

        req_list = list(requests)

        # --- Build R-R Edges ---
        for i in range(len(req_list)):
            for j in range(i + 1, len(req_list)):
                r1 = req_list[i]
                r2 = req_list[j]

                result = self._route_planner.check_feasibility(
                    {r1, r2}
                )  # Vehicle capacity is assumed to be at least 2

                if result.is_feasible:
                    feasible = cast(FeasibleRoute, result)
                    # Simplified: Just pass the route object
                    graph.add_rr_edge(r1.id, r2.id, feasible)

        # --- Build R-V Edges ---
        for req in requests:
            for vehicle in vehicles:
                if self._request_already_onboard(req, vehicle.passengers_onboard):
                    # This should not happen if request pool is correct
                    raise ValueError(
                        f"Request {req.id} is already onboard vehicle {vehicle.vehicle_id}"
                    )

                result = self._route_planner.check_feasibility({req}, vehicle)

                if result.is_feasible:
                    feasible = cast(FeasibleRoute, result)
                    vehicle_id = self._vehicle_identifier(vehicle)
                    # Simplified: Just pass the route object
                    graph.add_rv_edge(req.id, vehicle_id, feasible)

        return graph

    def _resolve_virtual_capacity(self, vehicles: set[VehicleAvailability]) -> int:
        if self._max_virtual_capacity is not None:
            return max(2, self._max_virtual_capacity)
        if not vehicles:
            return 2  # Default to 2 if no vehicles are present
        max_capacity = max(getattr(vehicle, "capacity", 2) for vehicle in vehicles)
        return max(2, max_capacity)

    @staticmethod
    def _request_already_onboard(
        request: Request, passengers: Iterable[Passenger] | None
    ) -> bool:
        if passengers is None:
            return False
        for passenger in passengers or []:
            if request.id == passenger.original_request.id:
                return True
        return False

    @staticmethod
    def _vehicle_identifier(vehicle: VehicleAvailability) -> str:
        return vehicle.vehicle_id
