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
from typing import cast

from config import get_logger
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    IRVGraphBuilder,
    FeasibleRoute,
    Passenger,
    Request,
)
from src.interfaces.graphs import RVGraph


logger = get_logger(__name__)


# --- The Builder class is now simpler ---


class RVGraphBuilder(IRVGraphBuilder):
    """Concrete implementation of :class:`IRVGraphBuilder`.
    (Docstring unchanged)
    """

    def __init__(
        self,
        route_planner: ISingleVehicleRoutePlanner,
    ) -> None:
        super().__init__(route_planner)

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
                    # TODO REMOVE THIS AFTER TESTING FOR PERFORMANCE
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
