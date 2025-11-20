from dataclasses import dataclass, field
from src.interfaces import (
    FeasibleRoute,
)
from src.interfaces.data_models import TripId


# TripId = FrozenSet[str]


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

    # NEW: Reverse index for fast vehicle lookup O(1)
    vehicle_request: dict[str, set[str]] = field(default_factory=dict)

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
        # NEW: Reverse link: Vehicle -> Request
        self.vehicle_request.setdefault(v_id, set()).add(r_id)

    def get_requests_for_vehicle(self, v_id: str) -> set[str]:
        """Get all requests that vehicle v_id can feasibly serve."""
        return self.vehicle_request.get(v_id, set())

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

    # def requests_by_id(self) -> dict[str, Request]:
    #     """Reconstructs a mapping of request IDs to Request objects
    #     from the stored edges.

    #     Note: This assumes that the FeasibleRoute objects stored in
    #     rr_details and rv_details contain the original Request objects.
    #     """
    #     requests: dict[str, Request] = {}

    #     # From R-R edges
    #     for (r1_id, r2_id), route in self.rr_details.items():
    #         for req in route.requests:
    #             requests[req.id] = req

    #     # From R-V edges
    #     for (r_id, v_id), route in self.rv_details.items():
    #         for req in route.requests:
    #             requests[req.id] = req

    #     return requests


@dataclass
class RTVGraph:
    """
    Request-Trip-Vehicle (RTV) graph.

    Nodes:
      - Requests: implicit, given by their IDs.
      - Trips: represented as frozensets of request IDs (TripId).
      - Vehicles: represented by vehicle_id strings.

    Edges:
      - e(r, T): request r belongs to trip T.
      - e(T, v): vehicle v can feasibly serve trip T.

    We store:
      - request_to_trips: r_id -> set of TripId
      - trip_to_vehicles: TripId -> set of vehicle_id
      - trip_vehicle_details: (TripId, vehicle_id) -> FeasibleRoute
    """

    request_to_trips: dict[str, set[TripId]] = field(default_factory=dict)
    trip_to_vehicles: dict[TripId, set[str]] = field(default_factory=dict)
    trip_vehicle_details: dict[tuple[TripId, str], FeasibleRoute] = field(
        default_factory=dict
    )
    # NEW: Reverse index for fast vehicle lookup O(1)
    vehicle_to_trips: dict[str, set[TripId]] = field(default_factory=dict)

    # --- Edge helpers ---

    def add_trip_for_vehicle(
        self,
        trip: TripId,
        vehicle_id: str,
        route: FeasibleRoute,
    ) -> None:
        """
        Register that 'vehicle_id' can serve 'trip' with the given route.
        Also connects each request in the trip to the trip node.
        """
        # Trip -> vehicle
        self.trip_to_vehicles.setdefault(trip, set()).add(vehicle_id)

        # NEW: Vehicle -> trip
        self.vehicle_to_trips.setdefault(vehicle_id, set()).add(trip)

        # (Trip, vehicle) -> route
        self.trip_vehicle_details[(trip, vehicle_id)] = route

        # Request -> trip
        for r_id in trip:
            self.request_to_trips.setdefault(r_id, set()).add(trip)

    # --- Convenience getters ---
    def get_trips_for_vehicle(self, v_id: str) -> set[TripId]:
        """Get all trips that vehicle v_id can serve."""
        return self.vehicle_to_trips.get(v_id, set())

    def get_trips_for_request(self, r_id: str) -> set[TripId]:
        return self.request_to_trips.get(r_id, set())

    def get_vehicles_for_trip(self, trip: TripId) -> set[str]:
        return self.trip_to_vehicles.get(trip, set())

    def get_route(self, trip: TripId, vehicle_id: str) -> FeasibleRoute | None:
        return self.trip_vehicle_details.get((trip, vehicle_id))

    def get_cost(self, trip: TripId, vehicle_id: str) -> float:
        route = self.get_route(trip, vehicle_id)
        return route.cost if route else float("inf")
