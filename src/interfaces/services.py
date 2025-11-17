from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from src.interfaces.data_models import (
    AssignmentResult,
    FeasibilityResult,
    NetworkPath,
    RebalancingResult,
    Request,
    VehicleAvailability,
)


class INetworkService(ABC):
    """Abstract interface for all road network and travel time queries."""

    @abstractmethod
    @lru_cache(maxsize=200_000)
    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:
        """
        Returns the pre-computed fastest path between two nodes.
        This provides the baseline travel time tau(o_r, d_r).
        """
        pass


# ---------------------------------------------------------------------------
#  II. The Core "Engine": Single-Vehicle Routing (DARP)
# ---------------------------------------------------------------------------


class ISingleVehicleRoutePlanner(ABC):
    """
    Solves the single-vehicle DARP.
    This encapsulates the 'travel(v, R_v)' function.
    """

    def __init__(self, network_service: INetworkService):
        self._network = network_service
        pass

    @abstractmethod
    def check_feasibility(
        self,
        requests_to_add: set[Request],
        vehicle: VehicleAvailability | None = None,
    ) -> FeasibilityResult:
        """
        Solves the DARP for a single vehicle.
        Attempts to find an optimal route (a permutation of stops) that serves:
        1. All passengers already onboard (vehicle.passengers_onboard)
        2. All new requests (requests_to_add)
        passwhile respecting all constraints.
        """
        pass


# ---------------------------------------------------------------------------
#  III. The Four-Step Assignment Pipeline: Abstract Interfaces
# ---------------------------------------------------------------------------

# A type alias for a graph object, e.g., networkx.Graph
# Nodes are Request and Vehicle objects.
RVGraph = Any  # TODO

# A type alias for a bipartite graph object (e.g., networkx.Graph)
# One set of nodes is 'Trip' objects (which are sets of Requests)
# The other set of nodes is 'Vehicle' objects.
RTVGraph = Any  # TODO


class IRVGraphBuilder(ABC):

    def __init__(self, route_planner: ISingleVehicleRoutePlanner):
        self._route_planner = route_planner
        pass

    @abstractmethod
    def build_graph(
        self, requests: set[Request], vehicles: set[VehicleAvailability]
    ) -> RVGraph:
        """
        Constructs the RV-graph by checking two types of pairwise shareability:

        1. Request-Request (r1, r2): Creates a virtual empty vehicle 'v_virtual'
           and checks if self._route_planner.check_feasibility(
               v_virtual, {r1, r2}
           ) is feasible.

        2. Request-Vehicle (r, v): Checks if self._route_planner.check_feasibility(
               v, {r}
           ) is feasible, given v's current state (passengers, location).
        """
        pass


class IRTVGraphBuilder(ABC):

    def __init__(self, route_planner: ISingleVehicleRoutePlanner):
        self._route_planner = route_planner
        pass

    @abstractmethod
    def build_graph(
        self, rv_graph: RVGraph, vehicles: set[VehicleAvailability], max_trip_size: int
    ) -> RTVGraph:
        """
        Builds the RTV-graph:

        1. Finds all cliques of size <= max_trip_size in the 'Request'
           subgraph of rv_graph. Each clique is a candidate 'Trip' T.

        2. For each candidate Trip T and each Vehicle v:
           a. Check feasibility:
              result = self._route_planner.check_feasibility(v, T.requests)

           b. If result.is_feasible:
              i.  Add an edge e(T, v) to the RTVGraph.
              ii. Store result.cost and result.route as attributes
                  of that edge.

        (This implementation can be optimized by checking subtrips first).
        """
        pass


class IAssignmentSolver(ABC):

    @abstractmethod
    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
    ) -> AssignmentResult:
        """
        Formulates and solves the ILP from Algorithm 1:

        - Variables:
          - epsilon_ij (binary) for each edge e(Trip_i, Vehicle_j) in RTVGraph.
          - chi_k (binary) for each Request_k in 'requests'.

        - Objective:
          - min sum(c_ij * epsilon_ij) + sum(c_ko * chi_k)
          - (c_ij is the 'cost' stored on the RTVGraph edge)

        - Constraints:
          - sum_i(epsilon_ij) <= 1  (Vehicle j assigned to at most one trip)
          - sum_i_j(epsilon_ij) + chi_k = 1 (Request k is in one assigned trip OR
                                           is unassigned, 'ignored')

        - Must provide a 'warm start' (greedy assignment) and adhere
          to the 'solver_timeout' to ensure 'anytime' performance.
        """
        pass


class IRebalancingSolver(ABC):

    def __init__(self, network_service: INetworkService):
        self._network = network_service
        pass

    @abstractmethod
    def solve(
        self, idle_vehicles: set[VehicleAvailability], unassigned_requests: set[Request]
    ) -> RebalancingResult:
        """
        Formulates and solves the LP from SI Appendix, Algorithm 4:

        - Objective: min sum(travel_time(v, r) * y_vr)
          - (Uses network_service to get simple travel times)

        - Variables: y_vr (binary/continuous) for each (v, r) pair.

        - Constraints (Standard assignment problem):
          - sum_r(y_vr) <= 1 (Vehicle v assigned to at most one request)
          - sum_v(y_vr) <= 1 (Request r assigned to at most one vehicle)
        """
        pass
