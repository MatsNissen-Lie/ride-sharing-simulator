from abc import ABC, abstractmethod
from functools import lru_cache

from src.interfaces.data_models import (
    AssignmentResult,
    FeasibilityResult,
    NetworkPath,
    NetworkPathExtended,
    RebalancingResult,
    Request,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph, RVGraph


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

    @abstractmethod
    def get_shortest_path_with_travel_time(
        self, origin_node: int, destination_node: int
    ) -> NetworkPathExtended | None:
        """
        Returns the pre-computed fastest path between two nodes,
        along with travel times between each consecutive node.
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
# RVGraph = Any  # TODO

# A type alias for a bipartite graph object (e.g., networkx.Graph)
# One set of nodes is 'Trip' objects (which are sets of Requests)
# The other set of nodes is 'Vehicle' objects.
# RTVGraph = Any  # TODO


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
        self,
        rv_graph: RVGraph,
        vehicles: set[VehicleAvailability],
        requests: set[Request],
        max_trip_size: int,
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


from abc import ABC, abstractmethod
from typing import Set

# Assuming these imports exist based on your previous code context
from src.interfaces import (
    AssignmentResult,
    Request,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph


class IAssignmentSolver(ABC):
    """
    Interface for the optimization module that assigns vehicles to trips.
    """

    @abstractmethod
    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: Set[Request],
        vehicles: Set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,  # <-- Added as requested
    ) -> AssignmentResult:
        """
        Formulates and solves the ILP described in Section 4.4 (Optimal Assignment).

        This method maps the RTVGraph data to the following mathematical model:

        Sets and Parameters:
        - R_t: Set of open requests (from `requests`).
        - V: Set of available vehicles (from `vehicles`).
        - E_{TV}: Feasible edges e(T, v) (from `rtv_graph`).
        - c_{T,v}: Cost of assigning vehicle v to trip T (from `rtv_graph`).
        - chi_r: Penalty for not serving request r (dynamic).

        Decision Variables:
        - x_{T,v} (binary): 1 if vehicle v is assigned to trip T, 0 otherwise.
        - y_r (binary): 1 if request r is NOT served in this batch, 0 otherwise.

        Objective Function:
        - min sum(c_{T,v} * x_{T,v}) + sum(chi_r * y_r)

        Constraints:
        1. Assign each request at most once:
           sum(x_{T,v} | r in T) + y_r = 1    for all r in R_t

        2. Assign each vehicle at most once:
           sum(x_{T,v}) <= 1                  for all v in V

        Requirements:
        - Must provide a 'warm start' (typically a greedy assignment).
        - Must adhere to `solver_timeout` to ensure 'anytime' performance constraints.
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
