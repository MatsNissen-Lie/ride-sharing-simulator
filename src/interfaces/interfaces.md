# A Minimalist Architectural Framework for the Alonso-Mora et al. High-Capacity Ride-Sharing Algorithm

This report outlines a minimalist, object-oriented structure for implementing the high-capacity ride-sharing algorithm presented by Alonso-Mora et al. [1]. The focus is on defining stable data classes and abstract interfaces that isolate system components, manage complexity, and map directly to the paper's methodology. The architecture is designed to be scalable, testable, and robust, respecting the data basis of street networks and ride requests [2].

## I. Core Data Entities and Services

The foundation of the system is built on immutable data structures (dataclasses) representing the core nouns of the ride-sharing domain, and a foundational service for network calculations.

### Core Entities

**1. Request** – Represents a customer's submitted travel desire before assignment. It is the fundamental unit of demand.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    """A single, unassigned travel request."""

    id: str
    origin_node: int          # Graph node ID of pickup
    destination_node: int     # Graph node ID of drop-off
    request_time: float       # Timestamp of request (t_r^r)
    earliest_dropoff_time: float  # t_r^* = t_r^r + tau(o_r, d_r)

    # --- Quality of Service (QoS) Constraints ---
    max_wait_time: float      # Max wait (Omega)
    max_total_delay: float    # Max delay (Delta)
```

**2. Passenger** – Represents a request that has been picked up and is currently onboard a vehicle. It is stateful and distinct from a `Request` [2].

```python
@dataclass
class Passenger:
    """A request that is currently in a vehicle's possession."""

    original_request: Request
    pickup_time: float        # The actual time t_r^p
    # Drop-off node and constraints live on original_request
```

**3. Vehicle** – Captures the state of a fleet vehicle, including location, occupancy, and the current plan.

```python
from typing import Optional, Set


@dataclass
class Vehicle:
    """Represents the current state of a fleet vehicle."""

    id: str
    current_node: int                # Last known graph node ID
    capacity: int                    # Max passengers (u)
    passengers_onboard: Set[Passenger]
    current_route: Optional["Route"] = None  # None when idle

    @property
    def current_occupancy(self) -> int:
        return len(self.passengers_onboard)
```
[16] Experiment timing and batching parameters.

[17] Batch orchestration overview.
1. `T = 0s`: Fetch current state (`requests = RequestPool.get_all()`, `vehicles = VehicleFleet.get_all()`).
2. `T = 1s`: `rv_graph = rv_builder.build_graph(requests, vehicles)`.
3. `T = 5s`: `rtv_graph = rtv_builder.build_graph(rv_graph, vehicles, max_trip_size=10)`.
4. `T = 10s`: `assignment_result = assignment_solver.solve(rtv_graph, requests, vehicles, solver_timeout=19.0)`.
5. `T = 29s`: Extract `idle_vehicles` and `unassigned_requests` from `assignment_result`.
6. `T = 29s`: `rebalance_result = rebalancing_solver.solve(idle_vehicles, unassigned_requests)`.
7. `T = 30s`: Commit results to `VehicleFleet` and `RequestPool`, then sleep until the next batch window.

### "Anytime" vs. "Batch" Orchestration

- The real-time requirement is met by the strict batch window (30 seconds) [16].
- The anytime property applies to `IAssignmentSolver` (Step 3), which begins with a fast greedy warm start [3].
- The controller gives the ILP solver a fixed time budget (e.g., 19 seconds). Upon timeout, it uses the best solution found so far [3].

### Data Flow for a Single Request

1. **Ingestion** – User submits a ride; a `Request` is added to `RequestPool`.
2. **Batching** – The request enters the active set at the next batch.
3. **Step 1 (RV)** – Request becomes a node in the RV-graph with shareable edges.
4. **Step 2 (RTV)** – Request participates in candidate trips connected to feasible vehicles.
5. **Step 3 (ILP)** – If an `epsilon_{i,j}` variable becomes 1, the request is assigned; otherwise `chi_k` becomes 1 and the request remains unassigned.
6. **Step 4 (Rebalance)** – Idle vehicles may be matched to unassigned requests; unsuccessful matches persist for the next batch, tightening their wait-time constraints.

### Architectural Recommendations

- **Parallelism** – `IRVGraphBuilder` and `IRTVGraphBuilder` are computational bottlenecks; distribute feasibility checks across worker pools to reduce runtime [16].
- **Solver abstraction** – `IAssignmentSolver` and `IRebalancingSolver` hide optimization library details (e.g., Gurobi, CPLEX).
- **Engine specialization** – Select `ISingleVehicleRoutePlanner` strategies at runtime: exhaustive search for `u <= 4`, heuristic approaches for higher capacities [13].

## V. Conclusion

This architectural blueprint defines dataclasses (`Request`, `Vehicle`, `Passenger`, `Route`) and abstract interfaces (`INetworkService`, `ISingleVehicleRoutePlanner`, `IRVGraphBuilder`, `IRTVGraphBuilder`, `IAssignmentSolver`, `IRebalancingSolver`) that tame the algorithm's complexity. The design decouples:

- Single-vehicle routing (`ISingleVehicleRoutePlanner`) from global assignment.
- Candidate generation (Steps 1–2) from centralized optimization (Step 3).
- Fleet rebalancing (Step 4) from service assignment.

The resulting structure is clear, scalable, and testable, supporting the real-time, high-capacity demands described by Alonso-Mora et al. [2].

---

**References**

[1] Alonso-Mora et al., 2017. Dynamic ride-sharing in high-capacity systems.

[2] Paper methodology notes and supplemental material.

[3] Anytime optimal property and greedy warm-start strategy.

[5] Dynamic Dial-a-Ride Problem literature.

[8] Pickup and Delivery with Time Windows studies.

[13] Heuristic approaches for higher-capacity vehicles.

[16] Experiment timing and batching parameters.

[17] Batch orchestration overview.
A Minimalist Architectural Framework for the Alonso-Mora et al. High-Capacity Ride-Sharing Algorithm
This report outlines a minimalist, object-oriented structure for implementing the high-capacity ride-sharing algorithm presented by Alonso-Mora et al..1 The focus is on defining stable data classes and abstract interfaces that isolate system components, manage complexity, and map directly to the paper's methodology. The architecture is designed to be scalable, testable, and robust, respecting the data basis of street networks and ride requests.2

I. Core Data Entities and Services
The foundation of the system is built on a set of immutable data structures (dataclasses) representing the core "nouns" of the ride-sharing domain, and a foundational service for network calculations.

Core Entities
1. Request
A Request represents a customer's submitted travel desire, existing before a vehicle is assigned. It is the fundamental unit of demand.

```Python
Python@dataclass(frozen=True)
class Request:
    """A single, unassigned travel request."""
    id: str
    origin_node: int          # Graph node ID of pickup 
    destination_node: int     # Graph node ID of drop-off 
    request_time: float       # Timestamp of request (t_r^r)
    
    # Pre-calculated earliest drop-off time 
    # t_r^* = t_r^r + tau(o_r, d_r)
    earliest_dropoff_time: float  

    # --- Quality of Service (QoS) Constraints ---
    max_wait_time: float      # Max wait (Omega) 
    max_total_delay: float    # Max delay (Delta) 

```

2. Passenger
A Passenger represents a request that has been picked up and is currently onboard a vehicle. It is a stateful entity, distinct from a Request.2

```Python
Python@dataclass
class Passenger:
    """A request that is currently in a vehicle's possession."""
    original_request: Request
    pickup_time: float        # The actual time t_r^p
    
    # The drop-off location (node) is in original_request.destination_node
    # The drop-off constraints (max_total_delay) are in original_request

```

1. Vehicle
A Vehicle represents the state of a single unit in the fleet. This is the most complex stateful object, tracking location, occupancy, and its current plan.

```Python
@dataclass
class Vehicle:
    """Represents the current state of a fleet vehicle."""
    id: str
    current_node: int         # Last known graph node ID
    capacity: int             # Max passengers (u) [1, 2]
    
    # Passengers currently onboard (P_v in the paper) 
    passengers_onboard: set[Passenger]
    
    # The vehicle's current committed plan
    # This is 'None' if the vehicle is idle.
    current_route: Optional = None
    
    @property
    def current_occupancy(self) -> int:
        return len(self.passengers_onboard)
    
    @property
    def is_idle(self) -> bool:
        return self.current_route is None and self.current_occupancy == 0
```

Route and Network Primitives
The system's logic is built upon the concept of a Route as an explicit sequence of actions (Stop objects) that a vehicle must follow.

1. Stop and Route
These classes define a vehicle's plan. The ISingleVehicleRoutePlanner (defined later) will create and return Route objects.

```Python
@dataclass(frozen=True)
class Stop:
    """A single pickup or drop-off action in a route."""
    node: int                 # The graph node of the stop
    is_pickup: bool           # True for pickup, False for drop-off
    request_id: str           # Identifies which request this stop serves
    
@dataclass(frozen=True)
class Route:
    """An ordered sequence of stops for a single vehicle."""
    # This sequence defines the vehicle's path.
    # e.g.,
    stops: list
    
    # Pre-computed metrics for this specific route
    total_travel_time: float
    total_travel_distance: float
    
    # The final cost (sum of delays, sum(delta_r)) for this route
    # This is the 'c_ij' value for the ILP 
    total_delay_cost: float

```

1. INetworkService
This abstract interface provides all interactions with the underlying road network graph. It abstracts away the graph implementation (e.g., networkx [Minimal code showcase]) and precomputed path data.
Pythonfrom abc import ABC, abstractmethod

```Python
@dataclass(frozen=True)
class Path:
    """The result of a shortest-path query."""
    nodes: list[int]
    travel_time: float      # Travel time in seconds
    travel_distance: float  # Travel distance in meters

class INetworkService(ABC):
    """Abstract interface for all road network and travel time queries."""
    
    @abstractmethod
    def get_shortest_path(self, origin_node: int, destination_node: int) -> Path:
        """
        Returns the pre-computed fastest path between two nodes.
        This provides the baseline travel time tau(o_r, d_r).
        """
       ...
```

II. The Core "Engine": Single-Vehicle Routing (DARP)
The most critical sub-problem in the entire system is determining if a single vehicle can feasibly serve a set of new requests given its current state (passengers and location). This is the travel(v, \mathcal{R}_v) function described in the paper.2
This sub-problem is a well-known variant of the Vehicle Routing Problem (VRP), specifically the Dynamic Dial-a-Ride Problem (DARP) 5 or Pickup and Delivery Problem with Time Windows (PDPTW).8 The interface ISingleVehicleRoutePlanner abstracts this complex solver.

Feasibility Result Dataclasses
The planner's output is not just a boolean "yes" or "no" but a structured result.

```Python
@dataclass(frozen=True)
class FeasibilityResult:
    """Base class for the route planner's response."""
    is_feasible: bool

@dataclass(frozen=True)
class FeasibleRoute(FeasibilityResult):
    """Returned when a valid route is found."""
    is_feasible: bool = True
    route: Route  # The optimal route (sequence of stops)
    cost: float   # The resulting sum of delays (sum(delta_r))

@dataclass(frozen=True)
class Infeasible(FeasibilityResult):
    """Returned when no valid route exists."""
    is_feasible: bool = False
    reason: str   # e.g., "Violates max_wait", "Violates capacity"

```

ISingleVehicleRoutePlanner Interface
This interface defines the "engine" that will be called thousands of times per batch to build the shareability graphs.

```Python
class ISingleVehicleRoutePlanner(ABC):
    """
    Solves the single-vehicle DARP.
    This encapsulates the 'travel(v, R_v)' function.
    """
    def __init__(self, network_service: INetworkService):
        self._network = network_service
       ...

    @abstractmethod
    def check_feasibility(
        self,
        vehicle: Vehicle,
        requests_to_add: set
    ) -> FeasibilityResult:
        """
        Solves the DARP for a single vehicle.
        Attempts to find an optimal route (a permutation of stops) that serves:
        1. All passengers already onboard (vehicle.passengers_onboard)
        2. All new requests (requests_to_add)
       ...while respecting all constraints.
        """
       ...

```

Core Constraints
Any implementation of ISingleVehicleRoutePlanner must enforce the following constraints within its search to find an optimal Route:
Capacity Constraint: The number of passengers in the vehicle ($n_v^{pass}$) must never exceed the vehicle's capacity ($u$) at any point along the proposed Route.2
Precedence Constraint: For every Request (both new and existing passengers), its pickup Stop must occur before its drop-off Stop.9
Max Wait Time Constraint ($\Omega$): For every new request $r \in$ requests_to_add, its planned pickup time $t_r^p$ must satisfy: $t_r^p - r.request\_time \le r.max\_wait\_time$.2
Max Total Delay Constraint ($\Delta$): For all requests and passengers (both requests_to_add and vehicle.passengers_onboard), their final drop-off time $t_r^d$ must satisfy: $t_r^d - r.earliest\_dropoff\_time \le r.max\_total\_delay$.2
For vehicles with low capacity (e.g., $u \le 4$), this check can be an exhaustive search over all valid permutations of stops. For high-capacity vehicles ($u=10$), this must be a heuristic (e.g., Lin-Kernighan, Tabu search) to remain computationally tractable, as noted in the paper.13

III. The Four-Step Assignment Pipeline: Abstract Interfaces
The algorithm's scalability is achieved by decoupling the intractable global VRP.2 It separates the problem into two phases:
Candidate Generation (Steps 1-2): An intensive, but highly parallelizable, search for all locally feasible vehicle-trip combinations using the ISingleVehicleRoutePlanner.
Global Assignment (Steps 3-4): A centralized, a-priori-NP-hard optimization that selects the best global assignment from the pre-vetted candidates.
This process is modeled as a pipeline of four services, each corresponding to a step in the paper.2

Step 1: IRVGraphBuilder
This component builds the pairwise shareability graph (RV-graph).2 This graph connects requests that can be shared and vehicles that can serve them.
Python# A type alias for a graph object, e.g., networkx.Graph
# Nodes are Request and Vehicle objects.

```Python
RVGraph = Any 

class IRVGraphBuilder(ABC):
    
    def __init__(self, route_planner: ISingleVehicleRoutePlanner):
        self._route_planner = route_planner
       ...

    @abstractmethod
    def build_graph(self, requests: set, vehicles: set[Vehicle]) -> RVGraph:
        """
        Constructs the RV-graph by checking two types of pairwise shareability [2, 15]:
        
        1. Request-Request (r1, r2): Creates a virtual empty vehicle 'v_virtual'
           and checks if self._route_planner.check_feasibility(
               v_virtual, {r1, r2}
           ) is feasible.
           
        2. Request-Vehicle (r, v): Checks if self._route_planner.check_feasibility(
               v, {r}
           ) is feasible, given v's current state (passengers, location).
        """
       ...
```

Step 2: IRTVGraphBuilder
This component finds cliques in the RVGraph to form candidate Trips (a set of requests) and then validates which real vehicles can serve them, creating the RTV-graph.2
Python# A type alias for a bipartite graph object (e.g., networkx.Graph)
# One set of nodes is 'Trip' objects (which are sets of Requests)
# The other set of nodes is 'Vehicle' objects.
```Python
RTVGraph = Any

class IRTVGraphBuilder(ABC):
    
    def __init__(self, route_planner: ISingleVehicleRoutePlanner):
        self._route_planner = route_planner
       ...

    @abstractmethod
    def build_graph(
        self, rv_graph: RVGraph, vehicles: set[Vehicle], max_trip_size: int
    ) -> RTVGraph:
        """
        Builds the RTV-graph :
        
        1. Finds all cliques of size <= max_trip_size in the 'Request'
           subgraph of rv_graph. Each clique is a candidate 'Trip' T.[14]
           
        2. For each candidate Trip T and each Vehicle v:
           a. Check feasibility:
              result = self._route_planner.check_feasibility(v, T.requests)
              
           b. If result.is_feasible:
              i.  Add an edge e(T, v) to the RTVGraph.
              ii. Store result.cost and result.route as attributes
                  of that edge.[2, 3]
                  
        (This implementation can be optimized by checking subtrips first ).
        """
       ...
```

Step 3: IAssignmentSolver (Main ILP)
This component formulates and solves the main Integer Linear Program (ILP) from Algorithm 1 in the paper to find the optimal assignment.1

```Python
@dataclass(frozen=True)
class AssignmentResult:
    """The output of the main assignment ILP."""
    # The (Vehicle, Trip, Route) tuples define the new assignments
    assignments: list[Tuple[Vehicle, set[Request], Route]]
    unassigned_requests: set[Request]
    total_cost: float

class IAssignmentSolver(ABC):
    
    @abstractmethod
    def solve(
        self, 
        rtv_graph: RTVGraph, 
        requests: set, 
        vehicles: set[Vehicle], 
        solver_timeout: float
    ) -> AssignmentResult:
        """
        Formulates and solves the ILP from Algorithm 1 [2, 3]:
        
        - Variables:
          - epsilon_ij (binary) for each edge e(Trip_i, Vehicle_j) in RTVGraph.
          - chi_k (binary) for each Request_k in 'requests'.
          
        - Objective:
          - min sum(c_ij * epsilon_ij) + sum(c_ko * chi_k)
          - (c_ij is the 'cost' stored on the RTVGraph edge )
          
        - Constraints:
          - sum_i(epsilon_ij) <= 1  (Vehicle j assigned to at most one trip)
          - sum_i_j(epsilon_ij) + chi_k = 1 (Request k is in one assigned trip OR 
                                           is unassigned, 'ignored' )
                                           
        - Must provide a 'warm start' (greedy assignment)  and adhere
          to the 'solver_timeout' to ensure 'anytime' performance.
        """
       ...
```

Step 4: IRebalancingSolver (Rebalancing LP)
The paper specifies a fourth step: rebalancing idle vehicles to serve unassigned requests.2 This is not part of the main ILP but a subsequent, simpler Linear Program (LP) that optimizes a different objective. This separation simplifies the main ILP by removing rebalancing concerns, which are handled only for the remaining idle vehicles and unserved requests.

```Python
@dataclass(frozen=True)
class RebalancingResult:
    """The output of the rebalancing LP."""
    # Simple (Vehicle, Request) pairs
    assignments: list]

class IRebalancingSolver(ABC):

    def __init__(self, network_service: INetworkService):
        self._network = network_service
       ...

    @abstractmethod
    def solve(
        self, 
        idle_vehicles: set[Vehicle], 
        unassigned_requests: set
    ) -> RebalancingResult:
        """
        Formulates and solves the LP from SI Appendix, Algorithm 4 [4, 16]:
        
        - Objective: min sum(travel_time(v, r) * y_vr)
          - (Uses network_service to get simple travel times)
          
        - Variables: y_vr (binary/continuous) for each (v, r) pair.
        
        - Constraints (Standard assignment problem):
          - sum_r(y_vr) <= 1 (Vehicle v assigned to at most one request)
          - sum_v(y_vr) <= 1 (Request r assigned to at most one vehicle)
        """
       ...
```

### Summary of Pipeline Interfaces

This table summarizes the data flow and dependencies for the complete four-step pipeline.

| Interface            | Primary Input(s)                           | Primary Output      | Core Task (from Paper)                                        | Key Dependencies             |
| -------------------- | ------------------------------------------ | ------------------- | ------------------------------------------------------------- | ---------------------------- |
| `IRVGraphBuilder`    | `set[Request]`, `set[Vehicle]`             | `RVGraph`           | "computing a pairwise request-vehicle shareability graph" [2] | `ISingleVehicleRoutePlanner` |
| `IRTVGraphBuilder`   | `RVGraph`, `set[Vehicle]`                  | `RTVGraph`          | "computing a graph of feasible trips... (RTV-graph)" [2]      | `ISingleVehicleRoutePlanner` |
| `IAssignmentSolver`  | `RTVGraph`, `set[Request]`, `set[Vehicle]` | `AssignmentResult`  | "solving an ILP to compute the best assignment" [2]           | ILP solver library           |
| `IRebalancingSolver` | `set[Vehicle]` (idle), `set[Request]`      | `RebalancingResult` | "rebalancing the remaining idle vehicles" [2]                 | `INetworkService`, LP solver |

### IV. System Orchestration and Data Flow

The interfaces above are orchestrated by a central controller that runs the assignment pipeline in discrete batches.

**BatchAssignmentController**
- Manages the main application loop on a fixed cadence (e.g., every 30 seconds [16]).
- Core dependencies: `RequestPool` for $\mathcal{R}$, `VehicleFleet` for $\mathcal{V}$, plus `IRVGraphBuilder`, `IRTVGraphBuilder`, `IAssignmentSolver`, and `IRebalancingSolver`.

**30-Second Batch Cycle**
1. `T = 0s`: Wake; fetch state via `RequestPool.get_all()` and `VehicleFleet.get_all()`.
2. `T = 1s`: Build `rv_graph = rv_builder.build_graph(requests, vehicles)`.
3. `T = 5s`: Build `rtv_graph = rtv_builder.build_graph(rv_graph, vehicles, max_trip_size=10)`.
4. `T = 10s`: Solve `assignment_result = assignment_solver.solve(..., solver_timeout=19.0)`.
5. `T = 29s`: Derive `idle_vehicles` and `unassigned_requests` from the assignment output.
6. `T = 29s`: Solve `rebalance_result = rebalancing_solver.solve(idle_vehicles, unassigned_requests)`.
7. `T = 30s`: Commit results to `VehicleFleet` and `RequestPool`, then sleep until the next batch.

### "Anytime" vs. "Batch" Orchestration

- The strict batch window enforces real-time guarantees (30 seconds is acceptable [16]).
- `IAssignmentSolver` honors the anytime requirement by starting from a greedy warm start [3].
- The controller interrupts the ILP at the timeout and keeps the best solution found so far [3].

### Data Flow for a Single Request

1. **Ingestion**: Create a `Request` and add it to the `RequestPool`.
2. **Batching**: Pull the request into the active set at the next batch boundary.
3. **Step 1 (RV)**: Add the request as a node in the RV-graph with shareable edges.
4. **Step 2 (RTV)**: Add the request to candidate trip nodes in the RTV-graph, linked to feasible vehicles.
5. **Step 3 (ILP)**: If some $\epsilon_{i,j} = 1$ for trip $T_i$, assign the request and update the vehicle route; otherwise $\chi_k = 1$ and the request remains unassigned.
6. **Step 4 (Rebalance)**: Attempt to match idle vehicles to unassigned requests; on success the vehicle and request are updated, otherwise the request waits for the next batch with tighter constraints.

### Architectural Recommendations

- **Parallelism**: Distribute `check_feasibility` calls across workers to accelerate Steps 1–2 [16].
- **Solver abstraction**: Keep optimization libraries behind `IAssignmentSolver` and `IRebalancingSolver`.
- **Engine specialization**: Choose exhaustive search for $u \le 4$ and heuristic planners for $u = 10$ during runtime [13].

### V. Conclusion

This architectural blueprint defines dataclasses (`Request`, `Vehicle`, `Passenger`, `Route`) and abstract services (`INetworkService`, `ISingleVehicleRoutePlanner`, `IRVGraphBuilder`, `IRTVGraphBuilder`, `IAssignmentSolver`, `IRebalancingSolver`) that encapsulate the algorithm’s complexity. The design decouples single-vehicle routing from global assignment, separates candidate generation from centralized optimization, and isolates fleet rebalancing into its own step—yielding a scalable, testable implementation for real-time, high-capacity ride-sharing.
