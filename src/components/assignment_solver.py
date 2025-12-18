import gurobipy as gp
from gurobipy import GRB

from config import get_logger
from src.interfaces import (
    AssignmentResult,
    Request,
    VehicleAvailability,
    IAssignmentSolver,
)
from src.interfaces.data_models import Assignment
from src.interfaces.graphs import RTVGraph, TripId


logger = get_logger(__name__)


class GurobiAssignmentSolver(IAssignmentSolver):
    """
    Implementation of the Assignment Solver using Gurobi Optimizer.
    """

    def __init__(self, penalty_base: float = 1e5, penalty_growth_rate: float = 10.0):
        self._penalty_base = penalty_base
        self._penalty_growth_rate = penalty_growth_rate

    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,
    ) -> AssignmentResult:

        # 1. Setup Model
        # Use a context manager to dispose of the model properly
        with gp.Model("RideSharing_Assignment") as model:

            # Silence console output (set to 1 for debugging)
            model.setParam(GRB.Param.OutputFlag, 0)
            model.setParam(GRB.Param.TimeLimit, solver_timeout)
            model.setParam(GRB.Param.MIPGap, 0.01)  # 1% gap is usually sufficient

            # ------------------------------------------------------
            # 2. Decision Variables
            # ------------------------------------------------------

            # x[trip, vehicle_id] -> binary
            x_vars: dict[tuple[TripId, str], gp.Var] = {}

            # y[request_id] -> binary (1 if unassigned)
            y_vars: dict[str, gp.Var] = {}

            # Create x_vars for all feasible edges e(T, v)
            # rtv_graph.trip_vehicle_details keys are (TripId, vehicle_id)
            for trip, v_id in rtv_graph.trip_vehicle_details.keys():
                cost = rtv_graph.trip_vehicle_details[(trip, v_id)].cost
                # Name is helpful for debugging .lp files
                name = f"x_{hash(trip)}_{v_id}"
                x_vars[(trip, v_id)] = model.addVar(
                    vtype=GRB.BINARY, obj=cost, name=name
                )

            # Create y_vars for all requests
            for req in requests:
                # Calculate dynamic penalty chi_r
                penalty = self._calculate_penalty(req, current_time)
                y_vars[req.id] = model.addVar(
                    vtype=GRB.BINARY, obj=penalty, name=f"y_{req.id}"
                )

            # ------------------------------------------------------
            # 3. Constraints
            # ------------------------------------------------------

            # C1: Assign each vehicle at most once
            # sum(x_{T,v}) <= 1  forall v
            for v in vehicles:
                # Get all trips this vehicle can serve
                possible_trips = rtv_graph.get_trips_for_vehicle(v.vehicle_id)

                # Construct expression: sum(x[t, v.id] for t in possible_trips)
                # We filter x_vars keys to ensure the var exists (it should)
                vars_for_v = [
                    x_vars[(t, v.vehicle_id)]
                    for t in possible_trips
                    if (t, v.vehicle_id) in x_vars
                ]

                if vars_for_v:
                    model.addConstr(
                        gp.quicksum(vars_for_v) <= 1, name=f"veh_{v.vehicle_id}"
                    )

            # C2: Assign each request at most once (or set y=1)
            # sum(x_{T,v} for T containing r) + y_r = 1
            for req in requests:
                # Get all trips containing this request
                trips_with_r = rtv_graph.get_trips_for_request(req.id)

                # Gather all x variables for these trips (across all vehicles)
                vars_for_r = []
                for t in trips_with_r:
                    # A trip might be servable by multiple vehicles
                    possible_vs = rtv_graph.get_vehicles_for_trip(t)
                    for v_id in possible_vs:
                        if (t, v_id) in x_vars:
                            vars_for_r.append(x_vars[(t, v_id)])

                model.addConstr(
                    gp.quicksum(vars_for_r) + y_vars[req.id] == 1, name=f"req_{req.id}"
                )

            # ------------------------------------------------------
            # 4. Warm Start (Greedy Heuristic)
            # ------------------------------------------------------
            self._apply_greedy_warm_start(model, rtv_graph, requests, x_vars, y_vars)

            # ------------------------------------------------------
            # 5. Optimize
            # ------------------------------------------------------
            model.optimize()

            # ------------------------------------------------------
            # 6. Extract Result
            # ------------------------------------------------------
            status = model.Status
            if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                # Extract assignments
                assignments: set[Assignment] = set()
                assigned_req_ids: set[str] = set()

                # Check x vars
                for (trip, v_id), var in x_vars.items():
                    # Gurobi vars are float, check against tolerance
                    if var.X > 0.5:
                        # Get the full route details
                        route_info = rtv_graph.get_route(trip, v_id)
                        if route_info:
                            assignments.add(
                                Assignment(
                                    vehicle_id=v_id,
                                    trip_id=trip,
                                    route=route_info.route,
                                    cost=route_info.cost,
                                )
                            )
                            for r_id in trip:
                                assigned_req_ids.add(r_id)

                # Identify unassigned requests
                unassigned_reqs = {r for r in requests if r.id not in assigned_req_ids}

                # mao status from int to str for logging
                status_dict = {
                    GRB.OPTIMAL: "OPTIMAL",
                    GRB.TIME_LIMIT: "TIME_LIMIT",
                    GRB.SUBOPTIMAL: "SUBOPTIMAL",
                }
                status = status_dict.get(status, str(status))

                logger.info(
                    f"Solver finished. Status: {status}. "
                    f"Assigned {len(assignments)} vehicles, {len(assigned_req_ids)} requests."
                )

                return AssignmentResult(
                    assignments=assignments,
                    unassigned_requests=unassigned_reqs,
                    total_cost=model.ObjVal,
                    is_optimal=(status == GRB.OPTIMAL),
                )

            else:
                raise RuntimeError(
                    f"Gurobi failed to find solution. Status code: {status}"
                )
                logger.error(f"Gurobi failed to find solution. Status code: {status}")
                # Fallback: return all unassigned
                return AssignmentResult(
                    assignments=set(), unassigned_requests=requests, is_optimal=False
                )

    def _calculate_penalty(self, request: Request, current_time: float) -> float:
        """
        Calculates the dynamic penalty chi_r.

        Logic:
        - Base penalty (ensure it's higher than any feasible trip cost).
        - Time component: penalty grows as the request waits longer.
        - If 'promised_pickup_time' exists (re-assigned request), implies
          it was already assigned once, so we might boost priority further.
        """
        wait_time = current_time - request.request_time

        penalty = self._penalty_base + (wait_time * self._penalty_growth_rate)

        # Optional: Massive boost for requests that were dropped (promised time set)
        # to prevent the system from "flicking" passengers between assigned/unassigned
        if request.promised_pickup_time is not None:
            penalty *= 2.0

        return penalty

    def _apply_greedy_warm_start(
        self,
        model: gp.Model,
        rtv_graph: RTVGraph,
        requests: set[Request],
        x_vars: dict[tuple[TripId, str], gp.Var],
        y_vars: dict[str, gp.Var],
    ) -> None:
        """
        Implements Algorithm 2 (Greedy Assignment) from Alonso-Mora et al.
        to provide an initial feasible solution (MIP Start) to Gurobi.
        """
        # 1. Collect all candidate assignments (edges in RTV)
        # Format: (cost, trip_size, trip, vehicle_id)
        candidates = []
        for (trip, v_id), route_info in rtv_graph.trip_vehicle_details.items():
            # Sorting criteria:
            # 1. Decreasing trip size (prioritize pooling)
            # 2. Increasing cost (minimize delay)
            # To use standard sort, we negate size.
            candidates.append((route_info.cost, -len(trip), trip, v_id))

        # Sort candidates
        # Primary key: cost (asc), Secondary key: -size (asc) -> size desc
        # Note: The paper suggests "decreasing size" first.
        # Let's prioritize Size then Cost as per paper.
        candidates.sort(key=lambda x: (x[1], x[0]))

        assigned_requests: set[str] = set()
        assigned_vehicles: set[str] = set()

        # 2. Greedy Selection
        for cost, neg_size, trip, v_id in candidates:

            # Check vehicle availability
            if v_id in assigned_vehicles:
                continue

            # Check request availability (all reqs in trip must be free)
            if any(r_id in assigned_requests for r_id in trip):
                continue

            # Make assignment
            assigned_vehicles.add(v_id)
            for r_id in trip:
                assigned_requests.add(r_id)

            # Tell Gurobi to start with this variable = 1
            if (trip, v_id) in x_vars:
                x_vars[(trip, v_id)].Start = 1.0

        # 3. Set unassigned y variables
        for r in requests:
            if r.id not in assigned_requests:
                y_vars[r.id].Start = 1.0
            else:
                y_vars[r.id].Start = 0.0
