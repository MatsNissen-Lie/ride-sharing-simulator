"""The assignment ILP with interchangeable solver backends.

min  sum c_{T,v} x_{T,v} + sum chi_r y_r
s.t. sum_{T contains r} x_{T,v} + y_r = 1   for every request r
     sum_T x_{T,v} <= 1                     for every vehicle v

c and chi come from an IObjective, so the same backends serve any objective.
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from config import get_logger
from src.components.objectives import DelayObjective
from src.interfaces import (
    AssignmentResult,
    IAssignmentSolver,
    IObjective,
    Request,
    VehicleAvailability,
)
from src.interfaces.data_models import Assignment
from src.interfaces.graphs import RTVGraph, TripId

logger = get_logger(__name__)

if __name__ != "__main__":
    logger.setLevel("WARNING")

# Optimality tolerance in objective units. The objective is dominated by large
# per-request penalties, so a relative gap would hide real cost differences.
MIP_ABS_GAP = 1e-3


class AssignmentSolverBase(IAssignmentSolver):
    """Shared model data and result extraction; backends implement `solve`."""

    def __init__(self, objective: IObjective | None = None):
        self._objective = objective or DelayObjective()

    def _edge_costs(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
    ) -> dict[tuple[TripId, str], float]:
        req_map = {r.id: r for r in requests}
        veh_map = {v.vehicle_id: v for v in vehicles}
        return {
            (trip, v_id): self._objective.trip_cost(
                route, veh_map[v_id], [req_map[r] for r in sorted(trip)]
            )
            for (trip, v_id), route in rtv_graph.trip_vehicle_details.items()
        }

    def _result(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        chosen: list[tuple[TripId, str]],
        total_cost: float,
        is_optimal: bool,
    ) -> AssignmentResult:
        assignments: set[Assignment] = set()
        assigned_req_ids: set[str] = set()
        for trip, v_id in chosen:
            route_info = rtv_graph.get_route(trip, v_id)
            assignments.add(
                Assignment(
                    vehicle_id=v_id,
                    trip_id=trip,
                    route=route_info.route,
                    cost=route_info.cost,
                )
            )
            assigned_req_ids.update(trip)
        return AssignmentResult(
            assignments=assignments,
            unassigned_requests={r for r in requests if r.id not in assigned_req_ids},
            total_cost=total_cost,
            is_optimal=is_optimal,
        )


class GurobiAssignmentSolver(AssignmentSolverBase):
    """Gurobi with a greedy warm start (Algorithm 2 in the SI Appendix)."""

    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,
    ) -> AssignmentResult:
        costs = self._edge_costs(rtv_graph, requests, vehicles)

        with gp.Model("RideSharing_Assignment") as model:
            model.setParam(GRB.Param.OutputFlag, 0)
            model.setParam(GRB.Param.TimeLimit, solver_timeout)
            model.setParam(GRB.Param.MIPGap, 0.0)
            model.setParam(GRB.Param.MIPGapAbs, MIP_ABS_GAP)

            x_vars: dict[tuple[TripId, str], gp.Var] = {
                key: model.addVar(vtype=GRB.BINARY, obj=cost, name=f"x_{hash(key[0])}_{key[1]}")
                for key, cost in costs.items()
            }
            y_vars: dict[str, gp.Var] = {
                req.id: model.addVar(
                    vtype=GRB.BINARY,
                    obj=self._objective.unserved_penalty(req, current_time),
                    name=f"y_{req.id}",
                )
                for req in requests
            }

            for v in vehicles:
                vars_for_v = [
                    x_vars[(t, v.vehicle_id)]
                    for t in rtv_graph.get_trips_for_vehicle(v.vehicle_id)
                    if (t, v.vehicle_id) in x_vars
                ]
                if vars_for_v:
                    model.addConstr(gp.quicksum(vars_for_v) <= 1, name=f"veh_{v.vehicle_id}")

            for req in requests:
                vars_for_r = [
                    x_vars[(t, v_id)]
                    for t in rtv_graph.get_trips_for_request(req.id)
                    for v_id in rtv_graph.get_vehicles_for_trip(t)
                    if (t, v_id) in x_vars
                ]
                model.addConstr(
                    gp.quicksum(vars_for_r) + y_vars[req.id] == 1, name=f"req_{req.id}"
                )

            self._apply_greedy_warm_start(costs, requests, x_vars, y_vars)
            model.optimize()

            status = model.Status
            if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                raise RuntimeError(f"Gurobi failed to find solution. Status code: {status}")
            chosen = [key for key, var in x_vars.items() if var.X > 0.5]
            logger.info(f"Gurobi finished with status {status}; {len(chosen)} trips assigned.")
            return self._result(
                rtv_graph, requests, chosen, model.ObjVal, status == GRB.OPTIMAL
            )

    @staticmethod
    def _apply_greedy_warm_start(
        costs: dict[tuple[TripId, str], float],
        requests: set[Request],
        x_vars: dict[tuple[TripId, str], gp.Var],
        y_vars: dict[str, gp.Var],
    ) -> None:
        """Largest trips first, then cheapest, as long as vehicle and requests are free."""
        assigned_requests: set[str] = set()
        assigned_vehicles: set[str] = set()
        for (trip, v_id), _ in sorted(costs.items(), key=lambda kv: (-len(kv[0][0]), kv[1])):
            if v_id in assigned_vehicles or any(r in assigned_requests for r in trip):
                continue
            assigned_vehicles.add(v_id)
            assigned_requests.update(trip)
            x_vars[(trip, v_id)].Start = 1.0
        for r in requests:
            y_vars[r.id].Start = 0.0 if r.id in assigned_requests else 1.0


class HighsAssignmentSolver(AssignmentSolverBase):
    """
    HiGHS through scipy.optimize.milp: open-source, no licence needed.
    HiGHS takes no MIP start, so there is no greedy warm start.
    """

    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,
    ) -> AssignmentResult:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        req_list = list(requests)
        if not req_list:
            return AssignmentResult(set(), set(), 0.0, True)
        costs = self._edge_costs(rtv_graph, requests, vehicles)
        edges = list(costs)
        req_index = {r.id: k for k, r in enumerate(req_list)}
        veh_index = {v.vehicle_id: k for k, v in enumerate(vehicles)}
        n_x, n_y = len(edges), len(req_list)

        cost = np.empty(n_x + n_y)
        rows, cols = [], []  # request rows first, then vehicle rows
        for e, (trip, v_id) in enumerate(edges):
            cost[e] = costs[(trip, v_id)]
            for r_id in trip:
                rows.append(req_index[r_id])
                cols.append(e)
            rows.append(n_y + veh_index[v_id])
            cols.append(e)
        for k, req in enumerate(req_list):
            cost[n_x + k] = self._objective.unserved_penalty(req, current_time)
            rows.append(k)
            cols.append(n_x + k)

        n_rows = n_y + len(veh_index)
        a = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_x + n_y)).tocsr()
        lower = np.concatenate([np.ones(n_y), np.zeros(len(veh_index))])
        result = milp(
            c=cost,
            constraints=LinearConstraint(a, lower, np.ones(n_rows)),
            integrality=np.ones(n_x + n_y),
            bounds=Bounds(0, 1),
            options={"time_limit": solver_timeout, "mip_rel_gap": 0.0},
        )
        if result.x is None:
            raise RuntimeError(f"HiGHS failed to find a solution: {result.message}")

        chosen = [edges[e] for e in range(n_x) if result.x[e] > 0.5]
        logger.info(f"HiGHS finished: {result.message} {len(chosen)} trips assigned.")
        return self._result(rtv_graph, requests, chosen, float(result.fun), result.status == 0)
