# src/graph_builder/workers.py

import numpy as np
from src.components.network_service import NetworkXService
from src.database.simple_db import DuckDBPathCache
from src.components.route_planner import RoutePlanner
from src.interfaces import Request, VehicleAvailability, FeasibleRoute


# We define a lightweight struct for the arguments to avoid pickling the whole world
class RRTask:
    def __init__(self, r1: Request, r2: Request):
        self.r1 = r1
        self.r2 = r2


class RVTask:
    def __init__(self, vehicle: VehicleAvailability, req: Request):
        self.vehicle = vehicle
        self.req = req


# Global resources for workers
_db_cache: DuckDBPathCache | None = None
_network_service: NetworkXService | None = None


def init_worker(shared_cache: np.ndarray | None = None):
    """
    Initializer for worker processes.
    Opens the DB connection once per process.
    """
    global _db_cache, _network_service
    _db_cache = DuckDBPathCache(read_only=True)
    # We cheat: we don't pass the NX graph. The planner usually only needs get_shortest_path
    # which hits the DB cache first.
    max_node_id = shared_cache.shape[0] if shared_cache is not None else None
    _network_service = NetworkXService(graph=None, db=_db_cache, max_node_id=max_node_id, prefetch=False)  # type: ignore

    if shared_cache is not None:
        _network_service.update_cache(shared_cache)
        _network_service.is_nuclear = True


# def process_rr_chunk(tasks: list[RRTask]) -> list[tuple[str, str, FeasibleRoute]]:
#     """
#     Worker function: Uses pre-initialized resources.
#     """
#     global _network_service
#     if _network_service is None:
#         raise RuntimeError("Worker not initialized")

#     planner = RoutePlanner(_network_service)

#     results = []
#     for task in tasks:
#         # Check feasibility
#         res = planner.check_feasibility({task.r1, task.r2}, find_best_route=False)
#         if res.is_feasible:
#             results.append((task.r1.id, task.r2.id, res))
#     return results


def process_rv_chunk(tasks: list[RVTask]) -> list[tuple[str, str, FeasibleRoute]]:
    """
    Worker function for RV checks.
    """
    global _network_service
    if _network_service is None:
        raise RuntimeError("Worker not initialized")

    planner = RoutePlanner(_network_service)

    results = []
    for task in tasks:
        res = planner.check_feasibility(
            {task.req}, vehicle=task.vehicle, find_best_route=True
        )
        if res.is_feasible:
            results.append((task.req.id, task.vehicle.vehicle_id, res))
    return results
