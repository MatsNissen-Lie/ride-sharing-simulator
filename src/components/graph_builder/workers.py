# src/graph_builder/workers.py

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


def process_rr_chunk(tasks: list[RRTask]) -> list[tuple[str, str, FeasibleRoute]]:
    """
    Worker function: Initializes its own resources, solves a list of RR tasks.
    """
    # 1. Initialize Local Resources (Per Process)
    # DuckDB handles concurrent read connections very well.
    # Note: We pass None as graph because RoutePlanner only needs the DB for travel times.
    # If your network service REQUIRES the networkx graph, you might need to load it
    # or use a 'Lazy' loader. ideally NetworkXService handles a None graph gracefully
    # if only using DB cache.

    # Assuming NetworkXService can work purely on DB for cache hits:
    db_cache = DuckDBPathCache()
    # We cheat: we don't pass the NX graph. The planner usually only needs get_shortest_path
    # which hits the DB cache first.
    # WARNING: If cache misses, this worker crashes if G is missing.
    # Ensure your cache is warm or pass the G path if needed.
    network_service = NetworkXService(graph=None)  # type: ignore
    network_service._cache = db_cache

    planner = RoutePlanner(network_service)

    results = []
    for task in tasks:
        # Check feasibility
        # We assume the matrix filter was already done by the main process
        res = planner.check_feasibility({task.r1, task.r2}, find_best_route=False)
        if res.is_feasible:
            # We must return simple data, not complex objects if possible,
            # though FeasibleRoute is usually fine if it's a Pydantic/Dataclass.
            results.append((task.r1.id, task.r2.id, res))

    return results


def process_rv_chunk(tasks: list[RVTask]) -> list[tuple[str, str, FeasibleRoute]]:
    """
    Worker function for RV checks.
    """
    db_cache = DuckDBPathCache()
    network_service = NetworkXService(graph=None)  # type: ignore
    network_service._cache = db_cache
    planner = RoutePlanner(network_service)

    results = []
    for task in tasks:
        res = planner.check_feasibility(
            {task.req}, vehicle=task.vehicle, find_best_route=True
        )
        if res.is_feasible:
            results.append((task.req.id, task.vehicle.vehicle_id, res))

    return results
