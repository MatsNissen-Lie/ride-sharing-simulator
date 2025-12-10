import time
from typing import cast, List, Tuple
import numpy as np
from src.components.graph_builder.trip_graph_dfs_astar import TripSolver as AStarSolver
from src.components.graph_builder.trip_graph_no_memo import TripSolver as DFSSolver
from src.components.graph_builder.trip_graph_bfs import TripSolver as BFSSolver
from src.interfaces import VehicleAvailability, Request
from src.interfaces.data_models import FeasibleRoute
from config.config import DELTA, OMEGA


def generate_problem(
    n_reqs: int, n_nodes: int, seed: int
) -> Tuple[List[Request], VehicleAvailability, np.ndarray]:
    np.random.seed(seed)

    # Random time matrix
    coords = np.random.rand(n_nodes, 2) * 10
    time_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            time_matrix[i, j] = dist

    # Vehicle
    vehicle = VehicleAvailability(
        vehicle_id="V1",
        available_at_node=0,
        available_from_time=0.0,
        capacity=4,
        passengers_onboard=frozenset(),
        remaining_stops_to_serve=(),
    )

    # Requests
    requests = []
    for i in range(n_reqs):
        origin = np.random.randint(1, n_nodes)
        dest = np.random.randint(1, n_nodes)
        while dest == origin:
            dest = np.random.randint(1, n_nodes)

        direct_time = time_matrix[origin, dest]
        req_time = 0.0

        # 10 minutes slack
        hard_limit = req_time + direct_time + DELTA

        req = Request(
            id=f"R{i}",
            origin_node=origin,
            destination_node=dest,
            request_time=req_time,
            latest_dropoff_time=hard_limit,
            latest_pickup_time=req_time + OMEGA,
        )
        requests.append(req)

    return requests, vehicle, time_matrix


def run_benchmark():
    cases = [
        ("Small", 2, 1000),  # 2 Reqs (4 stops)
        ("Medium", 3, 1000),  # 3 Reqs (6 stops)
        ("Large", 6, 1),  # 6 Reqs (12 stops) - Run once
    ]

    print("\n" + "=" * 75)
    print(
        f" {'Case':<10} | {'Nodes':<6} | {'Solver':<15} | {'Avg Time (ms)':<15} | {'Total Time':<10}"
    )
    print("-" * 75)

    for name, n_reqs, iterations in cases:
        n_nodes_solver = 2 * n_reqs
        times = {"DFS": [], "A* DFS": [], "A* BFS": []}

        for i in range(iterations):
            requests, vehicle, time_matrix = generate_problem(n_reqs, 20, seed=42 + i)

            # 1. DFS
            t0 = time.time()
            res_dfs = DFSSolver.solve(requests, vehicle, time_matrix)
            times["DFS"].append(time.time() - t0)

            # 2. A* DFS
            t0 = time.time()
            res_astar = AStarSolver.solve(requests, vehicle, time_matrix)
            times["A* DFS"].append(time.time() - t0)

            # 3. A* BFS
            t0 = time.time()
            res_bfs = BFSSolver.solve(requests, vehicle, time_matrix)
            times["A* BFS"].append(time.time() - t0)

            # Validation (only check first few to save time/logs)
            if i == 0 and res_dfs.is_feasible and res_bfs.is_feasible:
                c1 = cast(FeasibleRoute, res_dfs).cost
                c3 = cast(FeasibleRoute, res_bfs).cost
                if abs(c1 - c3) > 1e-5:
                    print(f"WARNING: Cost mismatch in {name} case! DFS={c1}, BFS={c3}")

        # Print Stats
        for solver in ["DFS", "A* DFS", "A* BFS"]:
            avg_t = np.mean(times[solver])
            tot_t = np.sum(times[solver])

            ms_val = avg_t * 1000
            if ms_val >= 100:
                time_str = f"{ms_val:.1f}"
            else:
                time_str = f"{ms_val:.4f}"

            print(
                f" {name:<10} | {n_nodes_solver:<6} | {solver:<15} | {time_str:<15} | {tot_t:.4f}s"
            )
        print("-" * 75)
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_benchmark()
