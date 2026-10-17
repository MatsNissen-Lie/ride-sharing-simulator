import time
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict
import os

from src.components.search_algorithm.bfs_astar import TripSolver as BFSSolver
from src.components.search_algorithm.bfs_memo import TripSolver as MemoSolver
from src.components.search_algorithm.bfs_memo import _UNIVERSAL_CACHE
from src.interfaces import VehicleAvailability, Request, Passenger
from config.config import DELTA, OMEGA


def generate_coherent_batch(
    size: int, n_nodes: int, time_matrix: np.ndarray
) -> List[Request]:
    """Generates a batch of requests that are geographically close, increasing feasibility."""
    # Pick a "center" for this batch
    center_node = np.random.randint(0, n_nodes)

    # Find neighbors within a reasonable radius (e.g., 20% of map)
    # For simplicity in this synthetic graph, just pick nodes with index close to center
    # or use time_matrix to find close nodes.

    # Let's use time_matrix to find real neighbors
    dists = time_matrix[center_node]
    # Get indices of nodes within 1500 seconds (25 mins)
    candidates = np.where(dists < 1500)[0]
    if len(candidates) < 2:
        candidates = np.arange(n_nodes)  # Fallback

    requests = []
    for r_idx in range(size):
        origin = np.random.choice(candidates)
        dest = np.random.choice(candidates)
        while dest == origin:
            dest = np.random.choice(candidates)

        dist = time_matrix[origin, dest]
        hard_limit = float(dist) + DELTA

        req = Request(
            id=f"Req_{r_idx}",
            origin_node=origin,
            destination_node=dest,
            request_time=0.0,
            latest_dropoff_time=hard_limit,
            latest_pickup_time=OMEGA,
            # earliest_dropoff_time=earliest_dropoff,
            # latest_dropoff_time=earliest_dropoff + DELTA,
        )
        requests.append(req)
    return requests


def generate_vehicles(
    n_vehicles: int, n_nodes: int, time_matrix: np.ndarray
) -> List[VehicleAvailability]:
    vehicles = []
    for v_idx in range(n_vehicles):
        start_node = np.random.randint(0, n_nodes)
        n_onboard = np.random.randint(0, 4)  # 0 to 3 passengers

        passengers = []
        for p_idx in range(n_onboard):
            dest_node = np.random.randint(0, n_nodes)
            while dest_node == start_node:
                dest_node = np.random.randint(0, n_nodes)

            dist = time_matrix[start_node, dest_node]

            req = Request(
                id=f"Onboard_{v_idx}_{p_idx}",
                origin_node=start_node,
                destination_node=dest_node,
                request_time=0.0,
                latest_dropoff_time=dist + DELTA,
                latest_pickup_time=0.0 + OMEGA,
            )

            p = Passenger(original_request=req, pickup_time=0.0)
            passengers.append(p)

        vehicle = VehicleAvailability(
            vehicle_id=f"V{v_idx}",
            available_at_node=start_node,
            available_from_time=0.0,
            capacity=6,
            passengers_onboard=frozenset(passengers),
            remaining_stops_to_serve=(),
        )
        vehicles.append(vehicle)
    return vehicles


def run_realistic_benchmark():
    # Setup
    N_NODES = 100
    N_VEHICLES = 1000

    # Realistic Distribution of FEASIBLE sets
    BATCH_COUNTS = {2: 270, 3: 280, 4: 95, 5: 10}

    # Overlap: How many vehicles check each batch?
    VEHICLES_PER_BATCH = 30

    # Environment
    np.random.seed(42)
    random.seed(42)
    coords = np.random.rand(N_NODES, 2) * 100
    time_matrix = np.zeros((N_NODES, N_NODES))
    for i in range(N_NODES):
        for j in range(N_NODES):
            time_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    print("Generating Resources...")
    # We don't use a single pool anymore, we generate coherent batches on the fly
    vehicles = generate_vehicles(N_VEHICLES, N_NODES, time_matrix)

    results = {
        k: {"bfs": 0.0, "memo": 0.0, "checks": 0, "feasible": 0} for k in BATCH_COUNTS
    }

    print(f"\nStarting Realistic Benchmark (Coherent Batches)")
    print(f"Fleet: {N_VEHICLES} vehicles")
    print(f"Overlap: {VEHICLES_PER_BATCH} vehicles check each batch")
    print("-" * 60)
    print(
        f"{'Size':<5} | {'Batches':<8} | {'Checks':<8} | {'BFS (s)':<10} | {'Memo (s)':<10} | {'Speedup':<8} | {'Feas %':<8}"
    )
    print("-" * 60)

    _UNIVERSAL_CACHE.clear()

    for size, count in BATCH_COUNTS.items():
        # Generate 'count' COHERENT batches of 'size'
        batches = []
        for _ in range(count):
            batches.append(generate_coherent_batch(size, N_NODES, time_matrix))

        t_bfs_total = 0.0
        t_memo_total = 0.0
        checks = 0
        feasible = 0

        for batch in batches:
            # Clear cache to prevent ID collisions between batches
            _UNIVERSAL_CACHE.clear()

            # Select candidate vehicles for this batch
            # To increase feasibility, pick vehicles near the batch origin
            # (Simulating a spatial index query)
            batch_origin = batch[0].origin_node

            # Sort vehicles by distance to batch origin
            # This is slow O(N), but fine for benchmark setup
            vehicles.sort(key=lambda v: time_matrix[v.available_at_node, batch_origin])

            # Pick top 30 closest vehicles
            candidates = vehicles[:VEHICLES_PER_BATCH]

            for v in candidates:
                checks += 1

                # BFS
                t0 = time.time()
                res_bfs = BFSSolver.solve(batch, v, time_matrix)
                t_bfs_total += time.time() - t0

                # Memo
                t0 = time.time()
                res_memo = MemoSolver.solve(batch, v, time_matrix)
                t_memo_total += time.time() - t0

                if res_bfs.is_feasible != res_memo.is_feasible:
                    print(f"MISMATCH! Size {size}, ReqIDs: {[r.id for r in batch]}")
                    print(f"BFS: {res_bfs.is_feasible}, Memo: {res_memo.is_feasible}")
                    # Optional: Break or raise error

                if res_memo.is_feasible:
                    feasible += 1

        results[size]["bfs"] = t_bfs_total
        results[size]["memo"] = t_memo_total
        results[size]["checks"] = checks
        results[size]["feasible"] = feasible

        speedup = t_bfs_total / t_memo_total if t_memo_total > 0 else 0
        feas_pct = (feasible / checks) * 100
        print(
            f"{size:<5} | {count:<8} | {checks:<8} | {t_bfs_total:<10.4f} | {t_memo_total:<10.4f} | {speedup:<8.2f}x | {feas_pct:.1f}%"
        )

    # Plotting
    sizes = list(BATCH_COUNTS.keys())
    bfs_times = [results[s]["bfs"] for s in sizes]
    memo_times = [results[s]["memo"] for s in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width / 2, bfs_times, width, label="Standard BFS", color="skyblue"
    )
    rects2 = ax.bar(
        x + width / 2, memo_times, width, label="Memoized BFS", color="lightgreen"
    )

    ax.set_ylabel("Total Execution Time (s)")
    ax.set_xlabel("Batch Size (Number of Requests)")
    ax.set_title(
        f"Solver Performance by Batch Size\n(Overlap: {VEHICLES_PER_BATCH} vehicles/batch)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)  # type: ignore
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt="%.2f")
    ax.bar_label(rects2, padding=3, fmt="%.2f")

    output_path = "data/plots/realistic_benchmark.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    run_realistic_benchmark()
