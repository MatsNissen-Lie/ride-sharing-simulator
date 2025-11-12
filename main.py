import time
from typing import Iterable, Set, Tuple

# --- Import your components ---
# (Assuming your project structure)
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.network_service import NetworkXService
from src.components.route_planner import RoutePlanner  # Your existing planner

# --- Import all interfaces ---
from src.interfaces.data_models import (
    AssignmentResult,
    MetaRequest,
    Request,
    VehicleAvailability,
    RTVGraph,  # Assuming these are defined
    RVGraph,
)
from src.interfaces.services import (
    IBatchController,  # Your new controller interface
    INetworkService,
    IRVGraphBuilder,
    IRTVGraphBuilder,
    IAssignmentSolver,
    ISingleVehicleRoutePlanner,
)
from config.config import BATCH_DURATION

# =========================================================================
# MAIN SIMULATION LOOP
# =========================================================================


def main():
    """
    Main simulation entrypoint.
    """
    print("🚀 Simulation starting...")

    # --- 2. Initialize Core Components ---
    print("Initializing components...")
    provider = SimulationDataProvider()
    network_service = NetworkXService(provider.graph)
    planner = RoutePlanner(network_service=network_service)

    # --- 3. Initialize Algorithm Pipeline ---
    # (Using the stubs, replace with your real ones)

    # State Management (Your new interface)
    controller = ConcreteBatchController(network_service)

    # Step 1: RV Builder
    rv_builder = ConcreteRVGraphBuilder(planner)

    # Step 2: RTV Builder
    rtv_builder = ConcreteRTVGraphBuilder(planner)

    # Step 3: ILP Solver
    assignment_solver = ConcreteAssignmentSolver()

    # --- 4. Load & Sort Simulation Data ---
    # For a real sim, you'd stream this, but for a loop
    # demo, we load and sort all requests by time.
    print("Loading and sorting simulation data...")
    all_requests = sorted(provider.iter_requests(), key=lambda r: r.request_time)
    if not all_requests:
        print("No requests found. Exiting.")
        return

    # --- 5. Run the Main Loop ---
    t_now = all_requests[0].request_time
    t_end = all_requests[-1].request_time

    request_idx = 0  # To track which requests we've processed

    print(f"Simulation loop starting from t={t_now} to t={t_end}...")
    while t_now <= t_end:
        start_loop_time = time.perf_counter()
        print(f"\n--- Batch starting at t={t_now:.2f} ---")

        # 1. Collect new requests for this batch
        t_batch_end = t_now + BATCH_DURATION_SECONDS
        new_meta_requests: list[MetaRequest] = []
        while (
            request_idx < len(all_requests)
            and all_requests[request_idx].request_time < t_batch_end
        ):
            new_meta_requests.append(all_requests[request_idx])
            request_idx += 1

        print(f"Found {len(new_meta_requests)} new requests in this batch.")

        # 2. Prepare Batch (Controller Step 1)
        #    - Projects fleet state
        #    - Converts new meta_requests
        #    - Gathers all promised requests
        all_requests_in_pool, vehicle_avail_pool = controller.prepare_batch(
            t_now, new_meta_requests
        )
        print(
            f"Controller prepared batch: {len(all_requests_in_pool)} total requests, {len(vehicle_avail_pool)} available vehicles."
        )

        if not all_requests_in_pool:
            print("No active requests. Advancing time.")
            t_now += BATCH_DURATION
            continue

        # 3. Build RV Graph (Algorithm Step 1)
        rv_graph = rv_builder.build_graph(all_requests_in_pool, vehicle_avail_pool)

        # 4. Build RTV Graph (Algorithm Step 2)
        rtv_graph = rtv_builder.build_graph(
            rv_graph, vehicle_avail_pool, max_trip_size=10
        )

        # 5. Solve Assignment (Algorithm Step 3)
        assignment_result = assignment_solver.solve(
            rtv_graph,
            all_requests_in_pool,
            vehicle_avail_pool,
            solver_timeout=BATCH_DURATION_SECONDS * 0.8,  # Give 80% of batch time
        )

        # 6. Commit Assignment (Controller Step 2)
        #    - Applies the 'assignment_result' to the internal fleet state
        #    - Updates/creates promises for assigned requests
        controller.commit_assignment(t_now, assignment_result)

        # 7. Advance time
        loop_duration = time.perf_counter() - start_loop_time
        print(f"Batch completed in {loop_duration:.4f} seconds.")
        t_now += BATCH_DURATION_SECONDS

    print("\n🏁 Simulation complete.")


if __name__ == "__main__":
    main()
