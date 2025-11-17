import time

from config import BATCH_DURATION, get_logger
from src.components.batch_controller import BatchController
from src.components.network_service import NetworkXService
from src.components.route_planner import RoutePlanner
from src.components.simulation_data_provider import SimulationDataProvider
from src.interfaces import RTVGraph, RVGraph
from src.interfaces.batch_control import IBatchController
from src.interfaces.data_models import MetaRequest, Request, VehicleAvailability
from src.interfaces.services import (
    IAssignmentSolver,
    IRVGraphBuilder,
    IRTVGraphBuilder,
)


logger = get_logger(__name__)

# =========================================================================
# MAIN SIMULATION LOOP
# =========================================================================


def main(
    controller: IBatchController | None = None,
    rv_builder: IRVGraphBuilder | None = None,
    rtv_builder: IRTVGraphBuilder | None = None,
    assignment_solver: IAssignmentSolver | None = None,
):
    """
    Main simulation entrypoint.
    """
    logger.info("🚀 Simulation starting...")

    # --- 2. Initialize Core Components ---
    logger.info("Initializing components...")
    provider = SimulationDataProvider()
    network_service = NetworkXService(provider.graph)
    planner = RoutePlanner(network_service=network_service)

    # --- 3. Initialize Algorithm Pipeline ---
    # (Using the stubs, replace with your real ones)

    # State Management (Your new interface)
    controller = controller or BatchController(network_service)

    if rv_builder is None or rtv_builder is None or assignment_solver is None:
        raise RuntimeError(
            "RV/RTV graph builders and assignment solver must be provided."
        )

    # --- 4. Load & Sort Simulation Data ---
    # For a real sim, you'd stream this, but for a loop
    # demo, we load and sort all requests by time.
    logger.info("Loading and sorting simulation data...")
    all_requests = sorted(provider.iter_requests(), key=lambda r: r.request_time)
    if not all_requests:
        logger.warning("No requests found. Exiting.")
        return

    # --- 5. Run the Main Loop ---
    t_now = all_requests[0].request_time
    t_end = all_requests[-1].request_time

    request_idx = 0  # To track which requests we've processed

    logger.info("Simulation loop starting from t=%s to t=%s...", t_now, t_end)
    while t_now <= t_end:
        start_loop_time = time.perf_counter()
        logger.info("--- Batch starting at t=%.2f ---", t_now)

        # 1. Collect new requests for this batch
        t_batch_end = t_now + BATCH_DURATION
        new_meta_requests: list[MetaRequest] = []
        while (
            request_idx < len(all_requests)
            and all_requests[request_idx].request_time < t_batch_end
        ):
            new_meta_requests.append(all_requests[request_idx])
            request_idx += 1

        logger.info("Found %s new requests in this batch.", len(new_meta_requests))

        # 2. Prepare Batch (Controller Step 1)
        #    - Projects fleet state
        #    - Converts new meta_requests
        #    - Gathers all promised requests
        all_requests_in_pool, vehicle_avail_pool = controller.prepare_batch(
            t_now, new_meta_requests
        )
        logger.info(
            "Controller prepared batch: %s total requests, %s available vehicles.",
            len(all_requests_in_pool),
            len(vehicle_avail_pool),
        )

        if not all_requests_in_pool:
            logger.info("No active requests. Advancing time.")
            t_now += BATCH_DURATION
            continue

        # 3. Build RV Graph (Algorithm Step 1)
        rv_graph = rv_builder.build_graph(all_requests_in_pool, vehicle_avail_pool)

        # 4. Build RTV Graph (Algorithm Step 2)
        rtv_graph = rtv_builder.build_graph(
            rv_graph, vehicle_avail_pool, max_trip_size=4
        )

        # 5. Solve Assignment (Algorithm Step 3)
        assignment_result = assignment_solver.solve(
            rtv_graph,
            all_requests_in_pool,
            vehicle_avail_pool,
            solver_timeout=BATCH_DURATION * 1.2,  # Give 120% of batch time
        )

        # 6. Commit Assignment (Controller Step 2)
        #    - Applies the 'assignment_result' to the internal fleet state
        #    - Updates/creates promises for assigned requests
        controller.commit_assignment(t_now, assignment_result)

        # 7. Advance time
        loop_duration = time.perf_counter() - start_loop_time
        logger.info("Batch completed in %.4f seconds.", loop_duration)
        t_now += BATCH_DURATION

    logger.info("🏁 Simulation complete.")


if __name__ == "__main__":
    main()
