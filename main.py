import time

from config import BATCH_DURATION, get_logger, NUMBER_OF_VEHICLES
from src.components.batch_controller import BatchController
from src.components.graph_builders import RVGraphBuilder
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
    controller = controller or BatchController(
        network_service, set(provider.iter_vehicles(NUMBER_OF_VEHICLES))
    )
    rv_builder = RVGraphBuilder(planner)

    if rv_builder is None or rtv_builder is None or assignment_solver is None:
        raise RuntimeError(
            "RV/RTV graph builders and assignment solver must be provided."
        )

    batch_count = 0
    for batch in controller.iter_batches(provider.iter_requests(), BATCH_DURATION):
        batch_count += 1
        t_now, all_requests_in_pool, vehicle_avail_pool = batch

        start_loop_time = time.perf_counter()
        logger.info("--- Batch %d starting at t=%.2f ---", batch_count, t_now)
        logger.info("Found %s new requests in this batch.", len(all_requests_in_pool))

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
