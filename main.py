import time

from config import BATCH_DURATION, get_logger, NUMBER_OF_VEHICLES
from config.config import MIN_REQUESTS_TO_PROCESS, MIN_START_TIME, PLOT_RTV_GRAPH
from config.paths import PLOT_DIR
from src.components.batch_controller import BatchController
from old_code.rtv_graph_builder import RTVGraphBuilder

# from src.components.rv_graph_builder import RVGraphBuilder

from src.components.graph_builder.rv_builder import RVGraphBuilder


from src.components.network_service import NetworkXService
from src.components.route_planner import RoutePlanner
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.assignment_solver import GurobiAssignmentSolver
from src.interfaces.batch_control import IBatchController
from src.interfaces.services import (
    IAssignmentSolver,
    IRVGraphBuilder,
    IRTVGraphBuilder,
)
from src.utils.rtv_plot import plot_rtv_graph
from src.utils.rv_plot import plot_rv_graph


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

    # check the node ids in the graph
    node_ids = list(provider.graph.nodes)
    logger.info(
        f"Graph has {len(node_ids)} nodes, ranging from {min(node_ids)} to {max(node_ids)}"
    )

    # Shared DB cache (Dependency Injection)
    from src.database.simple_db import DuckDBPathCache

    db_cache = DuckDBPathCache()

    network_service = NetworkXService(provider.graph, db=db_cache)
    planner = RoutePlanner(network_service=network_service)

    # --- 3. Initialize Algorithm Pipeline ---
    # (Using the stubs, replace with your real ones)

    # State Management (Your new interface)
    controller = controller or BatchController(
        network_service, provider.iter_vehicles(NUMBER_OF_VEHICLES)
    )
    rv_builder = RVGraphBuilder(planner, network_service)
    rtv_builder = RTVGraphBuilder(planner)
    assignment_solver = GurobiAssignmentSolver()

    batch_count = 0
    batch_count = 0

    req_iter = provider.iter_requests()

    import itertools

    # Peek first element to initialize clock
    try:
        first_req = next(req_iter)
    except StopIteration:
        logger.warning("No requests found.")
        return

    # Now clock is initialized
    clock = provider.get_simulation_clock()

    start_dt = clock.from_relative_seconds(0)  # This is the time of first request?

    if start_dt.hour >= 12:
        # jump to next day 6 AM
        logger.info("First request after noon, starting simulation next day 6 AM.")
        start_dt = (
            start_dt.replace(
                day=start_dt.day + 1, hour=6, minute=0, second=0, microsecond=0
            )
            if start_dt.day < 28
            else start_dt.replace(
                year=start_dt.year + (start_dt.month // 12),
                month=(start_dt.month % 12) + 1,
                day=1,
                hour=6,
                minute=0,
                second=0,
                microsecond=0,
            )
        )

    six_am = start_dt.replace(hour=6, minute=0, second=0, microsecond=0)

    min_start_time = clock.to_relative_seconds(six_am)
    min_start_time = MIN_START_TIME
    print(
        f"Simulation starting at {provider.get_simulation_clock().from_relative_seconds(min_start_time).isoformat()[:-6]}",
        min_start_time,
    )

    # Reconstruct iterator
    full_req_iter = itertools.chain([first_req], req_iter)

    for batch in controller.iter_batches(
        full_req_iter, BATCH_DURATION, min_start_time=min_start_time
    ):
        batch_start = time.perf_counter()
        batch_count += 1
        t_now, all_requests_in_pool, vehicle_avail_pool = batch
        batch_datetime = provider.get_simulation_clock().from_relative_seconds(t_now)

        if len(all_requests_in_pool) < MIN_REQUESTS_TO_PROCESS:
            continue

        start_loop_time = time.perf_counter()
        logger.info("--- Batch %d starting at t=%.2f ---", batch_count, t_now)
        logger.info(f"--------- {batch_datetime.isoformat()[:-6]} ---------")
        logger.info("Found %s new requests in this batch.", len(all_requests_in_pool))

        # --- SESSION MANAGEMENT: READ-ONLY ---
        # Close any existing RW session and switch to RO for graph building
        # This releases the lock so parallel workers (if any) can read.
        db_cache.close()
        db_cache.connect(read_only=True)

        # 3. Build RV Graph (Algorithm Step 1)
        rv_graph = rv_builder.build_graph(all_requests_in_pool, vehicle_avail_pool)
        batch_rv_end = time.perf_counter()
        logger.info(
            f"RV graph built in {batch_rv_end - batch_start:.4f} seconds with {len(rv_graph.request_vehicle)} rv-edges and {len(rv_graph.request_request)} rr-edges.",
        )

        # 4. Build RTV Graph (Algorithm Step 2)
        rtv_graph = rtv_builder.build_graph(
            rv_graph, vehicle_avail_pool, all_requests_in_pool, max_trip_size=4
        )
        batch_rtv_end = time.perf_counter()
        logger.info(
            "RTV graph built in %.4f seconds.",
            batch_rtv_end - batch_rv_end,
        )
        # log the number of trips. I want to know how many trips of each lenghth
        trip_stats = rtv_graph.get_trip_stats()
        logger.info(
            f"RTV graph trip stats: {', '.join(f'size {k}: {v}' for k, v in trip_stats.items())}"
        )
        # logg number of trips and num
        if PLOT_RTV_GRAPH:
            plot_rv_graph(
                rv_graph,
                vehicle_avail_pool,
                all_requests_in_pool,
                provider.graph,
                output_dir=PLOT_DIR,
                datetime=batch_datetime,
            )
        if PLOT_RTV_GRAPH:
            plot_rtv_graph(
                rtv_graph,
                vehicle_avail_pool,
                all_requests_in_pool,
                provider.graph,
                output_dir=PLOT_DIR,
                datetime=batch_datetime,
            )

        # --- SESSION MANAGEMENT: READ-WRITE ---
        # Switch back to RW for assignment (if it needs to write) or just to be ready for next batch
        # Note: If assignment solver writes to DB, we need RW.
        # If not, we could stay in RO until next batch preparation?
        # But NetworkXService might write-through cache misses during assignment cost lookups?
        # Safest to switch back to RW here.
        db_cache.close()
        db_cache.connect(read_only=False)

        # 5. Solve Assignment (Algorithm Step 3)
        if assignment_solver is None:
            raise RuntimeError("Assignment solver must be provided.")
        assignment_result = assignment_solver.solve(
            rtv_graph,
            all_requests_in_pool,
            vehicle_avail_pool,
            solver_timeout=BATCH_DURATION * 1.2,  # Give 120% of batch time
            current_time=t_now,  # Pass current time
        )

        logger.info(
            "Assignment solved in %.4f seconds.",
            time.perf_counter() - batch_rtv_end,
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
