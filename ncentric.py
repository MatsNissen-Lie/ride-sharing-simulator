from config import BATCH_DURATION, get_logger, NUMBER_OF_VEHICLES
from config.config import MIN_START_TIME
from src.components.batch_controller import BatchController
from src.components.graph_builder.rtv_builder_min import (
    TripCentricRTVGraphBuilder,
)

# from src.components.graph_builder.rtv_builder_trip_centric import (
#     TripCentricRTVGraphBuilder,
# )
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.network_service import NetworkXService
from src.components.rebalance_solver import RebalancingSolver
from src.components.route_planner import RoutePlanner
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.assignment_solver import GurobiAssignmentSolver
from src.database.simple_db import DuckDBPathCache
from src.simulator import Simulator

logger = get_logger(__name__)


def main():
    """
    Main simulation entrypoint.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the ride-sharing simulation.")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Run in read-only mode (do not write to the database).",
    )
    args = parser.parse_args()

    logger.info("🚀 Simulation starting...")
    if args.read_only:
        logger.info("🔒 Running in READ-ONLY mode.")

    # --- 1. Initialize Core Components ---
    logger.info("Initializing components...")
    provider = SimulationDataProvider()

    # check the node ids in the graph
    node_ids = list(provider.graph.nodes)
    logger.info(
        f"Graph has {len(node_ids)} nodes, ranging from {min(node_ids)} to {max(node_ids)}"
    )

    db_cache = DuckDBPathCache(read_only=args.read_only)
    network_service = NetworkXService(provider.graph, db=db_cache)
    planner = RoutePlanner(network_service=network_service)

    # --- 2. Initialize Algorithm Pipeline ---
    controller = BatchController(
        network_service, provider.iter_vehicles(NUMBER_OF_VEHICLES)
    )
    rv_builder = RVGraphBuilder(planner, network_service)
    rtv_builder = TripCentricRTVGraphBuilder(planner)
    assignment_solver = GurobiAssignmentSolver()
    rebalancing_solver = RebalancingSolver(network_service)

    # --- 3. Setup Time ---
    req_iter = provider.iter_requests()

    min_start_time = MIN_START_TIME  # Explicit override from config

    # --- 4. Initialize and Run Simulator ---
    simulator = Simulator(
        provider=provider,
        controller=controller,
        rv_builder=rv_builder,
        rtv_builder=rtv_builder,
        assignment_solver=assignment_solver,
        rebalancing_solver=rebalancing_solver,
        network_service=network_service,
        db_cache=db_cache,
    )

    simulator.run(
        request_iterator=req_iter,
        batch_duration=BATCH_DURATION,
        min_start_time=min_start_time,
    )


if __name__ == "__main__":
    main()
