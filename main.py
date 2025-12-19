import itertools
import config.config as config  # Import config module for metadata saving
from config import BATCH_DURATION, get_logger, NUMBER_OF_VEHICLES
from config.config import MIN_START_TIME, WARMUP_DURATION
from src.components.batch_controller import BatchController
from src.components.graph_builder.rtv_builder import RTVGraphBuilder
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.network_service import NetworkXService
from src.components.rebalance_solver import RebalancingSolver
from src.components.route_planner import RoutePlanner
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.assignment_solver import GurobiAssignmentSolver
from src.database.simple_db import DuckDBPathCache
from src.simulator import Simulator
from src.utils.output_manager import SimulationOutputManager

logger = get_logger(__name__)


def main():
    """
    Main simulation entrypoint.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run the ride-sharing simulation.")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Run in read-only mode (do not write to the database).",
    )
    args = parser.parse_args()

    # --- 0. Output Management ---
    output_manager = SimulationOutputManager(base_dir="reports", run_name="sim_standard")
    output_manager.save_metadata(config)
    
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
    rtv_builder = RTVGraphBuilder(planner)
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
    
    # --- 5. Run with Stats Logging ---
    from src.utils.stats_logger import SimulationStatsLogger
    from src.interfaces.stats import BatchStatistics
    
    stats_logger = SimulationStatsLogger()
    
    try:
        for stats in simulator.run(
            request_iterator=req_iter,
            batch_duration=BATCH_DURATION,
            min_start_time=min_start_time,
            warmup_duration=WARMUP_DURATION,
            output_dir=output_manager.get_run_dir(),
        ):
            if isinstance(stats, BatchStatistics):
                # Only log if not warmup (logic moved from Simulator)
                # Simulator yields all batches, but flags warmup.
                if not stats.is_warmup:
                    stats_logger.log_batch(stats)
                    # Periodic flush every 20 batches
                    if stats.batch_id % 20 == 0:
                        csv_path = os.path.join(output_manager.get_run_dir(), "simulation_stats.csv")
                        stats_logger.flush_to_csv(csv_path)
                else:
                    logger.info(f"🔥 Warm-up batch {stats.batch_id}. Stats NOT logged.")

        logger.info("🏁 Simulation complete.")
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Simulation stopped by user (Ctrl+C).")

    finally:
        # Final Flush/Export
        import os
        csv_path = os.path.join(output_manager.get_run_dir(), "simulation_stats.csv")
        # flush_to_csv will append any remaining batches
        stats_logger.flush_to_csv(csv_path)
        stats_logger.print_summary()

    # --- 6. Generate Plots ---
    logger.info("📊 Generating plots...")
    from src.utils.generate_plots import generate_plots
    import os
    csv_path = os.path.join(output_manager.get_run_dir(), "simulation_stats.csv")
    if os.path.exists(csv_path):
        generate_plots(csv_path)
    else:
        logger.warning("No stats CSV found, skipping plot generation.")


if __name__ == "__main__":
    main()
