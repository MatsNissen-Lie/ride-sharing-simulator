import argparse
import os
from datetime import datetime

import config.config as config
from config import BATCH_DURATION, get_logger
from src.components.assignment_solver import (
    GurobiAssignmentSolver,
    HighsAssignmentSolver,
)
from src.components.batch_controller import BatchController
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rtv_vehicle_centric import (
    VehicleCentricRTVGraphBuilder,
)
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.network_service import NetworkXService
from src.components.oracle_rebalance_solver import OracleRebalancingSolver
from src.components.rebalance_solver import RebalancingSolver
from src.components.simulation_data_provider import SimulationDataProvider
from src.database.simple_db import DuckDBPathCache
from src.interfaces.stats import BatchStatistics
from src.simulator import Simulator
from src.utils.generate_plots import generate_plots
from src.utils.output_manager import SimulationOutputManager
from src.utils.stats_logger import SimulationStatsLogger

logger = get_logger(__name__)

RTV_BUILDERS = {
    "trip": TripCentricRTVGraphBuilder,
    "vehicle": VehicleCentricRTVGraphBuilder,
}
ASSIGNMENT_SOLVERS = {
    "gurobi": GurobiAssignmentSolver,
    "highs": HighsAssignmentSolver,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ride-sharing simulation.")
    parser.add_argument(
        "--rtv",
        choices=RTV_BUILDERS,
        default="trip",
        help="RTV graph builder: trip-centric (default) or vehicle-centric.",
    )
    parser.add_argument(
        "--solver",
        choices=ASSIGNMENT_SOLVERS,
        default="gurobi",
        help="Assignment ILP solver (HiGHS needs no licence).",
    )
    parser.add_argument(
        "--fleet", type=int, default=config.NUMBER_OF_VEHICLES, help="Fleet size."
    )
    parser.add_argument(
        "--start",
        type=datetime.fromisoformat,
        default=None,
        help="Start of warm-up, local time, e.g. 2025-01-16T17:45 (default: MIN_START_TIME).",
    )
    parser.add_argument(
        "--rebalancing",
        choices=["reactive", "oracle"],
        default="reactive",
        help="Reactive (paper) or perfect-information benchmark.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=10.0,
        help="Oracle look-ahead in minutes.",
    )
    parser.add_argument(
        "--name", default="sim_standard", help="Suffix for the run folder in reports/."
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Simulated hours after warm-up (default: until END_TIME).",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Stop after this many batches."
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Do not write to the route database.",
    )
    return parser.parse_args()


def make_rebalancing_solver(args, provider, network, start_time, end_time):
    if args.rebalancing == "reactive":
        return RebalancingSolver(network)
    horizon = args.horizon * 60
    future = provider.request_origins_between(start_time, end_time + horizon)
    return OracleRebalancingSolver(
        network, future["t"].to_numpy(), future["origin"].to_numpy(), horizon
    )


def main():
    args = parse_args()
    provider = SimulationDataProvider()
    start_time = config.MIN_START_TIME
    if args.start is not None:
        start_time = (args.start - provider.first_request_time()).total_seconds()
    end_time = start_time + config.END_TIME - config.MIN_START_TIME
    if args.hours is not None:
        end_time = start_time + config.WARMUP_DURATION + args.hours * 3600

    output_manager = SimulationOutputManager(base_dir="reports", run_name=args.name)
    metadata = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    metadata |= {
        "rtv_builder": args.rtv,
        "assignment_solver": args.solver,
        "NUMBER_OF_VEHICLES": args.fleet,
        "MIN_START_TIME": start_time,
        "END_TIME": end_time,
        "rebalancing": args.rebalancing,
        "oracle_horizon_min": args.horizon,
    }
    output_manager.save_metadata(metadata)
    csv_path = os.path.join(output_manager.get_run_dir(), "simulation_stats.csv")

    logger.info("Initializing components...")
    db_cache = DuckDBPathCache(read_only=args.read_only)
    network_service = NetworkXService(provider.graph, db=db_cache)

    simulator = Simulator(
        provider=provider,
        controller=BatchController(network_service, provider.iter_vehicles(args.fleet)),
        rv_builder=RVGraphBuilder(network_service),
        rtv_builder=RTV_BUILDERS[args.rtv](network_service),
        assignment_solver=ASSIGNMENT_SOLVERS[args.solver](),
        rebalancing_solver=make_rebalancing_solver(
            args, provider, network_service, start_time, end_time
        ),
        network_service=network_service,
        db_cache=db_cache,
        read_only=args.read_only,
    )

    stats_logger = SimulationStatsLogger()
    try:
        for stats in simulator.run(
            request_iterator=provider.iter_requests(),
            batch_duration=BATCH_DURATION,
            min_start_time=start_time,
            warmup_duration=config.WARMUP_DURATION,
            max_batches=args.max_batches,
            end_time=end_time,
        ):
            if not isinstance(stats, BatchStatistics):
                continue
            if stats.is_warmup:
                logger.info(f"🔥 Warm-up batch {stats.batch_id}. Stats NOT logged.")
                continue
            stats_logger.log_batch(stats)
            if stats.batch_id % 20 == 0:
                stats_logger.flush_to_csv(csv_path)
        logger.info("🏁 Simulation complete.")
    except KeyboardInterrupt:
        logger.info("🛑 Simulation stopped by user (Ctrl+C).")
    finally:
        stats_logger.flush_to_csv(csv_path)
        stats_logger.print_summary()

    if os.path.exists(csv_path):
        logger.info("📊 Generating plots...")
        generate_plots(csv_path)
    else:
        logger.warning("No stats CSV found, skipping plot generation.")


if __name__ == "__main__":
    main()
