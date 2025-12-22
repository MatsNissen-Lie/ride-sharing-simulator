import os
import time
import logging
import traceback
from typing import Any, Dict, Type

# Adjust path if necessary, assuming run from root
import sys

sys.path.append(os.getcwd())

from config import BATCH_DURATION, NUMBER_OF_VEHICLES, OMEGA, DELTA
from config.config import MIN_START_TIME
from src.components.batch_controller import BatchController
from src.components.graph_builder.rtv_builder import RTVGraphBuilder
from src.components.graph_builder.rtv_builder_min import TripCentricRTVGraphBuilder
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.network_service import NetworkXService
from src.components.rebalance_solver import RebalancingSolver
from src.components.route_planner import RoutePlanner
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.assignment_solver import GurobiAssignmentSolver
from src.database.simple_db import DuckDBPathCache
from src.simulator import Simulator
from src.utils.output_manager import SimulationOutputManager
from src.utils.stress_test_plot import plot_stress_test_comparison as plot_breakdown
from src.utils.rtv_comparison_plot import plot_stress_test_comparison as plot_rtv
from src.interfaces.stats import BatchStatistics

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rtv_comparison")


def run_single_comparison(
    scenario_name: str,
    builder_class: Type,
    provider: SimulationDataProvider,
    db_cache: DuckDBPathCache,
    network_service: NetworkXService,
    planner: RoutePlanner,
    omega: float,
    delta: float,
    mega_batch_size: int,
    base_output_dir: str,
) -> Dict[str, Any] | None:

    run_id = f"{scenario_name}_{builder_class.__name__}"
    logger.info(f"=== Starting Run: {run_id} ===")

    output_manager = SimulationOutputManager(base_dir=base_output_dir, run_name=run_id)
    # Save explicit parameters
    run_params = {
        "SCENARIO_NAME": scenario_name,
        "BUILDER": builder_class.__name__,
        "OMEGA": omega,
        "DELTA": delta,
        "MEGA_BATCH_SIZE": mega_batch_size,
    }
    output_manager.save_metadata(run_params)

    # 1. Controller
    # iter_vehicles returns a generator, so we can call it multiple times for fresh iterators
    controller = BatchController(
        network_service,
        provider.iter_vehicles(NUMBER_OF_VEHICLES),
        omega=omega,
        delta=delta,
    )

    # 2. Builders
    rv_builder = RVGraphBuilder(planner, network_service, omega=omega, delta=delta)

    # Instantiate the specific RTV builder
    if builder_class == RTVGraphBuilder:
        rtv_builder = RTVGraphBuilder(planner)
    elif builder_class == TripCentricRTVGraphBuilder:
        rtv_builder = TripCentricRTVGraphBuilder(planner)
    else:
        raise ValueError(f"Unknown builder class: {builder_class}")

    assignment_solver = GurobiAssignmentSolver()
    rebalancing_solver = RebalancingSolver(network_service)

    req_iter = provider.iter_requests()

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

    # stats_logger = SimulationStatsLogger()
    final_stats: BatchStatistics | None = None

    try:
        # Run for 1 batch (Mega Batch Mode)
        for stats in simulator.run(
            request_iterator=req_iter,
            batch_duration=BATCH_DURATION,
            min_start_time=MIN_START_TIME,
            mega_batch_mode=True,  # Force True as per requirement
            mega_batch_size=mega_batch_size,
            warmup_duration=0,
            max_batches=1,
        ):
            if isinstance(stats, BatchStatistics):
                if not stats.is_warmup:
                    final_stats = stats
                    logger.info("✅ Batch processed.")
                    # break is handled by max_batches=1 inside simulator now,
                    # but explicit break is safer if simulator logic changes
                    break

    except Exception as e:
        logger.error(f"❌ Error in run {run_id}: {e}")
        traceback.print_exc()
        return None

    if final_stats:
        return {
            "name": run_id,
            "scenario": scenario_name,
            "builder": builder_class.__name__,
            "total_time": final_stats.time_total_batch,
            "breakdown": {
                "RV Graph": final_stats.time_rv,
                "RTV Graph": final_stats.time_rtv,
                "Assignment": final_stats.time_assign,
            },
        }
    return None


if __name__ == "__main__":
    # Scenarios from user request
    # 1. Pool Size = 200 (BATCH_SIZE*2)
    # 2. Pool Size = 200 (Same but maybe different omega/delta? reusing defaults for now)
    # 3. Pool Size = 400

    # Using defaults from config if not specified
    # The user request had some specific logic for "FACTOR=2", "BATCH_SIZE=100"

    BATCH_SIZE = 100
    FACTOR = 2

    scenarios = [
        {"name": "Pool Size 100", "omega": OMEGA, "delta": DELTA, "batch_size": 100},
        {
            "name": "Pool Size 200",
            "omega": OMEGA,
            "delta": DELTA,
            "batch_size": BATCH_SIZE * FACTOR,
        },
        {
            "name": "Pool Size 300",
            "omega": OMEGA,
            "delta": DELTA,
            "batch_size": BATCH_SIZE * 3,
        },
        {
            "name": "Pool Size 400",
            "omega": OMEGA,
            "delta": DELTA,
            "batch_size": BATCH_SIZE * FACTOR * FACTOR,
        },
    ]

    # Output Directory
    timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("reports", f"rtv_comparison_{timestamp_str}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Shared Components
    logger.info("Init shared components...")
    provider = SimulationDataProvider(vehicle_count=2500)
    db_cache = DuckDBPathCache(read_only=True)
    network_service = NetworkXService(provider.graph, db=db_cache)
    planner = RoutePlanner(network_service=network_service)

    results = []

    builders = [RTVGraphBuilder, TripCentricRTVGraphBuilder]

    try:
        for s in scenarios:
            for builder_cls in builders:
                res = run_single_comparison(
                    scenario_name=s["name"],
                    builder_class=builder_cls,
                    provider=provider,
                    db_cache=db_cache,
                    network_service=network_service,
                    planner=planner,
                    omega=s["omega"],
                    delta=s["delta"],
                    mega_batch_size=s["batch_size"],
                    base_output_dir=output_dir,
                )
                if res:
                    results.append(res)

    finally:
        db_cache.close()

    # Generate Plots
    if results:
        import json

        json_path = os.path.join(output_dir, "comparison_results.json")
        try:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"✅ Saved results to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")

        # Plot 1: Standard Breakdown (Total Runtime + Components)
        plot_breakdown(results, output_dir, title="General Runtime Comparison")

        # Plot 2: Specialized RTV Builder Comparison
        plot_rtv(results, output_dir, title="RTV Builder Performance")
