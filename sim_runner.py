import os
import time
import config.config as config
from config import BATCH_DURATION, get_logger, NUMBER_OF_VEHICLES
from config.config import (
    MIN_START_TIME,
    WARMUP_DURATION,
    OMEGA,
    DELTA,
    MEGA_BATCH_MODE,
    MEGA_BATCH_SIZE,
)
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
from src.utils.generate_plots import generate_plots
from src.utils.comparison_analytics import generate_comparisons

logger = get_logger("sim_runner")


def run_simulation(
    run_name: str,
    omega: float = OMEGA,
    delta: float = DELTA,
    mega_batch_mode: bool = MEGA_BATCH_MODE,
    mega_batch_size: int = MEGA_BATCH_SIZE,
    num_vehicles: int = NUMBER_OF_VEHICLES,
    max_batches: int = config.MAX_BATCHES,
) -> str | None:
    """
    Runs a single simulation scenario with specified parameters.
    """
    logger.info(f"=== Starting Scenario: {run_name} ===")
    logger.info(
        f"Params: omega={omega}, delta={delta}, mega_mode={mega_batch_mode}, mega_size={mega_batch_size}, vehicles={num_vehicles}"
    )

    # Temporary override for MAX_BATCHES if needed
    original_max_batches = config.MAX_BATCHES
    config.MAX_BATCHES = max_batches

    output_manager = SimulationOutputManager(base_dir="reports", run_name=run_name)

    # Initialize Core Components
    NUMBER_OF_VEHICLES = 2500
    provider = SimulationDataProvider(vehicle_count=NUMBER_OF_VEHICLES)
    db_cache = DuckDBPathCache(read_only=True)
    network_service = NetworkXService(provider.graph, db=db_cache)
    planner = RoutePlanner(network_service=network_service)

    # Initialize Algorithm Pipeline with Injected Params
    controller = BatchController(
        network_service, provider.iter_vehicles(num_vehicles), omega=omega, delta=delta
    )
    rv_builder = RVGraphBuilder(planner, network_service, omega=omega, delta=delta)
    rtv_builder = RTVGraphBuilder(planner)
    assignment_solver = GurobiAssignmentSolver()
    rebalancing_solver = RebalancingSolver(network_service)

    req_iter = provider.iter_requests()

    # Initialize and Run Simulator
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

    try:
        simulator.run(
            request_iterator=req_iter,
            batch_duration=BATCH_DURATION,
            min_start_time=MIN_START_TIME,
            warmup_duration=WARMUP_DURATION,
            output_dir=output_manager.get_run_dir(),
            mega_batch_mode=mega_batch_mode,
            mega_batch_size=mega_batch_size,
            max_batches=max_batches,
        )

        # Generate Plots
        csv_path = os.path.join(output_manager.get_run_dir(), "simulation_stats.csv")
        if os.path.exists(csv_path):
            logger.info(f"📊 Generating plots for {run_name}...")
            generate_plots(csv_path)
    finally:
        db_cache.close()
        # Restore config
        config.MAX_BATCHES = original_max_batches

    logger.info(f"=== Scenario {run_name} Completed ===\n")
    return os.path.join(output_manager.get_run_dir(), "simulation_stats.csv")


if __name__ == "__main__":
    # Define stress test scenarios
    scenarios = [
        {
            "name": "baseline_mega_200",
            "omega": 300,
            "delta": 600,
        },
        {
            "name": "tight_mega_200",
            "omega": 120,
            "delta": 240,
        },
        {
            "name": "loose_mega_200",
            "omega": 600,
            "delta": 1200,
        },
    ]

    # Run each scenario for a limited number of batches for quick verification
    # For a real stress test, you'd increase max_batches
    results_map = {}
    for s in scenarios:
        csv_path = run_simulation(
            run_name=s["name"],
            omega=s.get("omega", OMEGA),
            delta=s.get("delta", DELTA),
            mega_batch_mode=True,
            mega_batch_size=200,
            max_batches=1,  # mega batch is a one batch test #cannot run more than 1 batch
        )
        if csv_path:
            results_map[s["name"]] = csv_path

    if results_map:
        comparison_dir = os.path.join("reports", f"comparison_{int(time.time())}")
        logger.info(f"📊 Generating comparative analytics in {comparison_dir}...")
        generate_comparisons(results_map, comparison_dir)
