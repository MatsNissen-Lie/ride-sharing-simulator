import os
import time
import config.config as config
from config import BATCH_DURATION, get_logger, NUMBER_OF_VEHICLES
from config.config import (
    MIN_START_TIME,
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

from src.utils.stress_test_plot import plot_stress_test_comparison

logger = get_logger("sim_runner")


def run_simulation(
    run_name: str,
    provider: SimulationDataProvider,
    db_cache: DuckDBPathCache,
    network_service: NetworkXService,
    planner: RoutePlanner,
    omega: float = OMEGA,
    delta: float = DELTA,
    mega_batch_mode: bool = MEGA_BATCH_MODE,
    mega_batch_size: int = MEGA_BATCH_SIZE,
    num_vehicles: int = NUMBER_OF_VEHICLES,
    batch_duration: int = BATCH_DURATION,
    base_output_dir: str = "reports",
) -> dict | None:
    """
    Runs a single simulation scenario with specified parameters.
    Returns a dictionary with runtime stats if successful, else None.
    """
    logger.info(f"=== Starting Scenario: {run_name} ===")
    logger.info(
        f"Params: omega={omega}, delta={delta}, mega_mode={mega_batch_mode}, mega_size={mega_batch_size}, vehicles={num_vehicles}, batch_duration={batch_duration}"
    )

    output_manager = SimulationOutputManager(
        base_dir=base_output_dir, run_name=run_name
    )
    output_manager.save_metadata(config)

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

    from src.utils.stats_logger import SimulationStatsLogger
    from src.interfaces.stats import BatchStatistics

    stats_logger = SimulationStatsLogger()
    final_stats: BatchStatistics | None = None

    try:
        for stats in simulator.run(
            request_iterator=req_iter,
            batch_duration=batch_duration,
            min_start_time=MIN_START_TIME,
            warmup_duration=0,
            mega_batch_mode=mega_batch_mode,
            mega_batch_size=mega_batch_size,
            max_batches=1,
        ):
            if isinstance(stats, BatchStatistics):
                if not stats.is_warmup:
                    stats_logger.log_batch(stats)
                    # For mega batch, we capture this specific batch stats
                    if mega_batch_mode:
                        final_stats = stats
                        logger.info("✅ Mega batch processed. Stopping simulation.")
                        break
                else:
                    logger.info(f"🔥 Warm-up batch {stats.batch_id}. Stats NOT logged.")

        # Final Flush/Export
        csv_path = os.path.join(output_manager.get_run_dir(), "simulation_stats.csv")
        stats_logger.flush_to_csv(csv_path)
        stats_logger.print_summary()

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        raise e

    logger.info(f"=== Scenario {run_name} Completed ===\n")

    if final_stats:
        return {
            "name": run_name,
            "total_time": final_stats.time_total_batch,
            "breakdown": {
                "RV Graph": final_stats.time_rv,
                "RTV Graph": final_stats.time_rtv,
                "Assignment": final_stats.time_assign,
                "Rebalancing": final_stats.time_rebal,
            },
        }
    return None


if __name__ == "__main__":
    # Define stress test scenarios
    OMEGA, DELTA = 300 * 2, 600 * 2  # Base values
    FACTOR = 2
    BATCH_SIZE = 150
    scenarios = [
        {
            "name": f"Ω,Δ = {OMEGA / FACTOR},{DELTA / FACTOR}",
            "omega": OMEGA / FACTOR,
            "delta": DELTA / FACTOR,
        },
        {
            "name": f"Ω,Δ = {OMEGA},{DELTA}",
            "omega": OMEGA,
            "delta": DELTA,
        },
        {
            "name": f"Ω,Δ = {900},{1800}",
            "omega": 900,
            "delta": 1800,
        },
        {
            "name": f"Ω,Δ = {OMEGA * FACTOR},{DELTA * FACTOR}",
            "omega": OMEGA * FACTOR,
            "delta": DELTA * FACTOR,
        },
    ]

    # Create a shared timestamped folder for this batch of runs
    timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    batch_run_dir = os.path.join("reports", f"constraint_test_{timestamp_str}")
    os.makedirs(batch_run_dir, exist_ok=True)
    logger.info(f"📂 Saving all runs in: {batch_run_dir}")

    # Initialize Core Components ONCE
    # Use max required vehicles to ensure provider has enough
    MAX_VEHICLES = 2500
    logger.info("Initializing shared simulation components (Graph, DB, Planner)...")
    provider = SimulationDataProvider(vehicle_count=MAX_VEHICLES)
    db_cache = DuckDBPathCache(read_only=True)
    network_service = NetworkXService(provider.graph, db=db_cache)
    planner = RoutePlanner(network_service=network_service)

    results = []
    try:
        for s in scenarios:
            scenario_result = run_simulation(
                run_name=s["name"],
                provider=provider,
                db_cache=db_cache,
                network_service=network_service,
                planner=planner,
                omega=s.get("omega", OMEGA),
                delta=s.get("delta", DELTA),
                batch_duration=s.get("batch_duration", BATCH_DURATION),
                mega_batch_mode=True,
                mega_batch_size=BATCH_SIZE,
                base_output_dir=batch_run_dir,
            )
            if scenario_result:
                results.append(scenario_result)
    finally:
        logger.info("Closing shared DB connection.")
        db_cache.close()

    if results:
        logger.info(f"📊 Generating runtime comparison plot in {batch_run_dir}...")
        plot_stress_test_comparison(
            results,
            batch_run_dir,
            title=f"Simulation Runtime Comparison (Batch Size = {BATCH_SIZE})",
        )
