
import os
import time
from src.components.simulation_data_provider import SimulationDataProvider
from src.database.simple_db import DuckDBPathCache
from src.components.network_service import NetworkXService
from src.components.route_planner import RoutePlanner
from sim_runner import run_simulation
from config import get_logger

# Configure logging to stdout
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger("verify_fix")

def test_simulation_performance():
    # Setup
    MAX_VEHICLES = 100 
    OMEGA, DELTA = 300, 600
    
    logger.info("Initializing components...")
    provider = SimulationDataProvider(vehicle_count=MAX_VEHICLES)
    db_cache = DuckDBPathCache(read_only=True)
    network_service = NetworkXService(provider.graph, db=db_cache)
    planner = RoutePlanner(network_service=network_service)

    # Run for a few batches (implied by short duration or logic)
    # We'll rely on the simulation logic. 
    # run_simulation defaults: mega_batch_mode=False is safer for a quick check? 
    # actually user was using mega_batch_mode in the log provided ? 
    # "Found 154 new requests in this batch" ... "Batch 53 starting" 
    # "mega_batch_mode" isn't explicitly clear from the log snippet but likely.
    
    # We'll run a standard mode but limited time/batches if possible or just let it run one scenario briefly.
    # But run_simulation runs until MIN_START_TIME + WARMUP + 16h by default...
    # We need to monkeypatch END_TIME or MAX_BATCHES in simulator or config?
    # Sim runner allows passing max_batches? No, it uses simulator.run which takes max_batches.
    # sim_runner.py hardcodes simulator.run args?
    
    # Let's inspect sim_runner.py again.
    # It calls simulator.run(..., max_batches=MAX_BATCHES) where MAX_BATCHES is imported from config.
    # Validation script should probably just invoke Simulator directly to control max_batches
    
    # Actually, inspecting sim_runner.py in previous turns:
    # it passes `mega_batch_mode` and `mega_batch_size`.
    # It loops `for stats in simulator.run(...)`.
    # It does NOT appear to accept `max_batches` as an argument to `run_simulation`.
    
    # So I will construct the simulator directly here to keep it short (e.g. 2 batches).
    
    from src.components.batch_controller import BatchController
    from src.components.graph_builder.rv_builder import RVGraphBuilder
    from src.components.graph_builder.rtv_builder import RTVGraphBuilder
    from src.components.assignment_solver import GurobiAssignmentSolver
    from src.components.rebalance_solver import RebalancingSolver
    from src.simulator import Simulator
    from config.config import MIN_START_TIME, WARMUP_DURATION
    
    controller = BatchController(
        network_service, provider.iter_vehicles(MAX_VEHICLES), omega=OMEGA, delta=DELTA
    )
    rv_builder = RVGraphBuilder(planner, network_service, omega=OMEGA, delta=DELTA)
    rtv_builder = RTVGraphBuilder(planner)
    assignment_solver = GurobiAssignmentSolver()
    rebalancing_solver = RebalancingSolver(network_service)

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
    
    req_iter = provider.iter_requests()
    
    logger.info("Starting short simulation (max 3 batches)...")
    start_time = time.time()
    
    count = 0
    for stats in simulator.run(
        request_iterator=req_iter,
        batch_duration=30,
        min_start_time=MIN_START_TIME,
        warmup_duration=0,
        mega_batch_mode=False,
        max_batches=3 # Only 3 batches!
    ):
        logger.info(f"Batch {stats.batch_id} finished. Time: {stats.time_total_batch:.4f}s")
        count += 1
        
    end_time = time.time()
    logger.info(f"Simulation finished in {end_time - start_time:.4f}s")
    
    # Optional: check cache stats if we could access them (network_service._get_cached_path.cache_info())
    cache_info = network_service._get_cached_path.cache_info()
    logger.info(f"Cache Info: {cache_info}")
    
    db_cache.close()

if __name__ == "__main__":
    test_simulation_performance()
