import time
from typing import Iterator, Any
import logging

from config import get_logger
from config.config import (
    DELTA,
    MIN_REQUESTS_TO_PROCESS,
    OMEGA,
    PLOT_RTV_GRAPH,
    MAX_BATCHES,
    MEGA_BATCH_MODE,
    MEGA_BATCH_SIZE,
    BATCH_DURATION,
)
from config.paths import PLOT_DIR
from src.interfaces.batch_control import IBatchController
from src.interfaces.services import (
    IAssignmentSolver,
    IRVGraphBuilder,
    IRTVGraphBuilder,
    IRTVGraphBuilder,
    INetworkService,
    Request,
)
from src.components.simulation_data_provider import SimulationDataProvider
from src.utils.rtv_plot import plot_rtv_graph
from src.utils.rv_plot import plot_rv_graph
from src.utils.stats_logger import SimulationStatsLogger

logger = get_logger(__name__)


class Simulator:
    def __init__(
        self,
        provider: SimulationDataProvider,
        controller: IBatchController,
        rv_builder: IRVGraphBuilder,
        rtv_builder: IRTVGraphBuilder,
        assignment_solver: IAssignmentSolver,
        rebalancing_solver: Any,  # Using Any for now as interface might not be strictly defined in imports
        network_service: INetworkService,
        db_cache: Any,  # Using Any to avoid circular imports or strict typing if not needed here
        read_only: bool = False,
    ):
        self.provider = provider
        self.controller = controller
        self.rv_builder = rv_builder
        self.rtv_builder = rtv_builder
        self.assignment_solver = assignment_solver
        self.rebalancing_solver = rebalancing_solver
        self.network_service = network_service
        self.db_cache = db_cache
        self.stats_logger = SimulationStatsLogger()
        self.read_only = read_only

    def run(
        self,
        request_iterator: Iterator[
            Any
        ],  # Using Any for Request type to avoid import overhead if not needed
        batch_duration: int,
        min_start_time: float,
        warmup_duration: float = 0.0,
        output_dir: str | None = None,
    ):
        logger.info("🚀 Simulation starting...")

        batch_count = 0
        seen_requests = set()
        served_requests = set()
        self.mega_pool: list[Request] = []

        try:
            for batch in self.controller.iter_batches(
                request_iterator, batch_duration, min_start_time=min_start_time
            ):
                batch_start = time.perf_counter()
                batch_count += 1
                t_now, all_requests_in_pool, vehicle_avail_pool = batch

                batch_datetime = (
                    self.provider.get_simulation_clock().from_relative_seconds(t_now)
                )

                if (
                    len(all_requests_in_pool) < MIN_REQUESTS_TO_PROCESS
                    and not MEGA_BATCH_MODE
                ):
                    continue

                if MEGA_BATCH_MODE:
                    # Accumulate requests
                    for req in all_requests_in_pool:
                        if req not in self.mega_pool:
                            self.mega_pool.append(req)

                    if len(self.mega_pool) < MEGA_BATCH_SIZE:
                        logger.info(
                            f"🧬 Accumulating requests... [{len(self.mega_pool)}/{MEGA_BATCH_SIZE}]"
                        )
                        continue  # Skip processing until we reach the threshold

                    # Threshold reached: "Refresh" timestamps and inject back
                    logger.info(
                        f"🧬 MEGA BATCH threshold reached! Refreshing {len(self.mega_pool)} requests..."
                    )

                    refreshed_requests = set()
                    for req in self.mega_pool:
                        # Logic: shift request time to be within the current batch window
                        # We use % BATCH_DURATION to keep the relative offset
                        offset = req.request_time % BATCH_DURATION
                        new_req_time = t_now - BATCH_DURATION + offset

                        # Recalculate dropoff based on original duration
                        original_duration = (
                            req.latest_dropoff_time - DELTA - req.request_time
                        )

                        new_req = Request(
                            id=req.id,
                            origin_node=req.origin_node,
                            destination_node=req.destination_node,
                            request_time=new_req_time,
                            latest_pickup_time=new_req_time + OMEGA,
                            latest_dropoff_time=new_req_time
                            + original_duration
                            + DELTA,
                        )
                        refreshed_requests.add(new_req)
                        # Inject into controller to bypass expiration
                        self.controller._request_index[new_req.id] = new_req

                    # Override the current batch connection
                    all_requests_in_pool = refreshed_requests
                    self.mega_pool = []  # Reset for next mega batch (or we could break)

                start_loop_time = time.perf_counter()
                logger.info("--- Batch %d starting at t=%.2f ---", batch_count, t_now)
                logger.info(f"--------- {batch_datetime.isoformat()[:-6]} ---------")
                logger.info(
                    "Found %s new requests in this batch.", len(all_requests_in_pool)
                )

                # --- SESSION MANAGEMENT: READ-ONLY ---
                # Close any existing RW session and switch to RO for graph building

                if not self.read_only:
                    self.db_cache.close()
                    self.db_cache.connect(read_only=True)

                # 3. Build RV Graph (Algorithm Step 1)
                rv_start_time = time.perf_counter()
                rv_graph = self.rv_builder.build_graph(
                    all_requests_in_pool, vehicle_avail_pool
                )
                rv_end_time = time.perf_counter()
                rv_duration = rv_end_time - rv_start_time

                logger.info(
                    f"RV graph built in {rv_duration:.4f} seconds with {len(rv_graph.rv_details)} rv-edges and {len(rv_graph.rr_details) // 2} rr-edges.",
                )

                # 4. Build RTV Graph (Algorithm Step 2)
                rtv_start_time = time.perf_counter()
                rtv_graph = self.rtv_builder.build_graph(
                    rv_graph, vehicle_avail_pool, all_requests_in_pool
                )
                rtv_end_time = time.perf_counter()
                rtv_duration = rtv_end_time - rtv_start_time

                logger.info(
                    "RTV graph built in %.4f seconds.",
                    rtv_duration,
                )
                # log the number of trips. I want to know how many trips of each lenghth
                trip_stats, _ = rtv_graph.get_trip_stats()

                # Calculate total trips for stats
                total_trips = sum(trip_stats.values())

                logger.info(
                    f"Unique trips per size (request sets): {', '.join(f'size {k}: {v}' for k, v in trip_stats.items())}"
                )

                if PLOT_RTV_GRAPH:
                    plot_rv_graph(
                        rv_graph,
                        vehicle_avail_pool,
                        all_requests_in_pool,
                        self.provider.graph,
                        output_dir=PLOT_DIR,
                        datetime=batch_datetime,
                    )
                if PLOT_RTV_GRAPH:
                    plot_rtv_graph(
                        rtv_graph,
                        vehicle_avail_pool,
                        all_requests_in_pool,
                        self.provider.graph,
                        output_dir=PLOT_DIR,
                        datetime=batch_datetime,
                    )

                # --- SESSION MANAGEMENT: READ-WRITE ---
                if not self.read_only:
                    self.db_cache.close()
                    self.db_cache.connect(read_only=False)

                # 5. Solve Assignment (Algorithm Step 3)
                assignment_start_time = time.perf_counter()
                assignment_result = self.assignment_solver.solve(
                    rtv_graph,
                    all_requests_in_pool,
                    vehicle_avail_pool,
                    solver_timeout=batch_duration * 1.2,
                    current_time=t_now,
                )
                assignment_end_time = time.perf_counter()
                assignment_duration = assignment_end_time - assignment_start_time

                logger.info(
                    "Assignment solved in %.4f seconds.",
                    assignment_duration,
                )

                # --- Track Served Requests ---
                total_delay = 0.0
                newly_served_count = 0
                wait_times = []

                if assignment_result.assignments:
                    for assignment in assignment_result.assignments:
                        total_delay += assignment.cost
                        newly_served_count += len(assignment.trip_id)
                        
                        # assignment.trip_id is a frozen set of request IDs
                        for req_id in assignment.trip_id:
                            served_requests.add(req_id)
                            
                        # Calculate Wait Time by walking the route
                        # 1. Find the vehicle (it is in vehicle_avail_pool)
                        v_avail = next((v for v in vehicle_avail_pool if v.vehicle_id == assignment.vehicle_id), None)
                        
                        if v_avail and assignment.route:
                            current_time = v_avail.available_from_time
                            current_node = v_avail.available_at_node
                            
                            for stop in assignment.route.stops:
                                # Travel time
                                travel_time = self.network_service.get_travel_time(current_node, stop.node)
                                current_time += travel_time
                                current_node = stop.node
                                
                                if stop.is_pickup:
                                    # Find request
                                    # Since we don't have direct access to request object from ID easily without iteration, 
                                    # we rely on 'all_requests_in_pool'.
                                    # Note: this might be slow if pool is huge, but it's fine for now.
                                    req = next((r for r in all_requests_in_pool if r.id == stop.request_id), None)
                                    if req:
                                        wait_time = current_time - req.request_time
                                        wait_times.append(wait_time)


                # 6. Commit Assignment (Controller Step 2)
                self.controller.commit_assignment(t_now, assignment_result)
                commit_end = time.perf_counter()

                # 7. Rebalancing (Controller Step 3)
                idle_vehicles, unassigned_requests = (
                    self.controller.get_rebalancing_candidates(
                        assignment_result, vehicle_avail_pool
                    )
                )
                
                idle_vehicles_count = len(idle_vehicles)
                active_vehicles_count = len(vehicle_avail_pool) - idle_vehicles_count
                
                rebal_candidates_end = time.perf_counter()
                logger.debug(
                    f"Rebalancing candidates identified in {rebal_candidates_end - commit_end:.4f} seconds: {len(idle_vehicles)} idle vehicles, {len(unassigned_requests)} unassigned requests."
                )

                rebalancing_duration = 0.0
                if idle_vehicles and unassigned_requests:
                    logger.debug(
                        f"Rebalancing: {len(idle_vehicles)} idle vehicles, "
                        f"{len(unassigned_requests)} unassigned requests."
                    )

                    rebal_start_time = time.perf_counter()
                    rebalancing_routes = self.rebalancing_solver.solve(
                        idle_vehicles=idle_vehicles,
                        unassigned_requests=unassigned_requests,
                    )
                    rebal_end_time = time.perf_counter()
                    rebalancing_duration = rebal_end_time - rebal_start_time

                    logger.debug(
                        "Rebalancing solved in %.4f seconds.",
                        rebalancing_duration,
                    )

                    if rebalancing_routes:
                        self.controller.apply_rebalancing(rebalancing_routes, t_now)
                        logger.debug(
                            f"Rebalancing committed: {len(rebalancing_routes)} vehicles relocated."
                        )

                # 8. Advance time
                loop_end_time = time.perf_counter()
                loop_duration = loop_end_time - start_loop_time
                logger.info("Batch completed in %.4f seconds.", loop_duration)

                # --- LOG STATS ---
                is_warmup = t_now < (min_start_time + warmup_duration)
                
                stats_payload = {
                    "batch_id": batch_count,
                    "sim_time": t_now,
                    "is_warmup": is_warmup,
                    "requests_count": len(all_requests_in_pool),
                    "newly_served_count": newly_served_count,
                    "active_vehicles_count": active_vehicles_count,
                    "idle_vehicles_count": idle_vehicles_count,
                    "average_wait_time": sum(wait_times) / len(wait_times) if wait_times else 0.0,
                    "rv_build_time": rv_duration,
                    "rv_edges_count": len(rv_graph.rv_details),
                    "rtv_build_time": rtv_duration,
                    "trips_count": total_trips,
                    "assignment_time": assignment_duration,
                    "rebalancing_time": rebalancing_duration,
                    "total_batch_time": loop_duration,
                    "objective_value": assignment_result.total_cost,
                    "total_delay": total_delay,
                }

                # Add breakdown by size
                for size in trip_stats.keys():
                    stats_payload[f"trips_size_{size}"] = trip_stats.get(size, 0)

                # --- Track Unique Requests ---
                for req in all_requests_in_pool:
                    seen_requests.add(req.id)

                self.stats_logger.log_batch(stats_payload)
                
                if is_warmup:
                    logger.info("🔥 Warm-up batch. Stats logged but flagged.")

                if batch_count >= MAX_BATCHES:
                    logger.info(
                        f"🛑 Reached max batches limit ({MAX_BATCHES}). Stopping simulation."
                    )
                    break

            logger.info("🏁 Simulation complete.")
            
            # Export Stats
            import os
            final_output_dir = output_dir if output_dir else PLOT_DIR
            csv_path = os.path.join(final_output_dir, "simulation_stats.csv")
            self.stats_logger.export_to_csv(csv_path)

        except KeyboardInterrupt:
            logger.info("\n🛑 Simulation stopped by user (Ctrl+C).")
            # Attempt to export even on interrupt
            import os
            final_output_dir = output_dir if output_dir else PLOT_DIR
            csv_path = os.path.join(final_output_dir, "simulation_stats.csv")
            self.stats_logger.export_to_csv(csv_path)

        finally:
            self.stats_logger.log_scalar_stat(
                "Total Unique Requests Seen", len(seen_requests)
            )
            self.stats_logger.log_scalar_stat(
                "Total Requests Served", len(served_requests)
            )
            self.stats_logger.print_summary()
