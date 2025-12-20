from pathlib import Path
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
    END_TIME,
    WARMUP_DURATION,
)
from config.paths import PLOT_DIR
from src.components.batch_controller import BatchController
from src.interfaces import (
    VehicleAvailability,
    IAssignmentSolver,
    IRVGraphBuilder,
    IRTVGraphBuilder,
    INetworkService,
    Request,
    MetaRequest,
)
from src.components.simulation_data_provider import SimulationDataProvider
from src.interfaces.stats import BatchStatistics
from src.utils.rtv_plot import plot_rtv_graph
from src.utils.rv_plot import plot_rv_graph
from src.utils.stats_logger import SimulationStatsLogger

logger = get_logger(__name__)


class Simulator:
    def __init__(
        self,
        provider: SimulationDataProvider,
        controller: BatchController,
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
        self.read_only = read_only

    def run(
        self,
        request_iterator: Iterator[MetaRequest],
        min_start_time: float,
        batch_duration: int = BATCH_DURATION,
        warmup_duration: float = WARMUP_DURATION,
        mega_batch_mode: bool = MEGA_BATCH_MODE,
        mega_batch_size: int = MEGA_BATCH_SIZE,
        max_batches: int | float = MAX_BATCHES,
    ) -> Iterator[BatchStatistics]:
        logger.info("🚀 Simulation starting...")

        batch_count = 0
        self.mega_pool: list[Request] = []

        # No try/finally block needed here as we yield control back to caller
        for batch in self.controller.iter_batches(
            request_iterator, batch_duration, min_start_time=min_start_time
        ):
            t_now, all_requests_in_pool, vehicle_avail_pool = batch

            if t_now >= END_TIME:
                logger.info(f"🛑 Reached end time ({END_TIME}). Stopping simulation.")
                break

            batch_datetime = self.provider.get_simulation_clock().from_relative_seconds(
                t_now
            )

            if (
                len(all_requests_in_pool) < MIN_REQUESTS_TO_PROCESS
                and not mega_batch_mode
            ):
                continue

            if mega_batch_mode:
                # Accumulate requests
                for req in all_requests_in_pool:
                    if req not in self.mega_pool:
                        self.mega_pool.append(req)

                if len(self.mega_pool) < mega_batch_size:
                    logger.info(
                        f"🧬 Accumulating requests... [{len(self.mega_pool)}/{mega_batch_size}]"
                    )
                    continue  # Skip processing until we reach the threshold

                # filter so we have exactly mega_batch_size requests
                self.mega_pool = self.mega_pool[:mega_batch_size]

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
                        latest_dropoff_time=new_req_time + original_duration + DELTA,
                    )
                    refreshed_requests.add(new_req)
                    # Inject into controller to bypass expiration
                    self.controller._request_index[new_req.id] = new_req

                # Override the current batch connection
                all_requests_in_pool = refreshed_requests
                self.mega_pool = []  # Reset for next mega batch (or we could break)

            batch_count += 1
            if max_batches is not None and batch_count > max_batches:
                logger.info(
                    f"🛑 Reached max batches limit ({max_batches}). Stopping simulation at batch {batch_count}."
                )
                break

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

            # --- Capture & Aggregated Stats ---
            c_stats = self.controller.get_and_reset_metrics()

            # 1. Candidate Trip Stats (from RTV Graph)
            # trip_counts_by_size is a dict {size: count}
            candidate_trip_stats, _ = rtv_graph.get_trip_stats()

            # 2. Assigned Trip Stats (from Solver Result)
            assigned_trip_stats = {}
            for assignment in assignment_result.assignments:
                size = len(assignment.trip_id)
                assigned_trip_stats[size] = assigned_trip_stats.get(size, 0) + 1

            is_warmup = t_now < (min_start_time + warmup_duration)

            batch_stats = BatchStatistics(
                batch_id=batch_count,
                sim_time=t_now,
                timestamp=batch_datetime,
                is_warmup=is_warmup,
                requests_new=c_stats.new_requests_count,
                requests_active=len(all_requests_in_pool),
                requests_delivered=c_stats.delivered_count,
                requests_expired=c_stats.expired_count,
                vehicles_active=active_vehicles_count,
                vehicles_idle=idle_vehicles_count,
                avg_wait_time=c_stats.avg_wait_time,
                avg_total_delay=c_stats.avg_total_delay,
                avg_expired_wait=c_stats.avg_expired_wait,
                time_rv=rv_duration,
                time_rtv=rtv_duration,
                time_assign=assignment_duration,
                time_rebal=rebalancing_duration,
                time_total_batch=loop_duration,
                rv_edges=len(rv_graph.rv_details),
                trips_count=total_trips,
                objective_value=float(assignment_result.total_cost),
                # Expanded Trip Stats
                candidate_trips_size_1=candidate_trip_stats.get(1, 0),
                candidate_trips_size_2=candidate_trip_stats.get(2, 0),
                candidate_trips_size_3=candidate_trip_stats.get(3, 0),
                candidate_trips_size_4=candidate_trip_stats.get(4, 0),
                candidate_trips_size_5=candidate_trip_stats.get(5, 0),
                candidate_trips_size_6=candidate_trip_stats.get(6, 0),
                assigned_trips_size_1=assigned_trip_stats.get(1, 0),
                assigned_trips_size_2=assigned_trip_stats.get(2, 0),
                assigned_trips_size_3=assigned_trip_stats.get(3, 0),
                assigned_trips_size_4=assigned_trip_stats.get(4, 0),
                assigned_trips_size_5=assigned_trip_stats.get(5, 0),
                assigned_trips_size_6=assigned_trip_stats.get(6, 0),
            )

            if c_stats.delivered_count > 0:
                logger.info(f"📦 Delivered {c_stats.delivered_count} passengers.")
            if c_stats.expired_count > 0:
                logger.info(f"💀 Expired {c_stats.expired_count} requests.")

            # --- Yield Stats ---
            yield batch_stats
