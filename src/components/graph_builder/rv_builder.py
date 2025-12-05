"""
rv_graph_builder
=================

This module provides an implementation of the pairwise request-vehicle (RV)
graph described in the supplemental material of Alonso-Mora et al. (2017).
The RV-graph captures which pairs of requests may be served together by a
single vehicle and which individual requests may be served by a given
vehicle. Each edge in the graph is annotated with the cost returned from
the single vehicle route planner, allowing subsequent steps (such as trip
construction and assignment) to reason about the feasibility and cost of
possible matches.

The implementation here does not impose any particular limit on the number
of edges per node; however, users may choose to prune the graph to speed up
subsequent computations. See the paper for details on how such pruning can
trade off optimality for runtime.

The central class exported by this module is :class:`RVGraphBuilder`, which
implements the :class:`IRVGraphBuilder` interface from ``src.interfaces``.
It relies on a provided :class:`RoutePlanner` to evaluate feasibility of
pairings. The resulting graph is returned as an instance of
:class:`RVGraph` defined below.
"""

import numpy as np
from typing import cast
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import get_logger
from config.config import DELTA, OMEGA


from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    IRVGraphBuilder,
    Request,
    Stop,
    Route,
    FeasibleRoute,
)
from src.interfaces.graphs import RVGraph
from src.interfaces.services import INetworkService

from src.components.graph_builder.workers import (
    process_rv_chunk,
    RVTask,
    init_worker,
)

logger = get_logger("rv_graph_builder")


class RVGraphBuilder(IRVGraphBuilder):
    def __init__(
        self,
        route_planner: ISingleVehicleRoutePlanner,
        network: INetworkService,
    ) -> None:
        super().__init__(route_planner)
        self._network = network

    def build_graph(
        self, requests: set[Request], vehicles: set[VehicleAvailability]
    ) -> RVGraph:
        graph = RVGraph()

        # Convert to lists (stable indexing)
        req_list = list(requests)
        veh_list = list(vehicles)

        n_req = len(req_list)
        n_veh = len(veh_list)

        if n_req == 0:
            return graph

        # --- PREPARATION: DATA ARRAYS ---
        # Create numpy views for fast access
        req_origins = np.array([r.origin_node for r in req_list])
        req_dests = np.array([r.destination_node for r in req_list])
        req_times = np.array([r.request_time for r in req_list])

        # --- 1. RR PIPELINE (Prune + Solve Integrated) ---

        # A. Full Matrix Fetch (N x N)
        # We fetch this once. We will use it for BOTH pruning AND the solver.
        rr_matrix = self._network.get_travel_time_matrix(
            req_origins.tolist(), req_origins.tolist()
        )

        # B. Vectorized Pruning
        # Condition: Time(i,j) <= Omega OR Time(j,i) <= Omega
        # Note: We only look at upper triangle (k=1) to avoid duplicates (i,j) vs (j,i)
        rr_mask = np.triu((rr_matrix <= OMEGA) | (rr_matrix.T <= OMEGA), k=1)

        rr_idx_i, rr_idx_j = np.where(rr_mask)

        # C. Vectorized Solving
        if len(rr_idx_i) > 0:
            logger.info(f"RR Pruning: evaluating {len(rr_idx_i)} pairs.")

            # OPTIMIZATION: Extract the O->O times we already have!
            # We need both directions: i->j and j->i
            # rr_matrix[i, j] gives time(i->j)
            # rr_matrix[j, i] gives time(j->i) -> This is equivalent to rr_matrix.T[i, j]

            t_o1_o2 = rr_matrix[rr_idx_i, rr_idx_j]
            t_o2_o1 = rr_matrix.T[rr_idx_i, rr_idx_j]

            self._solve_rr_matrix_fast(
                rr_idx_i,
                rr_idx_j,
                req_list,
                req_times,
                req_origins,
                req_dests,
                t_o1_o2,
                t_o2_o1,  # Pass pre-calculated times
                graph,
            )

        # --- 2. RV PIPELINE (Split Execution) ---

        if n_veh > 0:
            veh_locs = [v.available_at_node for v in veh_list]

            # Broadcasting shapes
            v_times_arr = np.array([v.available_from_time for v in veh_list])[
                :, None
            ]  # (V, 1)
            r_times_arr = req_times[None, :]  # (1, R)

            # A. Matrix Fetch & Pruning
            rv_matrix = self._network.get_travel_time_matrix(
                veh_locs, req_origins.tolist()
            )  # (V, R)

            # We use the raw difference for the check first
            raw_diff = (v_times_arr + rv_matrix) - r_times_arr
            rv_mask = raw_diff <= OMEGA

            # Indices of ALL feasible pairs
            v_idx, r_idx = np.where(rv_mask)

            if len(v_idx) > 0:
                # B. Split Execution based on Occupancy
                
                # 1. Compute Occupancy for ALL candidates once
                occupancy = np.array([len(veh_list[i].passengers_onboard) for i in v_idx])
                
                # --- CASE 0: EMPTY (m=0) ---
                mask_0 = (occupancy == 0)
                if np.any(mask_0):
                    self._solve_rv_matrix_empty(
                        v_idx[mask_0], r_idx[mask_0],
                        veh_list, req_list,
                        raw_diff[v_idx[mask_0], r_idx[mask_0]], # delay_vals
                        graph
                    )

                # --- CASE 1: 1 PASSENGER (m=1) ---
                mask_1 = (occupancy == 1)
                if np.any(mask_1):
                    self._solve_rv_matrix_m1(
                        v_idx[mask_1], r_idx[mask_1], 
                        veh_list, req_list, 
                        graph
                    )

                # --- CASE 2+: COMPLEX (m >= 2) ---
                mask_high = (occupancy >= 2)
                if np.any(mask_high):
                    self._process_rv_occupied_parallel(
                        v_idx[mask_high], r_idx[mask_high], 
                        veh_list, req_list, 
                        graph
                    )

        return graph

    def _solve_rr_matrix_fast(
        self,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
        req_list: list[Request],
        req_times: np.ndarray,
        req_origins: np.ndarray,
        req_dests: np.ndarray,
        t_o1_o2: np.ndarray,
        t_o2_o1: np.ndarray,  # Pre-calculated
        graph: RVGraph,
    ):
        """
        Solved RR matrix using pre-fetched O-O times.
        Only fetches O-D and D-D times from cache.
        """
        # 1. Gather remaining data
        # Origins are not needed, we have t_o1_o2
        d1, d2 = req_dests[idx_i], req_dests[idx_j]
        # Origins ARE needed for O->D lookup
        o1, o2 = req_origins[idx_i], req_origins[idx_j]
        t1, t2 = req_times[idx_i], req_times[idx_j]

        # 2. Fetch only missing travel times (O->D, D->D)
        mat = self._network.get_l1_cache()

        def get_v(u, v):
            return mat[u, v]

        t_o1_d1 = get_v(o1, d1)
        t_o1_d2 = get_v(o1, d2)
        t_o2_d1 = get_v(o2, d1)
        t_o2_d2 = get_v(o2, d2)
        t_d1_d2 = get_v(d1, d2)
        t_d2_d1 = get_v(d2, d1)
        t_d1_o2 = get_v(d1, o2)
        t_d2_o1 = get_v(d2, o1)

        # 3. Compute Costs (Same Matrix Logic as before)
        costs = np.full((6, len(idx_i)), np.inf)

        # Helper to apply mask and cost
        def check(perm, p1, p2, d_arr1, d_arr2):
            # Delay = (Arrival - Req) - Direct_Travel
            del1 = (d_arr1 - t1) - t_o1_d1
            del2 = (d_arr2 - t2) - t_o2_d2

            # Also check Pickup delays
            p_del1 = p1 - t1
            p_del2 = p2 - t2

            mask = (
                (del1 <= OMEGA)
                & (del2 <= OMEGA)
                & (p_del1 <= OMEGA)
                & (p_del2 <= OMEGA)
            )
            costs[perm, mask] = del1[mask] + del2[mask]

        # Perm 0: o1->o2->d1->d2
        p1_0, p2_0 = t1, np.maximum(t1 + t_o1_o2, t2)
        d1_0 = p2_0 + t_o2_d1
        d2_0 = d1_0 + t_d1_d2
        check(0, p1_0, p2_0, d1_0, d2_0)

        # Perm 1: o1->o2->d2->d1
        p1_1, p2_1 = t1, np.maximum(t1 + t_o1_o2, t2)
        d2_1 = p2_1 + t_o2_d2
        d1_1 = d2_1 + t_d2_d1
        check(1, p1_1, p2_1, d1_1, d2_1)

        # Perm 2: o2->o1->d1->d2
        p2_2, p1_2 = t2, np.maximum(t2 + t_o2_o1, t1)
        d1_2 = p1_2 + t_o1_d1
        d2_2 = d1_2 + t_d1_d2
        check(2, p1_2, p2_2, d1_2, d2_2)

        # Perm 3: o2->o1->d2->d1
        p2_3, p1_3 = t2, np.maximum(t2 + t_o2_o1, t1)
        d2_3 = p1_3 + t_o1_d2
        d1_3 = d2_3 + t_d2_d1
        check(3, p1_3, p2_3, d1_3, d2_3)

        # Perm 4: o1->d1->o2->d2
        p1_4 = t1
        d1_4 = p1_4 + t_o1_d1
        p2_4 = np.maximum(d1_4 + t_d1_o2, t2)
        d2_4 = p2_4 + t_o2_d2
        check(4, p1_4, p2_4, d1_4, d2_4)

        # Perm 5: o2->d2->o1->d1
        p2_5 = t2
        d2_5 = p2_5 + t_o2_d2
        p1_5 = np.maximum(d2_5 + t_d2_o1, t1)
        d1_5 = p1_5 + t_o1_d1
        check(5, p1_5, p2_5, d1_5, d2_5)

        # 4. Reduce & Add
        best_costs = np.min(costs, axis=0)
        best_perms = np.argmin(costs, axis=0)
        valid = np.isfinite(best_costs)

        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            logger.info(f"RR Matrix: Found {len(valid_idx)} feasible edges.")
            for k in valid_idx:
                r1 = req_list[idx_i[k]]
                r2 = req_list[idx_j[k]]
                route = self._build_route_from_perm(
                    best_perms[k], r1, r2, best_costs[k]
                )
                graph.add_rr_edge(r1.id, r2.id, route)

    def _solve_rv_matrix_empty(
        self,
        v_idxs: np.ndarray,
        r_idxs: np.ndarray,
        veh_list: list[VehicleAvailability],
        req_list: list[Request],
        delay_vals: np.ndarray,
        graph: RVGraph,
    ):
        """
        Builds edges for empty vehicles immediately without workers.
        """
        if len(v_idxs) == 0:
            return

        logger.info(f"RV Matrix (Empty): Processing {len(v_idxs)} edges.")

        # We need the Pickup->Dropoff time to complete the route info
        # We can fetch this in bulk
        r_origins = np.array([req_list[i].origin_node for i in r_idxs])
        r_dests = np.array([req_list[i].destination_node for i in r_idxs])

        mat = self._network.get_l1_cache()
        t_pd = mat[r_origins, r_dests]

        # Calculate positive delay (if arrived early, delay is 0)
        costs = np.maximum(delay_vals, 0)

        # Insert into graph
        # This loop is fast because it's only doing dict insertions
        for k, (v_i, r_i) in enumerate(zip(v_idxs, r_idxs)):
            v = veh_list[v_i]
            r = req_list[r_i]

            route = FeasibleRoute(
                route=Route(
                    stops=(
                        Stop(r.origin_node, True, r.id),
                        Stop(r.destination_node, False, r.id),
                    ),
                    total_travel_time=float(t_pd[k]),
                    total_travel_distance=0.0,
                    total_delay_cost=float(costs[k]),
                ),
                cost=float(costs[k]),
            )
            graph.add_rv_edge(r.id, v.vehicle_id, route)

    def _process_rv_occupied_parallel(
        self,
        v_idxs: np.ndarray,
        r_idxs: np.ndarray,
        veh_list: list[VehicleAvailability],
        req_list: list[Request],
        graph: RVGraph,
    ):
        if len(v_idxs) == 0:
            return

        # 1. Create Task List
        occupied_tasks = [
            RVTask(veh_list[v], req_list[r]) for v, r in zip(v_idxs, r_idxs)
        ]

        count = len(occupied_tasks)

        # --- PATH A: SEQUENTIAL (Low Volume) ---
        # 1000 is a safe crossover point where overhead usually breaks even
        if count < 1000:
            logger.info(f"RV Occupied: Processing {count} tasks sequentially.")

            # 1. Calculation Phase (Pure, Parallelizable logic if needed)
            # We explicitly check feasibility for all tasks first
            results = [
                (
                    task.req.id,
                    task.vehicle.vehicle_id,
                    self._route_planner.check_feasibility({task.req}, task.vehicle),
                )
                for task in occupied_tasks
            ]

            # 2. Mutation Phase (Side Effects)
            # Filter and insert
            for req_id, veh_id, result in results:
                if result.is_feasible:
                    graph.add_rv_edge(req_id, veh_id, cast(FeasibleRoute, result))
            return

        # --- PATH B: PARALLEL (High Volume) ---
        logger.info(f"RV Occupied: Processing {count} tasks in Parallel Pool.")
        l1_cache_copy = self._network.get_l1_cache()
        chunk_size = 400

        def rv_chunk_generator():
            for i in range(0, count, chunk_size):
                yield occupied_tasks[i : i + chunk_size]

        with ProcessPoolExecutor(
            max_workers=12, initializer=init_worker, initargs=(l1_cache_copy,)
        ) as executor:
            future_to_chunk = {
                executor.submit(process_rv_chunk, chunk): chunk
                for chunk in rv_chunk_generator()
            }
            for future in as_completed(future_to_chunk):
                try:
                    results = future.result()
                    for req_id, veh_id, feasible_route in results:
                        graph.add_rv_edge(req_id, veh_id, feasible_route)
                except Exception as exc:
                    logger.error(f"RV Worker exception: {exc}")
                    raise exc

    def _solve_rv_matrix_m1(
        self,
        v_idxs: np.ndarray,
        r_idxs: np.ndarray,
        veh_list: list[VehicleAvailability],
        req_list: list[Request],
        graph: RVGraph,
    ):
        """Vectorized solver for Vehicles with 1 existing passenger (3 Permutations)."""
        logger.info(f"RV Matrix (m=1): Processing {len(v_idxs)} edges.")
        
        # 1. GATHER DATA
        v_data = []
        for i in v_idxs:
            v = veh_list[i]
            pax = list(v.passengers_onboard)[0] # The only passenger (set -> list)
            limit = getattr(pax.original_request, "earliest_dropoff_time", None)
            d1_dead = (limit + DELTA) if limit else 1e9
            v_data.append((v.available_at_node, v.available_from_time, pax.original_request.destination_node, d1_dead))
            
        v_arr = np.array(v_data)
        v_loc  = v_arr[:, 0].astype(int)
        v_time = v_arr[:, 1]
        d1_loc = v_arr[:, 2].astype(int)
        d1_dead = v_arr[:, 3]

        # Pull request data
        r_origins = np.array([req_list[i].origin_node for i in r_idxs])
        r_dests   = np.array([req_list[i].destination_node for i in r_idxs])
        r_times   = np.array([req_list[i].request_time for i in r_idxs])
        
        # 2. FETCH TRAVEL TIMES
        mat = self._network.get_l1_cache()
        def get_v(u, v): return mat[u, v]
        
        # Edges
        t_v_p = get_v(v_loc, r_origins)
        t_v_d1 = get_v(v_loc, d1_loc)
        
        t_p_d = get_v(r_origins, r_dests)
        t_p_d1 = get_v(r_origins, d1_loc)
        
        t_d1_p = get_v(d1_loc, r_origins)
        t_d1_d = get_v(d1_loc, r_dests)
        t_d_d1 = get_v(r_dests, d1_loc)
        
        # 3. PERMUTATIONS (3 Valid)
        costs = np.full((3, len(v_idxs)), np.inf)

        # -- Perm 0: P -> D -> D1 --
        a_p = np.maximum(v_time + t_v_p, r_times)
        a_d = a_p + t_p_d
        a_d1 = a_d + t_d_d1
        mask0 = (a_p - r_times <= OMEGA) & (a_d1 <= d1_dead)
        costs[0, mask0] = (a_d1 - v_time)[mask0]

        # -- Perm 1: P -> D1 -> D --
        a_p = np.maximum(v_time + t_v_p, r_times)
        a_d1 = a_p + t_p_d1
        a_d = a_d1 + t_d1_d
        mask1 = (a_p - r_times <= OMEGA) & (a_d1 <= d1_dead)
        costs[1, mask1] = (a_d - v_time)[mask1]

        # -- Perm 2: D1 -> P -> D --
        a_d1 = v_time + t_v_d1
        a_p = np.maximum(a_d1 + t_d1_p, r_times)
        a_d = a_p + t_p_d
        mask2 = (a_d1 <= d1_dead) & (a_p - r_times <= OMEGA)
        costs[2, mask2] = (a_d - v_time)[mask2]

        # 4. REDUCE
        best_costs = np.min(costs, axis=0)
        best_perms = np.argmin(costs, axis=0)
        
        # 5. BUILD
        valid = np.isfinite(best_costs)
        valid_idx = np.where(valid)[0]
        
        for k in valid_idx:
            self._add_m1_edge(graph, int(v_idxs[k]), int(r_idxs[k]), veh_list, req_list, int(best_perms[k]), float(best_costs[k]))

    def _add_m1_edge(
        self,
        graph: RVGraph,
        v_idx: int,
        r_idx: int,
        v_list: list[VehicleAvailability],
        r_list: list[Request],
        perm: int,
        cost: float,
    ):
        v = v_list[v_idx]
        r = r_list[r_idx]
        # Vehicle has 1 passenger (set->list->[0])
        d1_req = list(v.passengers_onboard)[0].original_request
        
        s_p = Stop(r.origin_node, True, r.id)
        s_d = Stop(r.destination_node, False, r.id)
        s_d1 = Stop(d1_req.destination_node, False, d1_req.id)
        
        if perm == 0: stops = (s_p, s_d, s_d1)
        elif perm == 1: stops = (s_p, s_d1, s_d)
        else: stops = (s_d1, s_p, s_d)
        
        route = FeasibleRoute(
            route=Route(stops=stops, total_travel_time=0.0, total_travel_distance=0.0, total_delay_cost=float(cost)),
            cost=float(cost)
        )
        graph.add_rv_edge(r.id, v.vehicle_id, route)

    def _build_route_from_perm(
        self, perm_id: int, r1: Request, r2: Request, cost: float
    ) -> FeasibleRoute:
        """Helper to reconstruct the Stop sequence for the chosen best permutation."""
        # Sequence definitions matching the matrix logic above
        # 0: o1 -> o2 -> d1 -> d2
        # 1: o1 -> o2 -> d2 -> d1
        # 2: o2 -> o1 -> d1 -> d2
        # 3: o2 -> o1 -> d2 -> d1
        # 4: o1 -> d1 -> o2 -> d2
        # 5: o2 -> d2 -> o1 -> d1

        # Build Stops
        s_o1 = Stop(r1.origin_node, True, r1.id)
        s_d1 = Stop(r1.destination_node, False, r1.id)
        s_o2 = Stop(r2.origin_node, True, r2.id)
        s_d2 = Stop(r2.destination_node, False, r2.id)

        if perm_id == 0:
            seq = [s_o1, s_o2, s_d1, s_d2]
        elif perm_id == 1:
            seq = [s_o1, s_o2, s_d2, s_d1]
        elif perm_id == 2:
            seq = [s_o2, s_o1, s_d1, s_d2]
        elif perm_id == 3:
            seq = [s_o2, s_o1, s_d2, s_d1]
        elif perm_id == 4:
            seq = [s_o1, s_d1, s_o2, s_d2]
        elif perm_id == 5:
            seq = [s_o2, s_d2, s_o1, s_d1]
        else:
            raise ValueError(f"Invalid perm_id {perm_id}")

        # Calculate metrics
        total_time = 0.0
        total_dist = 0.0

        # Iterate through the sequence to sum up travel times and distances
        # Note: The first stop is the start, so we iterate from 0 to N-1
        current_node = seq[0].node
        for i in range(1, len(seq)):
            next_node = seq[i].node
            path = self._network.get_shortest_path(current_node, next_node)
            if path:
                total_time += path.travel_time
                total_dist += path.travel_distance
            current_node = next_node

        return FeasibleRoute(
            route=Route(
                stops=tuple(seq),
                total_travel_time=total_time,
                total_travel_distance=total_dist,
                total_delay_cost=cost,
            ),
            cost=cost,
        )
