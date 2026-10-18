"""Batch controller implementation stub with meta request conversion."""

from collections.abc import Iterable, Iterator
from typing import Set, Tuple
import bisect

from config.config import DELTA, OMEGA, BATCH_DURATION
from config.logging import get_logger
from src.interfaces.batch_control import IBatchController
from src.interfaces import (
    VehicleAvailability,
    AssignmentResult,
    INetworkService,
    MetaRequest,
    Passenger,
    Request,
    Vehicle,
)
from src.interfaces.data_models import NetworkPathExtended, Route
from src.interfaces.stats import ControllerStats

logger = get_logger(__name__)


class BatchController(IBatchController):
    """Minimal batch controller focusing on MetaRequest conversion."""

    def __init__(
        self,
        network: INetworkService,
        vehicles: Iterable[Vehicle],
        omega: float = OMEGA,
        delta: float = DELTA,
    ) -> None:
        super().__init__(network, vehicles)
        self._id_to_vehicle: dict[str, Vehicle] = {v.id: v for v in self._fleet}
        # Optimized history: separate lists for bisect
        self._batch_end_times: list[float] = []

        # Constraint Parameters
        self._omega = omega
        self._delta = delta

        # Stats Aggregators
        self._current_stats = ControllerStats()
        self._last_processed_time: float = 0.0

    def project_fleet_and_update_requests(
        self, t_now: float
    ) -> Set[VehicleAvailability]:
        snapshots, last_promised = self._project_fleet(t_now)

        promised_requests: Set[Request] = set()
        for req_id, promised_time in last_promised.items():
            base_request = self._request_index.get(req_id)
            if base_request is not None:
                req = self._create_promised_request(
                    request=base_request,
                    promised_pickup_time=promised_time,
                )
                promised_requests.add(req)
                # update the request index to point to the promised version
                self._request_index[req_id] = req
            else:
                raise RuntimeError(f"Request ID {req_id} not found in request index.")

        return snapshots  # , promised_requests

    def _project_fleet(
        self, t_now: float
    ) -> Tuple[Set[VehicleAvailability], dict[str, float]]:
        """
        For each Vehicle in the controller-owned fleet, compute an immutable
        VehicleAvailability snapshot at batch time t_now:

        - If idle: available_at_node = last_visited_node, available_from_time = t_now.
        - If on a route:
            * Replay progress from route_assignment_time along the sequence of stops.
            * If a leg finishes before t_now, serve that stop (pickup/drop) and continue.
            * If we’re mid-leg at t_now, snap availability to the NEXT stop node and set
                available_from_time = (arrival time at that node) ≥ t_now.
            * Trim served stops; update passengers_onboard accordingly.

        Requires:
        - self._fleet: Iterable[Vehicle]
        - self._network: INetworkService
        - self._request_index: Dict[str, Request] (to materialize Passenger on pickups)
        """

        snapshots: Set[VehicleAvailability] = set()
        # (request_id -> promised_pickup_time)
        last_promised: dict[str, float] = {}

        def get_path(u: int, v: int) -> NetworkPathExtended:
            path = self._network.get_shortest_path_with_travel_time(u, v)
            if path is None:
                raise RuntimeError(f"No path found from {u} to {v}")
            return path

        for v in self._fleet:
            # --- Case 1: idle vehicle ---
            if v.is_idle:
                # An idle vehicle may still be arriving at its node (last_visited_time > t_now).
                v.last_visited_time = max(v.last_visited_time, t_now)
                snapshots.add(
                    VehicleAvailability(
                        vehicle_id=v.id,
                        capacity=v.capacity,
                        available_at_node=v.last_visited_node,
                        available_from_time=v.last_visited_time,
                        passengers_onboard=frozenset(),
                        remaining_stops_to_serve=(),
                    )
                )
                continue

            # --- Case 2: no usable route metadata → treat as idle-ish ---
            if v.current_route is None or v.route_assignment_time is None:
                has_route = v.current_route is not None
                has_assignment = v.route_assignment_time is not None
                logger.warning(
                    f"Vehicle {v.id} has incomplete route metadata "
                    f"(has_route={has_route}, has_assignment={has_assignment}); "
                    f"treating as idle at last_visited_node. "
                    f"Passengers: {len(v.passengers_onboard)} {[p.original_request.id for p in v.passengers_onboard]} "
                    f"Route Stops: {len(v.current_route.stops) if v.current_route else 'None'}"
                )
                raise RuntimeError(
                    f"Vehicle {v.id} has no route metadata but is not idle."
                )

            # --- Case 3: replay the assigned route from assignment time ---
            cur_node = v.last_visited_node
            cur_time = v.route_assignment_time
            onboard = v.passengers_onboard  # mutate in place
            stops = list(v.current_route.stops)

            # Use a local time for this vehicle to avoid polluting t_now for others
            vehicle_t_now = t_now

            # If for some reason t_now is before assignment, clamp
            if vehicle_t_now < cur_time:
                vehicle_t_now = cur_time

            route_finished = True

            for idx, stop in enumerate(stops):
                path = get_path(cur_node, stop.node)
                arrival = cur_time + path.travel_time

                if arrival <= vehicle_t_now:
                    # Fully reach this stop before or at vehicle_t_now → serve it
                    cur_time = arrival
                    cur_node = stop.node

                    if stop.is_pickup:
                        req = self._request_index.get(stop.request_id)
                        if req is not None:
                            onboard.add(
                                Passenger(
                                    original_request=req,
                                    pickup_time=cur_time,
                                )
                            )
                            # remove the request once it is picked up
                            del self._request_index[stop.request_id]
                    else:
                        target_id = stop.request_id

                        # Find the passenger before removing
                        passenger = next(
                            (p for p in onboard if p.original_request.id == target_id),
                            None,
                        )

                        # Track stats if this is a new event
                        if passenger and cur_time > self._last_processed_time:
                            req = passenger.original_request

                            # Validating constraint
                            if cur_time > req.latest_dropoff_time + 1e-6:  # float tolerance
                                raise RuntimeError(
                                    f"HARD CONSTRAINT VIOLATION: Request {req.id} dropped off at {cur_time:.2f} "
                                    f"but latest_dropoff_time is {req.latest_dropoff_time:.2f}. "
                                    f"Delay: {cur_time - req.latest_dropoff_time:.2f}s"
                                )

                            wait_time = passenger.pickup_time - req.request_time

                            direct_travel_time = self._network.get_travel_time(
                                req.origin_node, req.destination_node
                            )
                            # Fallback if path not found (though it should be)
                            if direct_travel_time == float("inf"):
                                logger.warning(
                                    f"Could not calc direct travel time for req {req.id}"
                                )
                                direct_travel_time = (
                                    0.0  # Should not happen in connected graph
                                )

                            total_delay = (
                                cur_time - req.request_time - direct_travel_time
                            )

                            self._current_stats.delivered_count += 1
                            self._current_stats.sum_wait_time += wait_time
                            self._current_stats.sum_total_delay += total_delay

                        onboard = {
                            p for p in onboard if p.original_request.id != target_id
                        }
                        if target_id in self._request_index:
                            raise RuntimeError(
                                f"Request {target_id} still in index after drop-off by vehicle {v.id}"
                            )

                    continue  # proceed to next stop

                # --- First stop we *haven't* reached by vehicle_t_now ---
                route_finished = False

                # 1) Snap physical location along this leg to a graph node
                anchor_node, anchor_time = self._snap_to_next_node_on_leg(
                    path=path,
                    leg_start_time=cur_time,
                    t_now=vehicle_t_now,
                )

                if anchor_time < vehicle_t_now:
                    path_sum = sum(path.travel_times) if path.travel_times else 0
                    raise ValueError(
                        f"Snap failed for {v.id}: anchor_time={anchor_time} < t_now={vehicle_t_now}. "
                        f"cur_time={cur_time}, path_tt={path.travel_time}, path_sum={path_sum}, "
                        f"arrival={arrival}, path_nodes={path.nodes}, path_times={path.travel_times}"
                    )

                # 2) Compute original scheduled arrival times for *all remaining* stops
                #    to generate promised pickup times.
                leg_prev_node = cur_node
                leg_prev_time = cur_time
                for stop2 in stops[idx:]:
                    path2 = get_path(leg_prev_node, stop2.node)
                    arrival2 = leg_prev_time + path2.travel_time

                    if stop2.is_pickup:
                        req = self._request_index.get(stop2.request_id)
                        if req is not None:
                            # If same id appears multiple times, last one wins (they should not).
                            last_promised[req.id] = arrival2

                    leg_prev_node = stop2.node
                    leg_prev_time = arrival2

                onboard_request_ids = {p.original_request.id for p in onboard}

                drop_off_remaining = tuple(
                    [
                        drop_off
                        for drop_off in stops[idx:]
                        if not drop_off.is_pickup
                        and drop_off.request_id in onboard_request_ids
                    ]
                )

                # 3) Mutate vehicle state to reflect projection
                v.last_visited_node = anchor_node
                v.last_visited_time = anchor_time
                v.passengers_onboard = onboard
                v.current_route = Route(
                    (drop_off_remaining),
                    0,
                    0,
                    0,  # the route metrics are only used by ILP solver. This will get passed thourgh vehicle_availability
                )  # This route is only to ensure valid projection of vehicle state.
                v.route_assignment_time = anchor_time
                # If we sill have stops to serve, we keep the route as is. It might be needed next batch.
                # v.current_route = None  # TODO: SHOULD THIS ROUTE BE TRIMMED? What if the planner does not assign a new route? we still have to drop off people!

                # 4) Vehicle availability: it's anchored at this node,
                #    and can be used for new routes no earlier than anchor_time (>= vehicle_t_now).
                if anchor_time < t_now:
                    raise ValueError(
                        f"Vehicle {v.id} projected available_from_time {anchor_time} is before batch time {t_now}"
                    )

                snapshots.add(
                    VehicleAvailability(
                        vehicle_id=v.id,
                        capacity=v.capacity,
                        available_at_node=anchor_node,
                        available_from_time=anchor_time,
                        passengers_onboard=frozenset(onboard),
                        remaining_stops_to_serve=drop_off_remaining,  # TODO: remaining dropoffs should be here
                    )
                )
                break  # done with this vehicle

            if route_finished:
                # We have served all stops by or before vehicle_t_now → route is done
                v.last_visited_node = cur_node
                available_from = max(vehicle_t_now, cur_time)
                # The vehicle waits at its last stop until the batch time.
                v.last_visited_time = available_from
                v.passengers_onboard = onboard
                v.current_route = None
                v.route_assignment_time = None

                if available_from < t_now:
                    raise ValueError(
                        f"Vehicle {v.id} projected available_from_time {available_from} is before batch time {t_now}"
                    )

                snapshots.add(
                    VehicleAvailability(
                        vehicle_id=v.id,
                        capacity=v.capacity,
                        available_at_node=cur_node,
                        # It is effectively idle at vehicle_t_now
                        available_from_time=available_from,
                        passengers_onboard=frozenset(),  # frozenset(onboard),
                        remaining_stops_to_serve=(),
                    )
                )

        for snapshot in snapshots:
            if snapshot.available_from_time < t_now:
                raise ValueError(
                    f"Vehicle {snapshot.vehicle_id} projected availability {snapshot.available_from_time} is before batch time {t_now}"
                )

        return snapshots, last_promised

    def _snap_to_next_node_on_leg(
        self,
        path: NetworkPathExtended,
        leg_start_time: float,
        t_now: float,
    ) -> tuple[int, float]:
        """
        Given a NetworkPath and the time we started traversing it,
        find the next graph node after t_now and the time we reach it.

        Guarantees returned_time >= t_now (unless the leg has zero length,
        in which case we just return the final node at leg_start_time).

        If t_now is before the first edge completes, we snap to the
        first downstream node. If t_now is beyond the leg (should not
        happen if caller checked), we fall back to the last node.
        """

        # Degenerate leg: no movement
        if not path.nodes:
            raise RuntimeError("NetworkPath has no nodes")

        if not path.travel_times or len(path.nodes) == 1:
            # Zero-length leg (u == v): we are already at the final node
            return path.nodes[-1], leg_start_time

        t = leg_start_time

        # The vehicle has not left the leg's first node yet: it stays there.
        if t >= t_now:
            return path.nodes[0], t

        for i, dt in enumerate(path.travel_times):
            # travel_times is length len(nodes) - 1 by contract
            # so i+1 is always a valid index here
            next_node = path.nodes[i + 1]
            t_next = t + dt

            if t_next >= t_now:
                # This is the first downstream node whose arrival is >= t_now
                return next_node, t_next

            # Otherwise we fully passed this edge before t_now; advance along the path
            t = t_next

        # If we get here, we've fully traversed the leg and t < t_now.
        # Caller should generally avoid this, but fall back safely:
        if t < t_now:
            raise ValueError(
                f"Snap reached end of leg but t={t} < t_now={t_now}. "
                f"leg_start={leg_start_time}, path_times={path.travel_times}"
            )
        return path.nodes[-1], t

    def convert_meta_requests(self, new_meta: Iterable[MetaRequest]) -> Set[Request]:
        converted: Set[Request] = set()
        for meta in new_meta:
            travel_time = getattr(
                self._network.get_shortest_path(
                    meta.origin_node, meta.destination_node
                ),
                "travel_time",
                None,
            )
            if travel_time is None:
                raise RuntimeError(
                    f"No path found from {meta.origin_node} to {meta.destination_node}"
                )
            converted.add(
                Request(
                    id=meta.id,
                    origin_node=meta.origin_node,
                    destination_node=meta.destination_node,
                    request_time=meta.request_time,
                    latest_pickup_time=meta.request_time + self._omega,
                    latest_dropoff_time=meta.request_time + travel_time + self._delta,
                )
            )
        return converted

    def iter_batches(
        self,
        meta_stream: Iterable[MetaRequest],
        batch_window: float,
        min_start_time: float = 0.0,
    ) -> Iterator[Tuple[float, Set[Request], Set[VehicleAvailability]]]:
        """
        Stream MetaRequests into fixed-size time windows without materializing them.

        Assumes:
          - meta_stream yields MetaRequest in non-decreasing request_time order.

        For each window [t_start, t_start + batch_window):

          1. Collect all MetaRequests with request_time < t_window_end.
          2. Convert them to Requests.
          3. Add them into _request_index.
          4. Project the fleet to t_window_end (project_fleet_and_update_requests).
          5. Yield (t_now, batch_requests, vehicle_availabilities),

        where:
          - t_now = t_window_end
          - batch_requests = new_requests | promised_requests
        """

        meta_iter = iter(meta_stream)

        try:
            current = next(meta_iter)
        except StopIteration:
            return

        t_start = current.request_time

        # Fast-forward if needed
        if t_start < min_start_time:
            while current is not None and current.request_time < min_start_time:
                try:
                    current = next(meta_iter)
                except StopIteration:
                    current = None
                    return

            # Now current.request_time >= min_start_time.
            # We set t_start to min_start_time (or current.request_time?)
            # To maintain batch alignment, we might want to start exactly at min_start_time.
            t_start = min_start_time

        while True:
            t_window_end = t_start + batch_window
            window_meta: list[MetaRequest] = []

            while current is not None and current.request_time < t_window_end:
                window_meta.append(current)
                try:
                    current = next(meta_iter)
                except StopIteration:
                    current = None
                    break

            batch_requests, snapshots = self.prepare_batch(
                t_now=t_window_end, new_meta=window_meta
            )

            # validate that all vehicle availabilities are >= t_window_end
            # and all request times are <= t_window_end
            self._validate_batch_consistency(
                t_now=t_window_end,
                snapshots=snapshots,
                requests=batch_requests,
            )

            # Record batch history
            self._batch_end_times.append(t_window_end)

            # Prune history: remove batches older than OMEGA (plus a buffer)
            cutoff_time = t_window_end - OMEGA - BATCH_DURATION
            while self._batch_end_times and self._batch_end_times[0] < cutoff_time:
                self._batch_end_times.pop(0)

            if batch_requests or snapshots:
                yield t_window_end, batch_requests, snapshots

            # Mark this time as processed to avoid duplicate stats in future replays
            self._last_processed_time = t_window_end

            if (
                current is None
            ):  #  and not self._request_index and all(v.is_idle for v in self._fleet):
                break

            t_start = t_window_end

    def prepare_batch(
        self, t_now: float, new_meta: Iterable[MetaRequest]
    ) -> Tuple[Set[Request], Set[VehicleAvailability]]:

        # 1) update vehicles and remove served requests
        snapshots = self.project_fleet_and_update_requests(t_now)

        # 2) Expire zombie requests
        self._expire_stale_requests(t_now)

        # 3) Convert new meta requests and add to index
        new_requests = self.convert_meta_requests(new_meta)
        self._current_stats.new_requests_count += len(new_requests)
        for req in new_requests:
            self._request_index[req.id] = req

        batch_requests = set(self._request_index.values())
        return batch_requests, snapshots

    def commit_assignment(self, t_now: float, result: AssignmentResult) -> None:
        """
        Apply the ILP solution to the controller-owned fleet.

        For each Assignment in result.assignments:
          - Find the corresponding Vehicle in self._fleet.
          - Attach the new Route as current_route, with
                * route_assignment_time  = vehicle.last_visited_time

        Vehicles that are NOT mentioned in result.assignments keep whatever
        route/state _project_fleet() left them with (e.g. a dropoff-only
        fallback route for existing passengers).

        Requests are NOT removed here; they are removed when drop-offs are
        actually served in _project_fleet, and stale ones are expired by
        _expire_stale_requests.
        """
        # Build a quick lookup from vehicle_id -> Vehicle
        # id_to_vehicle: dict[str, Vehicle] = {v.id: v for v in self._fleet}

        for assignment in result.assignments:
            vehicle = self._id_to_vehicle.get(assignment.vehicle_id)
            if vehicle is None:
                raise RuntimeError(
                    f"Assignment references unknown vehicle_id={assignment.vehicle_id!r}"
                )

            # 1) The vehicle's physical state is already snapped to the availability snapshot
            #    by _project_fleet (called before optimization).
            #    So vehicle.last_visited_time IS the available_from_time.

            # 2) Attach the new route starting from that availability time.
            #    The planner routed from available_from_time (>= t_now); starting
            #    the replay at t_now would let the vehicle skip the rest of its leg.
            vehicle.current_route = assignment.route
            vehicle.route_assignment_time = vehicle.last_visited_time

        # NOTE:
        # - result.unassigned_requests are left in self._request_index.
        #   They will either be matched in a future batch or expired by
        #   _expire_stale_requests(t_now).
        # - Vehicles not mentioned in result.assignments keep whatever
        #   current_route / route_assignment_time projection gave them.

    def get_rebalancing_candidates(
        self,
        assignment_result: AssignmentResult,
        vehicle_avail_pool: Set[VehicleAvailability],
    ) -> Tuple[list[VehicleAvailability], list[Request]]:
        """
        Identifies the inputs for the Rebalancing LP (Algorithm 4).

        According to the paper, rebalancing targets:
        1. V_idle: Vehicles that are "empty, stopped and unassigned".
        2. R_ko: The set of "unassigned requests".

        Args:
            assignment_result: The result from the main assignment solver.
            snapshots: The vehicle states used during the assignment.

        Returns:
            A tuple containing (idle_vehicles, unassigned_requests).
        """
        # 1. Identify Unassigned Requests (R_ko)
        # Directly from the assignment result[cite: 128].
        unassigned_requests = list(assignment_result.unassigned_requests)

        # 2. Identify Idle Vehicles (V_idle)
        # We need vehicles that:
        #   a) Are physically empty and have no remaining stops (stopped/idle state).
        #   b) Were NOT assigned a new trip in the main optimization batch.

        assigned_vehicle_ids = {a.vehicle_id for a in assignment_result.assignments}

        idle_vehicles = []
        for v in vehicle_avail_pool:
            # Check a: Empty and Stopped
            is_physically_idle = (
                len(v.passengers_onboard) == 0 and len(v.remaining_stops_to_serve) == 0
            )

            # Check b: Unassigned in this batch
            is_unassigned = v.vehicle_id not in assigned_vehicle_ids

            if is_physically_idle and is_unassigned:
                idle_vehicles.append(v)

        return idle_vehicles, unassigned_requests

    def apply_rebalancing(
        self, rebalancing_routes: dict[str, Route], t_now: float
    ) -> None:
        """
        Applies rebalancing routes (Algorithm 4) to idle vehicles.

        These routes move empty vehicles to high-demand areas (or specific request origins)
        but do NOT technically 'pickup' the request yet.
        """
        for v_id, route in rebalancing_routes.items():
            vehicle = self._id_to_vehicle.get(v_id)

            # Safety check: ensure vehicle exists and we aren't overwriting a busy vehicle
            # (Though get_rebalancing_candidates filters for idle ones, double-check safe state)
            if vehicle and vehicle.is_idle:
                vehicle.current_route = route
                vehicle.route_assignment_time = t_now
                # Note: We do NOT remove requests from _request_index.
                # Rebalancing targets are "ghosts"; the actual requests remain in R_ko[cite: 194].

    #################################################################
    # Helper Methods
    #################################################################
    def _create_promised_request(
        self, request: Request, promised_pickup_time: float
    ) -> Request:
        """Placeholder for deriving promised requests from base requests."""
        return Request(
            id=request.id,
            origin_node=request.origin_node,
            destination_node=request.destination_node,
            request_time=request.request_time,
            latest_pickup_time=promised_pickup_time,  # cant go back on a promise
            latest_dropoff_time=request.latest_dropoff_time,
            promised_pickup_time=promised_pickup_time,  # we need this info later
        )

    def _expire_stale_requests(self, t_now: float) -> None:
        """Remove all requests that have waited longer than omega."""
        to_remove: list[str] = []
        for req_id, req in self._request_index.items():
            if t_now - req.request_time > self._omega:
                to_remove.append(req_id)
                # Capture stats
                self._current_stats.expired_count += 1
                self._current_stats.sum_expired_wait += t_now - req.request_time

        for req_id in to_remove:
            del self._request_index[req_id]

    def get_batch_index(self, t: float) -> int:
        """
        Returns the batch index for a given time t using bisect for O(log N) lookup.
        """
        idx = bisect.bisect_right(self._batch_end_times, t)

        if idx < len(self._batch_end_times):
            return idx

        # If t is beyond our history (future), it belongs to the next batch
        raise ValueError(f"Time {t} is beyond known batch end times.")

    def get_requests_by_batch(self) -> dict[float, list[Request]]:
        """
        Groups all currently active requests by their origin batch index.
        """
        grouped: dict[float, list[Request]] = {}
        for req in self._request_index.values():
            batch_time = self._batch_end_times[self.get_batch_index(req.request_time)]
            grouped.setdefault(batch_time, []).append(req)
        return grouped

    def _validate_batch_consistency(
        self, t_now: float, snapshots: Set[VehicleAvailability], requests: Set[Request]
    ) -> None:
        """Validate that all vehicle availabilities are >= t_now
        and all request times are <= t_now."""
        for snapshot in snapshots:
            if snapshot.available_from_time < t_now:
                raise ValueError(
                    f"Vehicle {snapshot.vehicle_id} projected availability {snapshot.available_from_time} is before batch time {t_now}"
                )
        for req in requests:
            if req.request_time > t_now:
                raise ValueError(
                    f"Request {req.id} has request_time {req.request_time} after batch time {t_now}"
                )

    def get_and_reset_metrics(self) -> ControllerStats:
        """Returns the accumulated delivered stats and clears the buffer."""
        stats = self._current_stats
        self._current_stats = ControllerStats()
        return stats


__all__ = ["BatchController"]
