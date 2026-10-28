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
    IFareModel,
    IFleetReplay,
    INetworkService,
    MetaRequest,
    Passenger,
    Request,
    Vehicle,
)
from src.interfaces.data_models import Route
from src.interfaces.stats import ControllerStats
from src.utils.timing import timed

logger = get_logger(__name__)

PROMISE_TOLERANCE = 1e-3  # seconds


class BatchController(IBatchController):
    """Minimal batch controller focusing on MetaRequest conversion."""

    def __init__(
        self,
        network: INetworkService,
        vehicles: Iterable[Vehicle],
        replay: IFleetReplay,
        omega: float = OMEGA,
        delta: float = DELTA,
        fare_model: IFareModel | None = None,
    ) -> None:
        super().__init__(network, vehicles)
        self._fare_model = fare_model
        self._replay = replay  # moves the fleet to each batch time
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
        """Move the fleet to t_now (the replay component; see IFleetReplay)."""
        return self._replay.replay(
            self._fleet, t_now, self._request_index, self._current_stats, self._last_processed_time
        )

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
                    fare=(
                        self._fare_model.price(meta.origin_node, meta.destination_node)
                        if self._fare_model
                        else 0.0
                    ),
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

        # Fast-forward if needed
        if current.request_time < min_start_time:
            while current is not None and current.request_time < min_start_time:
                try:
                    current = next(meta_iter)
                except StopIteration:
                    current = None
                    return

        # Batches start at min_start_time when one is given (the stream may
        # already begin there), else at the first request.
        t_start = min_start_time if min_start_time > 0 else current.request_time

        while True:
            t_window_end = t_start + batch_window
            window_meta: list[MetaRequest] = []

            with timed("read_requests"):
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
            with timed("validate"):
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
        with timed("project_fleet"):
            snapshots = self.project_fleet_and_update_requests(t_now)

        # 2) Expire zombie requests
        with timed("expire"):
            self._expire_stale_requests(t_now)

        # 3) Convert new meta requests and add to index
        with timed("new_requests"):
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
    ) -> Tuple[list[VehicleAvailability], list[Request], list[VehicleAvailability]]:
        """
        Identifies the inputs for the Rebalancing LP (Algorithm 4).

        Also returns the vehicles still driving to a rebalancing destination
        and not assigned a rider in this batch (en route), so a rebalancer can
        count them where they are heading.

        According to the paper, rebalancing targets:
        1. V_idle: Vehicles that are "empty, stopped and unassigned".
        2. R_ko: The set of "unassigned requests".

        Args:
            assignment_result: The result from the main assignment solver.
            snapshots: The vehicle states used during the assignment.

        Returns:
            (idle_vehicles, unassigned_requests, en_route_vehicles).
        """
        # 1. Identify Unassigned Requests (R_ko)
        # Directly from the assignment result[cite: 128].
        unassigned_requests = list(assignment_result.unassigned_requests)

        # 2. Identify Idle Vehicles (V_idle)
        # We need vehicles that:
        #   a) Are physically empty and have no remaining stops (stopped/idle state).
        #   b) Were NOT assigned a new trip in the main optimization batch.

        assigned_vehicle_ids = {a.vehicle_id for a in assignment_result.assignments}
        assigned_vehicle_ids |= assignment_result.rebalancing.keys()

        idle_vehicles = []
        en_route = []
        for v in vehicle_avail_pool:
            if (
                v.vehicle_id not in assigned_vehicle_ids
                and not v.passengers_onboard
                and v.remaining_stops_to_serve
                and all(s.is_rebalancing for s in v.remaining_stops_to_serve)
            ):
                en_route.append(v)
                continue
            # Check a: Empty and Stopped
            is_physically_idle = (
                len(v.passengers_onboard) == 0 and len(v.remaining_stops_to_serve) == 0
            )

            # Check b: Unassigned in this batch
            is_unassigned = v.vehicle_id not in assigned_vehicle_ids

            if is_physically_idle and is_unassigned:
                idle_vehicles.append(v)

        return idle_vehicles, unassigned_requests, en_route

    def apply_rebalancing(
        self, rebalancing_routes: dict[str, Route], t_now: float
    ) -> None:
        """
        Applies rebalancing routes (Algorithm 4) to idle vehicles.

        These routes move empty vehicles to high-demand areas (or specific request origins)
        but do NOT technically 'pickup' the request yet. A vehicle already
        rebalancing may be redirected; an empty route stops it at its next node.
        """
        for v_id, route in rebalancing_routes.items():
            vehicle = self._id_to_vehicle.get(v_id)

            # Never overwrite a vehicle that carries or is fetching riders.
            if vehicle and vehicle.current_occupancy == 0 and (
                vehicle.current_route is None
                or all(s.is_rebalancing for s in vehicle.current_route.stops)
            ):
                vehicle.current_route = route if route.stops else None
                vehicle.route_assignment_time = vehicle.last_visited_time
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
            # Can't go back on a promise. The slack absorbs float noise: re-planning
            # recomputes the same arrival time, which may come out a hair later.
            latest_pickup_time=promised_pickup_time + PROMISE_TOLERANCE,
            latest_dropoff_time=request.latest_dropoff_time,
            promised_pickup_time=promised_pickup_time,  # we need this info later
            fare=request.fare,
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
