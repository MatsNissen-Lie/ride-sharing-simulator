"""Fleet replay: move every vehicle along its route to the batch time.

The replay logic is shared; implementations differ only in how they read a
leg (the fastest path between two nodes) from the network:

- PathFleetReplay builds each leg as a path object
  (get_shortest_path_with_travel_time), as the controller always did.
- ArrayFleetReplay asks for a leg's totals and first edges only
  (path_time_and_distance, path_steps), which NetworkXService answers from
  per-graph arrays without building paths.

Both give the same numbers, so the same simulation; ArrayFleetReplay is faster.
"""

from abc import abstractmethod
from collections.abc import Iterator

from config.logging import get_logger
from src.interfaces import INetworkService, Passenger, Request, Vehicle, VehicleAvailability
from src.interfaces.data_models import Route
from src.interfaces.services import IFleetReplay
from src.interfaces.stats import ControllerStats

logger = get_logger(__name__)


class FleetReplay(IFleetReplay):
    """The replay; subclasses say how a leg is read from the network."""

    def __init__(self, network: INetworkService) -> None:
        self._network = network

    @abstractmethod
    def _leg(self, origin: int, destination: int) -> tuple[float, float] | None:
        """Travel time and distance of the fastest path, or None without one."""

    @abstractmethod
    def _steps(self, origin: int, destination: int) -> Iterator[tuple[int, float]]:
        """(node, travel time of the edge into it) along that path, lazily."""

    def replay(
        self,
        fleet: list[Vehicle],
        t_now: float,
        requests: dict[str, Request],
        stats: ControllerStats,
        last_processed_time: float,
    ) -> tuple[set[VehicleAvailability], dict[str, float]]:
        """
        For each Vehicle in the fleet, compute an immutable
        VehicleAvailability snapshot at batch time t_now:

        - If idle: available_at_node = last_visited_node, available_from_time = t_now.
        - If on a route:
            * Replay progress from route_assignment_time along the sequence of stops.
            * If a leg finishes before t_now, serve that stop (pickup/drop) and continue.
            * If we’re mid-leg at t_now, snap availability to the NEXT stop node and set
                available_from_time = (arrival time at that node) ≥ t_now.
            * Trim served stops; update passengers_onboard accordingly.

        Picked-up requests leave `requests`; deliveries after
        last_processed_time and vehicle-km are added to `stats`. Returns the
        snapshots and the promised pickup time of each request still on a route.
        """

        snapshots: set[VehicleAvailability] = set()
        # (request_id -> promised_pickup_time)
        last_promised: dict[str, float] = {}

        def get_path(u: int, v: int) -> tuple[float, float]:
            """Travel time and distance of the fastest path from u to v."""
            path = self._leg(u, v)
            if path is None:
                raise RuntimeError(f"No path found from {u} to {v}")
            return path

        for v in fleet:
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
                travel_time, travel_distance = get_path(cur_node, stop.node)
                arrival = cur_time + travel_time

                if arrival <= vehicle_t_now:
                    # Fully reach this stop before or at vehicle_t_now → serve it
                    self._count_distance(stats, travel_distance, loaded=bool(onboard))
                    cur_time = arrival
                    cur_node = stop.node

                    if stop.is_pickup:
                        req = requests.get(stop.request_id)
                        if req is not None:
                            onboard.add(
                                Passenger(
                                    original_request=req,
                                    pickup_time=cur_time,
                                )
                            )
                            # remove the request once it is picked up
                            del requests[stop.request_id]
                    else:
                        target_id = stop.request_id

                        # Find the passenger before removing
                        passenger = next(
                            (p for p in onboard if p.original_request.id == target_id),
                            None,
                        )

                        # Track stats if this is a new event
                        if passenger and cur_time > last_processed_time:
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

                            stats.delivered_count += 1
                            stats.sum_wait_time += wait_time
                            stats.sum_total_delay += total_delay
                            stats.revenue += req.fare

                        onboard = {
                            p for p in onboard if p.original_request.id != target_id
                        }
                        if target_id in requests:
                            raise RuntimeError(
                                f"Request {target_id} still in index after drop-off by vehicle {v.id}"
                            )

                    continue  # proceed to next stop

                # --- First stop we *haven't* reached by vehicle_t_now ---
                route_finished = False

                # 1) Snap physical location along this leg to a graph node
                anchor_node, anchor_time = self._snap_to_next_node_on_leg(
                    origin=cur_node,
                    destination=stop.node,
                    leg_start_time=cur_time,
                    t_now=vehicle_t_now,
                )

                if anchor_node != cur_node:
                    # Distance driven so far on this leg (a prefix of a shortest path).
                    self._count_distance(stats, 
                        get_path(cur_node, anchor_node)[1], loaded=bool(onboard)
                    )

                if anchor_time < vehicle_t_now:
                    raise ValueError(
                        f"Snap failed for {v.id}: anchor_time={anchor_time} < t_now={vehicle_t_now}. "
                        f"cur_time={cur_time}, path_tt={travel_time}, arrival={arrival}, "
                        f"leg={cur_node}->{stop.node}"
                    )

                # 2) Compute original scheduled arrival times for *all remaining* stops
                #    to generate promised pickup times.
                leg_prev_node = cur_node
                leg_prev_time = cur_time
                for stop2 in stops[idx:]:
                    arrival2 = leg_prev_time + get_path(leg_prev_node, stop2.node)[0]

                    if stop2.is_pickup:
                        req = requests.get(stop2.request_id)
                        if req is not None:
                            # If same id appears multiple times, last one wins (they should not).
                            last_promised[req.id] = arrival2

                    leg_prev_node = stop2.node
                    leg_prev_time = arrival2

                onboard_request_ids = {p.original_request.id for p in onboard}

                # Keep drop-offs of riders on board and rebalancing moves (the
                # vehicle keeps driving to its rebalancing destination). Pickups
                # are dropped: those riders are re-planned in the next assignment.
                drop_off_remaining = tuple(
                    stop2
                    for stop2 in stops[idx:]
                    if stop2.is_rebalancing
                    or (not stop2.is_pickup and stop2.request_id in onboard_request_ids)
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
        origin: int,
        destination: int,
        leg_start_time: float,
        t_now: float,
    ) -> tuple[int, float]:
        """
        Given a leg (the network's fastest path from origin to destination) and
        the time we started traversing it, find the next graph node after t_now
        and the time we reach it.

        Guarantees returned_time >= t_now (unless the leg has zero length,
        in which case we just return the final node at leg_start_time).

        If t_now is before the first edge completes, we snap to the
        first downstream node. If t_now is beyond the leg (should not
        happen if caller checked), we fall back to the last node.
        """
        # Degenerate leg (u == v): we are already at the final node
        if origin == destination:
            return destination, leg_start_time

        t = leg_start_time

        # The vehicle has not left the leg's first node yet: it stays there.
        if t >= t_now:
            return origin, t

        # Walk the edges only as far as needed; times add up as along the path.
        node = origin
        for node, dt in self._steps(origin, destination):
            t_next = t + dt

            if t_next >= t_now:
                # This is the first downstream node whose arrival is >= t_now
                return node, t_next

            # Otherwise we fully passed this edge before t_now; advance along the path
            t = t_next

        # If we get here, we've fully traversed the leg and t < t_now.
        # Caller should generally avoid this, but fall back safely:
        if t < t_now:
            raise ValueError(
                f"Snap reached end of leg but t={t} < t_now={t_now}. "
                f"leg_start={leg_start_time}, leg={origin}->{destination}"
            )
        return node, t

    @staticmethod
    def _count_distance(stats: ControllerStats, metres: float, loaded: bool) -> None:
        if loaded:
            stats.km_loaded += metres / 1000
        else:
            stats.km_empty += metres / 1000


class PathFleetReplay(FleetReplay):
    """Reads each leg as a path object (get_shortest_path_with_travel_time)."""

    def _leg(self, origin: int, destination: int) -> tuple[float, float] | None:
        path = self._network.get_shortest_path_with_travel_time(origin, destination)
        return None if path is None else (path.travel_time, path.travel_distance)

    def _steps(self, origin: int, destination: int) -> Iterator[tuple[int, float]]:
        path = self._network.get_shortest_path_with_travel_time(origin, destination)
        if path is not None:
            yield from zip(path.nodes[1:], path.travel_times)


class ArrayFleetReplay(FleetReplay):
    """Reads only a leg's totals and first edges (path_time_and_distance,
    path_steps); NetworkXService answers them from per-graph arrays."""

    def _leg(self, origin: int, destination: int) -> tuple[float, float] | None:
        return self._network.path_time_and_distance(origin, destination)

    def _steps(self, origin: int, destination: int) -> Iterator[tuple[int, float]]:
        return self._network.path_steps(origin, destination)
