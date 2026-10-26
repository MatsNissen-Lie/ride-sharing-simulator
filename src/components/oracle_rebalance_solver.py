"""Perfect-information rebalancing: a benchmark for how much anticipation could gain.

Each batch, idle vehicles are matched (minimum total travel time) to the origins
of requests that will appear within the next `horizon` seconds, plus currently
unassigned requests. A pair is only allowed if the vehicle can reach the origin
before that request's pickup deadline. Vehicles still go through the normal
assignment, so this only changes where empty vehicles wait.

The oracle knows exactly where and when future requests appear, which no real
forecast does; treat its gain over reactive rebalancing as an optimistic bound
for anticipatory rebalancing, not as a method.
"""

import bisect

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import OMEGA, get_logger
from src.interfaces import INetworkService, Request, VehicleAvailability
from src.interfaces.data_models import Route, Stop

logger = get_logger(__name__)

UNREACHABLE = 1e9


class OracleRebalancingSolver:
    def __init__(
        self,
        network: INetworkService,
        request_times: np.ndarray,
        request_origins: np.ndarray,
        horizon: float,
        omega: float = OMEGA,
    ):
        self._network = network
        self._times = request_times
        self._origins = request_origins
        self._horizon = horizon
        self._omega = omega

    def solve(
        self,
        idle_vehicles: list[VehicleAvailability],
        unassigned_requests: list[Request],
        t_now: float | None = None,
    ) -> dict[str, Route]:
        if not idle_vehicles or t_now is None:
            return {}

        lo = bisect.bisect_right(self._times, t_now)
        hi = bisect.bisect_right(self._times, t_now + self._horizon)
        target_nodes = [r.origin_node for r in unassigned_requests]
        deadlines = [r.latest_pickup_time for r in unassigned_requests]
        target_nodes += self._origins[lo:hi].tolist()
        deadlines += (self._times[lo:hi] + self._omega).tolist()
        if not target_nodes:
            return {}

        travel = self._network.get_travel_time_matrix(
            [v.available_at_node for v in idle_vehicles], target_nodes
        )
        start = np.array([v.available_from_time for v in idle_vehicles])[:, None]
        cost = np.where(start + travel <= np.array(deadlines)[None, :], travel, UNREACHABLE)
        rows, cols = linear_sum_assignment(cost)

        routes: dict[str, Route] = {}
        for i, j in zip(rows, cols):
            if cost[i, j] >= UNREACHABLE or travel[i, j] == 0:
                continue  # unreachable in time, or already there
            vehicle = idle_vehicles[i]
            stop = Stop(node=target_nodes[j], is_pickup=False, request_id=f"rebalance_{j}")
            routes[vehicle.vehicle_id] = Route((stop,), float(travel[i, j]), 0.0, 0.0)
        return routes
