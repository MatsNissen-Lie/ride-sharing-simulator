import numpy as np
from scipy.optimize import linear_sum_assignment

from config import get_logger
from src.interfaces import (
    INetworkService,
    IRebalancingSolver,
    Request,
    VehicleAvailability,
)
from src.interfaces.data_models import Route, Stop

logger = get_logger(__name__)


class ReactiveRebalancingSolver(IRebalancingSolver):
    """
    Implements Algorithm 4 (Rebalancing) using Scipy and Vectorized Matrix Lookups.

    Instead of an ILP, this solves the Linear Sum Assignment Problem (LSAP)
    to match idle vehicles to unassigned requests minimizing total travel time.
    """

    def __init__(self, network: INetworkService):
        self._network = network

    def solve(
        self,
        idle_vehicles: list[VehicleAvailability],
        unassigned_requests: list[Request],
        t_now: float,
    ) -> dict[str, Route]:

        n_vehicles = len(idle_vehicles)
        n_requests = len(unassigned_requests)

        if n_vehicles == 0 or n_requests == 0:
            return {}

        # 1. Vectorized Cost Lookup (The "Nuclear" approach)
        # Extract location IDs
        vehicle_locs = [v.available_at_node for v in idle_vehicles]
        request_locs = [r.origin_node for r in unassigned_requests]

        # Fetch the NxM cost matrix directly from the cache
        # This uses the fancy indexing from your NetworkXService
        cost_matrix = self._network.get_travel_time_matrix(vehicle_locs, request_locs)

        # 2. Pre-processing
        # Scipy linear_sum_assignment cannot handle np.inf or NaN well.
        # We replace them with a "Limit" value.
        # Since we usually have |V| > |R|, the solver will naturally avoid these
        # high-cost edges if valid ones exist.

        # Check for unreachability (inf) or missing data (nan)
        # Note: Your service returns inf for unreachability if nuclear is on.
        invalid_mask = np.isinf(cost_matrix) | np.isnan(cost_matrix)

        # Set invalid costs to a massive number (e.g., 1e9) so they are picked last
        # We assume typical travel times are < 100,000s
        cost_matrix[invalid_mask] = 1e9

        # 3. Solve (The "Quick Arg Min")
        # linear_sum_assignment solves min-cost matching for rectangular matrices.
        # It assigns exactly min(n_vehicles, n_requests) pairs.
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            logger.error(f"Scipy solver failed: {e}")
            return {}

        # 4. Construct Routes
        rebalancing_routes: dict[str, Route] = {}

        for r_idx, c_idx in zip(row_ind, col_ind):
            cost = cost_matrix[r_idx, c_idx]

            # If the best assignment is still "infinite" (unreachable), skip it.
            if cost >= 1e8:
                continue

            vehicle = idle_vehicles[r_idx]
            request = unassigned_requests[c_idx]

            # Create the rebalancing route (Move to request origin)
            # This is a deadhead trip.
            stop = Stop(
                node=request.origin_node,
                request_id=f"rebalance_{request.id}",
                is_pickup=False,  # Just a move
            )

            # We assume a direct trip, so delay/wait is 0
            route = Route(
                stops=(stop,),
                total_travel_time=float(cost),
                total_delay_cost=0.0,
                total_travel_distance=0.0,  # Could be filled with distance if needed
            )

            rebalancing_routes[vehicle.vehicle_id] = route

        return rebalancing_routes
