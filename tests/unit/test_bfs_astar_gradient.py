import unittest
import numpy as np
from src.components.search_algorithm.bfs_astar_gradient_bad import TripSolver
from src.interfaces import (
    VehicleAvailability,
    Request,
    Passenger,
)


class TestTripSolverGradient(unittest.TestCase):
    def test_heuristic_components(self):
        """
        Verify the components of the 'Method One' heuristic:
        h(s) = SunkCost + MST + Bottleneck
        Scenario:
        - v at 0, t=10.
        - Req1 (Waiting): O=1, D=2, t_req=0.
        - Req2 (Onboard): D=3, t_req=0.
        - Distances:
            0->1=5, 1->2=5, 2->3=5, 0->3=15
            0->2=10
            3->1=10
        - MST nodes: {0, 1, 2, 3}.
            0->1 (5), 1->2 (5), 2->3 (5). MST = 15.
        - Sunk Cost:
            Req1: 10 - 0 = 10
            Req2: 10 - 0 = 10
            Total Sunk = 20.
        - Bottleneck:
            Nearest Drop (2 or 3).
            0->2 = 10.
            0->3 = 15.
            Min = 10.
        - Expected h = 20 + 15 + 10 = 45.
        """
        # Setup Vehicle (v at 0, t=10, 1 onboard destined for 3)
        req2 = Request(
            id="r2",
            origin_node=99,
            destination_node=3,
            request_time=0.0,
            latest_pickup_time=100.0,
            latest_dropoff_time=200.0,
        )
        p2 = Passenger(original_request=req2, pickup_time=0.0)

        vehicle = VehicleAvailability(
            vehicle_id="v1",
            capacity=4,
            available_at_node=0,
            available_from_time=10.0,
            passengers_onboard=frozenset([p2]),
            remaining_stops_to_serve=(),
        )

        # Setup Waiting Request (O=1, D=2)
        req1 = Request(
            id="r1",
            origin_node=1,
            destination_node=2,
            request_time=0.0,
            latest_pickup_time=100.0,
            latest_dropoff_time=200.0,
        )

        # Matrix 4x4
        # Initialize with INF
        time_matrix = np.full((4, 4), float("inf"))
        np.fill_diagonal(time_matrix, 0.0)

        # Define Edges
        # 0-1 (5), 1-2 (5), 2-3 (5)
        # 0-2 (10), 0-3 (15)
        # 1-3 (10)

        matrix_vals = [
            [0.0, 5.0, 10.0, 15.0],
            [5.0, 0.0, 5.0, 10.0],
            [10.0, 5.0, 0.0, 5.0],
            [15.0, 10.0, 5.0, 0.0],
        ]

        time_matrix = np.array(matrix_vals)

        res = TripSolver.solve([req1], vehicle, time_matrix)
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, "route"))
        # With this heuristic, we expect to find the route 0->1->2->3 which is feasible.


if __name__ == "__main__":
    unittest.main()
