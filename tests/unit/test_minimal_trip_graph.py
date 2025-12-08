
import unittest
from unittest.mock import MagicMock
from typing import Set, List, Tuple
from src.components.graph_builder.rtv_builder_trip_centric import TripGraph, Stop, StopType, INF
from src.interfaces.services import INetworkService
from src.interfaces.data_models import VehicleAvailability, Request, Passenger
import numpy as np

class TestMinimalTripGraph(unittest.TestCase):

    def setUp(self):
        self.mock_network = MagicMock(spec=INetworkService)
        # Mock travel time: 10.0 between any two different nodes
        def get_travel_time_matrix(locs1, locs2):
            n = len(locs1)
            adj = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if locs1[i] != locs2[j]:
                        adj[i, j] = 10.0
            return adj
        self.mock_network.get_travel_time_matrix.side_effect = get_travel_time_matrix
        
    def test_solve_capacity_soft_pruning(self):
        """
        Test that B&B correctly falls through cache on capacity violation.
        Scenario:
          - Vehicle Load: 3/4.
          - Request: A(P), B(P), X(D-veh).
          - Opitmal Suffix (Geom): A -> B -> X (Cost: 10+5+10 = 25).
            - Load: 3->4->5 (FAIL).
          - Suboptimal: A -> X -> B (Cost: 10+10+10 = 30).
            - Load: 3->4->3->4 (PASS).
        """
        # Manually constructing TripGraph to control matrix
        stops = [
            Stop("V_Start", 0, 0, 100, StopType.VEHICLE_START),
            Stop("A", 1, 0, 100, StopType.PICKUP),
            Stop("B", 2, 0, 100, StopType.PICKUP),
            Stop("X", 3, 0, 100, StopType.VEHICLE_DROP)
        ]
        
        n = 4
        adj = np.full((n, n), INF)
        np.fill_diagonal(adj, 0)
        # V->All
        adj[0, 1] = 10; adj[0, 2] = 10; adj[0, 3] = 100 # V->X is slow! Forces V->A
        # A->B Fast
        adj[1, 2] = 5
        # Rest standard
        adj[1, 3] = 10; adj[3, 1] = 10
        adj[2, 3] = 10; adj[3, 2] = 10
        adj[2, 1] = 10
        
        prereqs = np.zeros(n, dtype=int)
        # All require V_Start
        prereqs[1:] |= 1
        
        graph = TripGraph(stops, adj, prereqs)
        
        # Inject Valid Cache for Invalid Capacity Path
        # u="A", rem={B, X}. Best is B->X.
        # Cost 5 (A->B) + 10 (B->X) = 15.
        # L = 100 - 15 = 85.
        # Load: B(+1), X(-1). max(0, +1, +1-1) = +1.
        
        graph.suffix_cache = {
             ("A", frozenset(["B", "X"])): (15.0, ["B", "X"], 85.0, 1)
        }
        
        # Solve
        # Start Load 3. 
        # Path V->A (Load becomes 3+1=4).
        # Cache Check at A: 
        #   Load=4. Cached MaxAdd=+1. Total=5. 5 > Cap 4.
        #   Cache FAIL (Soft).
        # Fall through to manual check.
        # Manual should pick A -> X -> B.
        
        res = graph.solve(vehicle_start_time=0, capacity=4, current_load=3)
        
        self.assertIsNotNone(res)
        # Check path: V->A->X->B
        path_ids = [s.id for s in res.route.stops]
        self.assertEqual(path_ids, ["A", "X", "B"])
        self.assertEqual(res.cost, 30.0)

    def test_solve_time_hard_pruning(self):
        """
        Test that B&B Hard Prunes on Time violation in cache.
        """
        stops = [
            Stop("V_Start", 0, 0, 100, StopType.VEHICLE_START),
            Stop("A", 1, 0, 10, StopType.PICKUP), # Deadline 10
            Stop("B", 2, 0, 100, StopType.PICKUP)
        ]
        n = 3
        adj = np.full((n, n), INF)
        np.fill_diagonal(adj, 0)
        adj[0, 1] = 20 # V->A late
        adj[0, 2] = 20 # V->B late
        # But maybe B->A is fast? 
        adj[2, 1] = 2
        
        prereqs = np.zeros(n, dtype=int)
        prereqs[1:] |= 1
        
        # Cache says best way B->A takes 2.
        # L_start at B for suffix {A} is 10 - 2 = 8.
        
        graph = TripGraph(stops, adj, prereqs)
        graph.suffix_cache = {
            ("B", frozenset(["A"])): (2.0, ["A"], 8.0, 1)
        }
        
        # V->B arrives at 20.
        # 20 > 8. Hard Fail.
        
        
        res = graph.solve(vehicle_start_time=0, capacity=4, current_load=0)
        self.assertIsNone(res)

    def test_solve_lazy_caching_populates_cache(self):
        """
        Verify that solving a blank graph populates the suffix cache on-demand.
        """
        stops = [
            Stop("V_Start", 0, 0, 100, StopType.VEHICLE_START),
            Stop("A", 1, 0, 100, StopType.PICKUP),
            Stop("B", 2, 0, 100, StopType.PICKUP)
        ]
        n = 3
        adj = np.full((n, n), INF)
        np.fill_diagonal(adj, 0)
        adj[0, 1] = 10; adj[0, 2] = 10
        adj[1, 2] = 2; adj[2, 1] = 20
        
        prereqs = np.zeros(n, dtype=int)
        prereqs[1:] |= 1
        
        graph = TripGraph(stops, adj, prereqs)
        # Empty Cache
        graph.suffix_cache = {}
        
        # Solving should populate cache for ("A", {B}) and ("B", {A}) etc.
        # Path V->A->B (Cost 10+2=12).
        res = graph.solve(0, 4, 0)
        self.assertEqual(res.cost, 12.0)
        
        # Identify "A", {B} key
        key = ("A", frozenset(["B"]))
        self.assertIn(key, graph.suffix_cache)
        # B is best next.
        # Cost B->End (Empty) is 0. Distance A->B is 2. Total 2.
        # L_start? B.latest=100. L_B = 100. L_A = 100 - 2 = 98.
        # Result: (2.0, ["B"], 98.0, 1) if B is pickup.
        
        val = graph.suffix_cache[key]
        self.assertEqual(val[0], 2.0)
        self.assertEqual(val[1], ["B"])
        self.assertEqual(val[2], 98.0)

if __name__ == '__main__':
    unittest.main()
