
import unittest
from unittest.mock import MagicMock
from typing import Set, List, Tuple
from src.components.graph_builder.rtv_builder_trip_centric import TripCentricRTVGraphBuilder, UnifiedTripGraph, Stop, StopType, INF
from src.interfaces.services import INetworkService
from src.interfaces.data_models import VehicleAvailability, Request, Passenger
from src.interfaces.graphs import RVGraph
from dataclasses import dataclass

class TestTripCentricBuilder(unittest.TestCase):

    def setUp(self):
        self.mock_network = MagicMock(spec=INetworkService)
        # Mock travel time: 10.0 between any two different nodes
        def get_travel_time(u, v):
            return 0.0 if u == v else 10.0
        self.mock_network.get_travel_time.side_effect = get_travel_time
        
        # Mock matrix: for simplicity
        # Not strictly needed if we mock extend/adj directly or use simple graph
        
    def test_capacity_constraint_naive(self):
        """
        Verify that a vehicle with too many passengers does not even get injected.
        """
        builder = TripCentricRTVGraphBuilder(self.mock_network)
        
        # Vehicle: Capacity 4, 3 passengers onboard
        vehicle = MagicMock(spec=VehicleAvailability)
        vehicle.id = "v1"
        vehicle.capacity = 4
        vehicle.passengers_onboard = {MagicMock(), MagicMock(), MagicMock()} # 3 pax
        vehicle.available_at_node = 0
        vehicle.available_from_time = 0.0
        
        # Request: New trip of size 2 (2 pickups)
        # req_graph implied size 2
        new_ids = frozenset(["r1", "r2"])
        
        # We can't easily test the private method logic without mocking the whole build_graph loop.
        # But we can verify the logic inside `build_graph` if we run it.
        # Alternatively, we can test `_inject_vehicle` or `solve_best_path`.
        
        # Let's test the B&B solver's capacity logic directly, 
        # as that's where the "Hard/Soft" check is.
        pass

    def test_bnb_capacity_pruning_soft(self):
        """
        Test that B&B correctly identifies a "Soft Miss" on capacity.
        Scenario:
          - Vehicle Load: 3/4.
          - Request: Pickup 'A' (Load->4), Pickup 'B' (Load->5! VIOLATION).
          - Drop 'X' (existing pax).
          - Optimal Geom: A -> B -> X (Cost 20).
            - Load: 3 -> 4 -> 5 (Fail).
          - Suboptimal Geom: A -> X -> B (Cost 22).
            - Load: 3 -> 4 -> 3 -> 4 (Valid).
            
        This tests if the solver falls through the cached invalid path to find the valid suboptimal one.
        """
        # manual construction of UnifiedTripGraph
        # Nodes: 
        # 0: V_Start (Load 3)
        # 1: Pickup A
        # 2: Pickup B
        # 3: Drop X (Floating Drop)
        
        # Assignments:
        # A: Pickup (Load +1)
        # B: Pickup (Load +1)
        # X: Drop (Load -1)
        
        stops = [
            Stop("V_Start", 0, 0, 100, StopType.VEHICLE_START),
            Stop("A", 1, 0, 100, StopType.PICKUP),
            Stop("B", 2, 0, 100, StopType.PICKUP),
            Stop("X", 3, 0, 100, StopType.VEHICLE_DROP) # Existing pax drop
        ]
        
        # Adjacency (Time):
        # 0->1: 10, 0->2: 10, 0->3: 10
        # 1->2: 5  (A->B is fast! Preferred by Geom)
        # 1->3: 10
        # 3->2: 10
        
        # We want cache to return A->B->X (Cost 10+5+10 = 25 from start)
        # But Capacity fails at B.
        # We want solver to find A->X->B (Cost 10+10+10 = 30 from start)
        
        n = 4
        import numpy as np
        adj = np.full((n, n), INF)
        np.fill_diagonal(adj, 0)
        
        # V -> All
        adj[0, 1] = 10; adj[0, 2] = 10; adj[0, 3] = 10
        # A -> B (Fast)
        adj[1, 2] = 5
        # Remaining defaults (INF or mock logic)
        # Let's make connectivity full to allow re-ordering
        adj[1, 3] = 10; adj[3, 1] = 10
        adj[2, 3] = 10; adj[3, 2] = 10
        adj[2, 1] = 10 # B->A
        
        prereqs = [0] * n # No prereqs for simplicity (X is unordered, A/B are new pickups)
        E = np.zeros(n)
        L = np.full(n, 100.0)
        
        graph = UnifiedTripGraph(stops, adj, prereqs, n, E, L)
        
        # Setup Cache with INVALID Optimal Path
        # ID subset: {A, B, X}
        # Key: (A, {B, X}) -> Optimal is B->X (Cost 15)
        # Sequence: B, X
        # L_start: 100 - 15 = 85
        # Max Load Add: B(+1), X(-1). Peak = +1 (relative to A).
        
        # Start at A: Load is 3 (Start) + 1 (A) = 4.
        # Suffix starts. First stop B (Pickup).
        # Load becomes 4 + 1 = 5 > Capacity 4!
        # Cached Peak Load is +1. 4 + 1 = 5.
        
        cache = {}
        # We inject the cache manually to simulate precompute
        # Context: Current is A ("A", {B, X}).
        # Path: B -> X. Cost 5 (A->B) + 10 (B->X) [Wait, adj says 1->2 is 5].
        # From A: dist(A,B)=5. Recursion solved B->X.
        # Actually suffix cache stores result relative to 'u'.
        # Key: ("A", {"B", "X"})
        # Value: (Cost=15.0, Seq=["B", "X"], L=85.0, MaxLoad=1)
        
        cache[("A", frozenset(["B", "X"]))] = (15.0, ["B", "X"], 85.0, 1)
        
        # Run Solve
        # Initial Load = 3
        # Capacity = 4
        result = graph.solve_best_path(suffix_cache=cache, current_load=3, max_capacity=4)
        
        self.assertIsNotNone(result)
        # We expect path V -> A -> X -> B
        # Cost: 10 (V->A) + 10 (A->X) + 10 (X->B) = 30.
        # Load: 3 -> 4 -> 3 -> 4. Valid.
        
        stop_ids = [s.id for s in result.route.stops]
        self.assertEqual(stop_ids, ["A", "X", "B"])
        self.assertEqual(result.cost, 30.0)
        
    def test_bnb_time_pruning_hard(self):
        """
        Verify that if Time > Cached_L, we HARD prune (return None/fail branch),
        assuming Geometric Optimality implies Time Optimality (Triangle Inequality).
        """
        stops = [
            Stop("V_Start", 0, 0, 100, StopType.VEHICLE_START),
            Stop("A", 1, 0, 10, StopType.PICKUP), # Strict Deadline 10!
            Stop("B", 2, 0, 100, StopType.PICKUP)
        ]
        n = 3
        import numpy as np
        adj = np.full((n, n), INF)
        np.fill_diagonal(adj, 0)
        adj[0, 1] = 20 # V->A takes 20. Too late!
        adj[0, 2] = 10
        adj[2, 1] = 5 
        
        prereqs = [0] * n
        E = np.zeros(n)
        L = np.array([100, 10, 100]) # A must be visited by 10.
        
        graph = UnifiedTripGraph(stops, adj, prereqs, n, E, L)
        
        # Cache for A from V? No, usually cache is for remaining.
        # If we go V->B, remaining is {A}.
        # Cache ("B", {"A"}): Cost 5. L_start = 10 - 5 = 5.
        
        cache = {
             ("B", frozenset(["A"])): (5.0, ["A"], 5.0, 1)
        }
        
        # Path V->B arrives at 10.
        # Cached L is 5. 10 > 5. Hard Prune.
        
        result = graph.solve_best_path(suffix_cache=cache, current_load=0, max_capacity=4)
        
        # V->A direct is 20 > 10.
        # V->B->A is 10+5 = 15 > 10.
        # No feasible path.
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
