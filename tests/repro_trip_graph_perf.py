
import unittest
from unittest.mock import MagicMock
import numpy as np
from src.components.graph_builder.trip_graph import TripGraph
from src.interfaces import VehicleAvailability, Request, Passenger
from src.interfaces.services import INetworkService
from src.interfaces.data_models import FeasibilityResult

class TestTripGraphOldParams(unittest.TestCase):
    """
    Tests for src.components.graph_builder.trip_graph.TripGraph
    using the 'Old Code' parameters and structure.
    """

    def setUp(self):
        self.mock_network = MagicMock(spec=INetworkService)
        # Mock travel time: 10.0 between any two different nodes
        def get_travel_time_matrix(locs1, locs2):
            n1 = len(locs1)
            n2 = len(locs2)
            adj = np.zeros((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    if locs1[i] != locs2[j]:
                        adj[i, j] = 10.0
            return adj
        self.mock_network.get_travel_time_matrix.side_effect = get_travel_time_matrix

    def test_simple_sequence(self):
        """
        Scenario:
          Vehicle at Node 0.
          Request 1: Pick at 1, Drop at 2.
          Cost: V->1 (10) + 1->2 (10) = 20.
        """
        # 1. Setup Request
        req1 = Request(
            id="R1",
            origin_node=1,
            destination_node=2,
            request_time=0.0,
            earliest_dropoff_time=100.0, # plenty of time
        )
        
        # 2. Setup Vehicle
        vehicle = VehicleAvailability(
            vehicle_id="V1",
            available_at_node=0,
            available_from_time=0.0,
            capacity=4,
            passengers_onboard=frozenset(),
            remaining_stops_to_serve=(),
        )

        # 3. Build Graph
        graph = TripGraph.from_requests({req1}, self.mock_network)
        
        # 4. Solve
        result = graph.solve(vehicle, self.mock_network)
        
        self.assertTrue(result.is_feasible)
        # Route should be V -> R1_Pick -> R1_Drop
        # Cost: 10 + 10 = 20
        # Wait, get_dist assumes '10' for logic.
        # Delay Cost: Arrival at Drop is 20.
        # Req time is 0. Delay = 20 - 0 = 20.
        # Total Cost = 20.
        self.assertAlmostEqual(result.cost, 20.0)
        
        stops = result.route.stops
        self.assertEqual(len(stops), 2)
        self.assertEqual(stops[0].node, 1) # Pickup
        self.assertEqual(stops[1].node, 2) # Dropoff

    def test_onboard_passenger(self):
        """
        Scenario:
          Vehicle at Node 0. Has P1 onboard (Drop at 2).
          Request 2: Pick at 1, Drop at 3.
          
          Route options:
            A: V -> P1_Drop(2) -> R2_Pick(1) -> R2_Drop(3)
               Dist: 0->2=10, 2->1=10, 1->3=10. Total 30.
            B: V -> R2_Pick(1) -> P1_Drop(2) -> R2_Drop(3)
               Dist: 0->1=10, 1->2=10, 2->3=10. Total 30.
            C: V -> R2_Pick(1) -> R2_Drop(3) -> P1_Drop(2)
               Dist: 0->1=10, 1->3=10, 3->2=10. Total 30.
               
          Wait, costs are symmetric (10).
          However, usually we want to drop onboard passengers ASAP if it reduces delay?
          
          Delay calc:
            P1 (ReqTime 0). R2 (ReqTime 0).
            
            Case A:
              Arr 2 (t=10). P1 Delay=10.
              Arr 1 (t=20).
              Arr 3 (t=30). R2 Delay=30.
              Total Delay = 40.
              
            Case B:
              Arr 1 (t=10).
              Arr 2 (t=20). P1 Delay=20.
              Arr 3 (t=30). R2 Delay=30.
              Total Delay = 50.
              
            Case C:
              Arr 1 (t=10).
              Arr 3 (t=20). R2 Delay=20.
              Arr 2 (t=30). P1 Delay=30.
              Total Delay = 50.
              
            Optimal: Case A (Drop P1 first).
        """
        # Mock Passenger's Request
        req_p = Request(
            id="P1",
            origin_node=99, # Irrelevant, already picked up
            destination_node=2,
            request_time=0.0,
            earliest_dropoff_time=100.0,
        )
        passenger = Passenger(original_request=req_p, pickup_time=0.0)
        
        vehicle = VehicleAvailability(
            vehicle_id="V1",
            available_at_node=0,
            available_from_time=0.0,
            capacity=4,
            passengers_onboard=frozenset([passenger]),
            remaining_stops_to_serve=(),
        )
        
        req2 = Request(
            id="R2",
            origin_node=1,
            destination_node=3,
            request_time=0.0,
            earliest_dropoff_time=100.0,
        )
        
        graph = TripGraph.from_requests({req2}, self.mock_network)
        result = graph.solve(vehicle, self.mock_network)
        
        self.assertTrue(result.is_feasible)
        self.assertAlmostEqual(result.cost, 40.0) # From Case A
        
        # Check order: [P1_Drop, R2_Pick, R2_Drop] vs [R2_Pick, R2_Drop] (TripGraph returns both?) 
        # TripGraph returns reconstructed list.
        # P1_Drop is Node 2. R2_Pick is 1. R2_Drop is 3.
        # Expected: 2 -> 1 -> 3
        
        stop_nodes = [s.node for s in result.route.stops]
        self.assertEqual(stop_nodes, [2, 1, 3])

if __name__ == '__main__':
    unittest.main()
