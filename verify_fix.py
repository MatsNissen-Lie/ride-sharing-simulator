
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
from collections import defaultdict

# Add src to path
import os
sys.path.append(os.getcwd())

from src.components.graph_builder.rtv_builder_min import TripCentricRTVGraphBuilder
from src.interfaces import ISingleVehicleRoutePlanner, VehicleAvailability, Request
from src.interfaces.services import INetworkService
from src.interfaces.graphs import RVGraph
from src.interfaces.data_models import FeasibilityResult, FeasibleRoute, Route, Stop

class ReproduceCrash(unittest.TestCase):
    def test_k4_crash_fixed(self):
        # 1. Mock Route Planner & Network
        mock_route_planner = MagicMock(spec=ISingleVehicleRoutePlanner)
        mock_network = MagicMock(spec=INetworkService)
        
        # Setup Global Matrix (Size 20 to accommodate node IDs up to 7*2+1)
        matrix_size = 20
        # Simple travel time: 0 if same node, 5.0 otherwise
        global_matrix = np.full((matrix_size, matrix_size), 5.0)
        np.fill_diagonal(global_matrix, 0.0)
        
        mock_network.get_matrix_and_mapping.return_value = (global_matrix, {i: i for i in range(matrix_size)})
        mock_route_planner._network = mock_network

        # 2. Setup requests and vehicle
        # We need 4 requests to reach k=4
        reqs = []
        for i, rid in enumerate(["R1", "R2", "R3", "R4"]):
            r = MagicMock(spec=Request)
            r.id = rid
            r.origin_node = i*2
            r.destination_node = i*2 + 1
            r.request_time = 0.0
            r.earliest_dropoff_time = 1000.0
            reqs.append(r)
        
        req_set = set(reqs)
        
        vehicle = MagicMock(spec=VehicleAvailability)
        vehicle.vehicle_id = "V1"
        vehicle.capacity = 10 # Large capacity to allow k=4
        vehicle.passengers_onboard = []
        vehicle.available_at_node = 15 # Some other node
        vehicle.available_from_time = 0.0
        
        vehicles = {vehicle}
        
        # 3. Build RV Graph (Fully connected)
        rv_graph = MagicMock(spec=RVGraph)
        
        # Requests -> Vehicle
        rv_graph.request_vehicle = {r.id: {"V1"} for r in reqs}
        rv_graph.rv_details = {(r.id, "V1"): MagicMock() for r in reqs}
        
        # Requests -> Requests (Fully connected cloque for simplicity)
        # Each request is neighbor to all others
        rv_graph.request_request = {
            r.id: {other.id for other in reqs if other.id != r.id}
            for r in reqs
        }

        # 4. Run Builder
        builder = TripCentricRTVGraphBuilder(mock_route_planner)
        
        print("Starting RTV Build (Zero-Copy)...")
        try:
            rtv = builder.build_graph(rv_graph, vehicles, req_set)
            print("Finished RTV Build successfully.")
            
            # Verify that we actually found the k=4 trip
            # k=4 trip ID would be frozenset(["R1", "R2", "R3", "R4"])
            trips = rtv.trips
            found_k4 = False
            for t_id in trips:
                if len(t_id) == 4:
                    print(f"Found K=4 Trip: {t_id}")
                    found_k4 = True
            
            self.assertTrue(found_k4, "Should have found the K=4 trip")

        except AttributeError as e:
            print(f"Caught ERROR: {e}")
            self.fail(f"Should not crash! Error: {e}")
        except Exception as e:
            # print stack trace
            import traceback
            traceback.print_exc()
            self.fail(f"Unexpected error: {e}")

if __name__ == '__main__':
    unittest.main()
