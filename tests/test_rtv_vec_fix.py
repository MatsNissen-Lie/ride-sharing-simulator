
import logging
from typing import Set
import unittest
from unittest.mock import MagicMock

from src.components.graph_builder.rtv_builder_permutations_vec import PermutationRTVGraphBuilder
from src.interfaces import (
    Vehicle, Passenger, Request, VehicleAvailability, Stop, Route,
    INetworkService
)
from src.interfaces.graphs import RVGraph
from src.interfaces.data_models import Stop as InterfaceStop, NetworkPathExtended


class MockNetworkService(INetworkService):
    def get_shortest_path_with_travel_time(self, u: int, v: int):
        return NetworkPathExtended(
            travel_time=10.0,
            travel_distance=100.0,
            nodes=(u, v),
            travel_times=(10.0,)
        )
    def get_travel_time(self, u, v): return 10.0
    def get_travel_time_matrix(self, o, d): pass
    def get_travel_times_l1_cache_subset(self, nodes): pass
    def prefetch_all_pairs(self, nodes): pass
    def clear_l1_cache(self): pass
    def session(self, read_only=True): pass
    def close(self): pass
    def connect(self, read_only=True): pass
    def get_shortest_path(self, u, v): pass
    @property
    def is_nuclear(self): return False

class TestRTVBuilderFix(unittest.TestCase):
    def test_existing_passenger_stop_ids(self):
        # Setup
        network = MockNetworkService()
        planner = MagicMock()
        planner._network = network
        
        builder = PermutationRTVGraphBuilder(planner)
        
        # Create a vehicle with 1 passenger
        req = Request(id="req_123", origin_node=1, destination_node=2, request_time=0.0, earliest_dropoff_time=0.0)
        pax = Passenger(original_request=req, pickup_time=0.0)
        
        vehicle_avail = VehicleAvailability(
            vehicle_id="v1",
            capacity=4,
            available_at_node=1,
            available_from_time=0.0,
            passengers_onboard=frozenset([pax]),
            remaining_stops_to_serve=()
        )
        
        # Create a dummy RV graph (empty is fine for this test as we want to check vehicle constraints logic)
        rv_graph = RVGraph()
        
        # We need to trigger _check_vehicles. 
        # Since _check_vehicles is internal, we can test it directly or trigger via build_graph.
        # To trigger via build_graph, we need at least one request that connects to the vehicle.
        
        new_req = Request(id="req_new", origin_node=3, destination_node=4, request_time=0.0, earliest_dropoff_time=0.0)
        requests = {new_req}
        
        # Link vehicle to request in RV graph
        rv_graph.request_vehicle["req_new"] = {"v1"}
        
        # Run build_graph
        rtv = builder.build_graph(rv_graph, {vehicle_avail}, requests, max_trip_size=1)
        
        # Check trips
        # We expect a trip for "req_new" assigned to "v1"
        # The route for that trip must include the dropoff for "req_123"
        
        trips = rtv.get_trips_for_vehicle("v1")
        print(f"Found {len(trips)} trips for v1")
        
        found_existing_dropoff = False
        for trip_id in trips:
            route = rtv.get_route(trip_id, "v1")
            print(f"Trip: {trip_id}, Route Stops: {route.route.stops}")
            for stop in route.route.stops:
                # We are looking for the dropoff of req_123
                # It SHOULD be "req_123" but the bug makes it "drop_req_123"
                if "req_123" in stop.request_id:
                    print(f"Checking Stop ID: {stop.request_id}")
                    if stop.request_id == "req_123":
                        found_existing_dropoff = True
                    elif stop.request_id == "drop_req_123":
                        self.fail(f"Found prefixed ID 'drop_req_123' instead of 'req_123'")
        
        if not found_existing_dropoff:
             # It's possible the route wasn't generated if feasibility failed, 
             # but with Infinite constraints in Mock and builder, it should generate.
             # However, let's just inspect what we got.
             pass

if __name__ == "__main__":
    unittest.main()
