
from src.components.route_planner import RoutePlanner
from src.interfaces.data_models import Request, VehicleAvailability, Passenger, Stop, Route, NetworkPath, NetworkPathExtended
from src.interfaces import INetworkService
from typing import Iterable
import numpy as np

class MockNetworkService(INetworkService):
    def get_shortest_path(self, origin_node: int, destination_node: int) -> NetworkPath | None:
        return NetworkPath(travel_time=10.0, travel_distance=100.0)
    
    def get_shortest_path_with_travel_time(self, origin_node: int, destination_node: int) -> NetworkPathExtended | None:
        return NetworkPathExtended(travel_time=10.0, travel_distance=100.0, nodes=(origin_node, destination_node), travel_times=(10.0,))

    def get_travel_time_matrix(self, origins: list[int], destinations: list[int]) -> np.ndarray:
        return np.zeros((len(origins), len(destinations)))

    def prefetch_all_pairs(self, nodes: list[int]) -> None:
        pass

    def clear_l1_cache(self) -> None:
        pass

    def get_travel_times_l1_cache_subset(self, nodes: Iterable[int]) -> dict[tuple[int, int], float]:
        return {}

def test_handle_rtv_check_new():
    network_service = MockNetworkService()
    planner = RoutePlanner(network_service)
    
    # Setup
    v = VehicleAvailability(
        vehicle_id="v1",
        capacity=4,
        available_at_node=1,
        available_from_time=0.0,
        passengers_onboard=frozenset(),
        remaining_stops_to_serve=()
    )
    
    r1 = Request(id="r1", origin_node=2, destination_node=3, request_time=0.0, earliest_dropoff_time=100.0)
    requests_to_add = {r1}
    
    print("Testing _handle_rtv_check_new...")
    try:
        result = planner._handle_rtv_check_new(v, requests_to_add, find_best_route=True)
        print(f"Result: {result}")
        if not result.is_feasible:
             print(f"Infeasible reason: {getattr(result, 'reason', 'Unknown')}")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_handle_rtv_check_new()
