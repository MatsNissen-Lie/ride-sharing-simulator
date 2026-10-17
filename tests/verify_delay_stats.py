import unittest
from unittest.mock import MagicMock
from src.components.batch_controller import BatchController
from src.interfaces import Vehicle, Request, Passenger
from src.interfaces.data_models import Route, Stop, NetworkPathExtended

class TestDelayStats(unittest.TestCase):
    def test_stats_collection(self):
        # Mock Network Service
        mock_network = MagicMock()
        mock_network.get_travel_time.return_value = 5.0
        mock_network.get_shortest_path_with_travel_time.return_value = NetworkPathExtended(
            nodes=(1, 2),
            travel_times=(10.0,),
            travel_time=10.0,
            travel_distance=100.0,
        )

        # Create Request
        req = Request(
            id="r1",
            origin_node=1,
            destination_node=2,
            request_time=0.0,
            latest_pickup_time=100.0,
            latest_dropoff_time=200.0,
            promised_pickup_time=0.0
        )

        # Create Passenger (picked up at t=2.0)
        passenger = Passenger(original_request=req, pickup_time=2.0)

        # Create Vehicle with passenger and route to dropoff
        v1 = Vehicle(
            id="v1",
            capacity=4,
            passengers_onboard={passenger},
            last_visited_node=1,
            last_visited_time=0.0,
            current_route=Route(
                stops=(Stop(node=2, is_pickup=False, request_id="r1"),),
                total_travel_time=10.0,
                total_travel_distance=0,
                total_delay_cost=0,
            ),
            route_assignment_time=0.0,
        )

        controller = BatchController(network=mock_network, vehicles=[v1])
        # Register request to avoid "not in index" error if dropoff code checks index (it shouldn't for dropoff)
        # However, the dropoff logic checks: if passenger and cur_time > last_processed...
        # It also does `if target_id in self._request_index: raise` -> so we should NOT put it in index if it's onboard.
        
        # Act: Project fleet to t=20.0 (covers the dropoff at t=10.0)
        snapshots, _ = controller._project_fleet(t_now=20.0)

        # Assert
        stats = controller.consume_delivered_stats()
        print(f"Stats: {stats}")
        
        self.assertEqual(len(stats), 1)
        entry = stats[0]
        self.assertEqual(entry["req_id"], "r1")
        self.assertEqual(entry["vehicle_id"], "v1")
        
        # Wait time = pickup_time (2.0) - request_time (0.0) = 2.0
        self.assertEqual(entry["wait_time"], 2.0)
        
        # Dropoff time = 0.0 + 10.0 = 10.0
        # Direct travel time = 5.0 (mocked)
        # Total delay = 10.0 - 0.0 - 5.0 = 5.0
        self.assertEqual(entry["total_delay"], 5.0)

if __name__ == "__main__":
    unittest.main()
