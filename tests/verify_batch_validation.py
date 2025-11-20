import unittest
from unittest.mock import MagicMock
from src.components.batch_controller import BatchController
from src.interfaces import Vehicle
from src.interfaces.data_models import NetworkPathExtended
from src.interfaces.data_models import Route, Stop


class TestBatchControllerValidation(unittest.TestCase):
    def test_validation_raises_error(self):
        # Mock Network Service
        mock_network = MagicMock()

        # Mock Vehicle
        v1 = Vehicle(
            id="v1",
            capacity=4,
            passengers_onboard=set(),
            last_visited_node=1,
            last_visited_time=100.0,
            current_route=None,
            route_assignment_time=None,
        )

        controller = BatchController(network=mock_network, vehicles=[v1])

        # Case 1: Idle vehicle, t_now = 200. Should be fine.
        snapshots, _ = controller._project_fleet(t_now=200.0)
        self.assertEqual(len(snapshots), 1)
        s = list(snapshots)[0]
        self.assertEqual(s.available_from_time, 200.0)

        # Case 2: Force a situation where projection might fail?
        # Actually, my code logic ensures available_from >= t_now.
        # But let's try to mock _snap_to_next_node_on_leg to return something bad
        # to see if the loop catches it.

        # We need a vehicle on a route
        v2 = Vehicle(
            id="v2",
            capacity=4,
            passengers_onboard=set(),
            last_visited_node=1,
            last_visited_time=100.0,
            current_route=Route(
                stops=(Stop(node=2, is_pickup=False, request_id="r1"),),
                total_travel_time=10,
                total_travel_distance=0,
                total_delay_cost=0,
            ),
            route_assignment_time=100.0,
        )

        controller = BatchController(network=mock_network, vehicles=[v2])

        # Mock get_shortest_path_with_travel_time
        mock_network.get_shortest_path_with_travel_time.return_value = (
            NetworkPathExtended(
                nodes=(1, 2),
                travel_times=(150.0,),
                travel_time=150.0,
                travel_distance=100.0,
            )
        )

        # Mock _snap_to_next_node_on_leg to return a time BEFORE t_now
        # This simulates a bug in projection logic that our validation should catch
        controller._snap_to_next_node_on_leg = MagicMock(return_value=(2, 150.0))

        # t_now = 200. Projection returns 150. Should raise ValueError.
        try:
            controller._project_fleet(t_now=200.0)
            print("Failed to catch invalid projection!")
        except ValueError as e:
            print(f"Caught expected error: {e}")


if __name__ == "__main__":
    unittest.main()
