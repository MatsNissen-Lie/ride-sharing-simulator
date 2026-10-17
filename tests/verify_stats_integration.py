import unittest
from unittest.mock import MagicMock
from src.components.batch_controller import BatchController
from src.interfaces import Vehicle, Request
from config.config import OMEGA

class TestStatsIntegration(unittest.TestCase):
    def test_expiry_tracking(self):
        # Mock Network
        mock_network = MagicMock()
        
        # Create Vehicle (idle)
        v1 = Vehicle(id="v1", capacity=4, passengers_onboard=set(), last_visited_node=1, last_visited_time=0.0)
        
        controller = BatchController(network=mock_network, vehicles=[v1])
        
        # Add a request that is OLD
        # OMEGA is usually 600 or similar. Let's make t_now much larger than request_time + OMEGA
        req = Request(
            id="r_expired",
            origin_node=1, 
            destination_node=2, 
            request_time=100.0,
            latest_pickup_time=200.0,
            latest_dropoff_time=300.0
        )
        
        # Manually inject request
        controller._request_index[req.id] = req
        
        # Act: Prepare batch at a time that expires the request
        # t_now = request_time + OMEGA + 1
        t_expire = req.request_time + OMEGA + 10.0
        
        # This calls _expire_stale_requests internally
        controller.prepare_batch(t_now=t_expire, new_meta=[])
        
        # Assert: Request should be gone from index
        self.assertNotIn("r_expired", controller._request_index)
        
        # Assert: Expiry stats should be captured
        stats = controller.consume_expired_stats()
        print(f"Expired Stats: {stats}")
        
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["req_id"], "r_expired")
        self.assertTrue(stats[0]["wait_time"] > OMEGA)

if __name__ == "__main__":
    unittest.main()
