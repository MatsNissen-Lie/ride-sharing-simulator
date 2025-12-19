import unittest
from src.utils.stats_logger import SimulationStatsLogger

class TestStatsLoggerConsolidation(unittest.TestCase):
    def test_aggregation_no_storage(self):
        logger = SimulationStatsLogger()
        
        # Log a batch of delivered requests
        requests = [
            {"total_delay": 10.0, "wait_time": 2.0},
            {"total_delay": 20.0, "wait_time": 4.0}
        ]
        logger.log_delivered_requests(requests)
        
        # Check aggregates
        self.assertEqual(logger._metrics["realized_avg_delay"][-1], 15.0)
        self.assertEqual(logger._metrics["realized_avg_wait_time"][-1], 3.0)
        self.assertEqual(logger._metrics["delivered_count"][-1], 2)
        
        # Verify raw data is NOT stored (if _delivered_requests attr exists, it should be empty or undefined)
        if hasattr(logger, "_delivered_requests"):
            self.assertEqual(len(logger._delivered_requests), 0, "Raw requests should not be stored!")
            
    def test_export_method_removed(self):
        logger = SimulationStatsLogger()
        self.assertFalse(hasattr(logger, "export_trips_to_csv"), "export_trips_to_csv should be removed")

if __name__ == "__main__":
    unittest.main()
