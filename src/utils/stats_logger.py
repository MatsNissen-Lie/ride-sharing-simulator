import statistics
from collections import defaultdict
from typing import Dict, List, Any, Union
from dataclasses import asdict

from config import get_logger
from src.interfaces.stats import BatchStatistics

logger = get_logger(__name__)


class SimulationStatsLogger:
    """
    Collects runtime metrics and statistics during the simulation
    and provides a summary report upon completion.
    """

    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._scalars: Dict[str, Any] = {}
        self._batch_count = 0

    def log_batch(self, metrics: Union[Dict[str, Any], BatchStatistics]) -> None:
        """
        Log a set of metrics for a single batch.
        
        Args:
            metrics: Dictionary or BatchStatistics object containing metric names and values for this batch.
                     Numerical values will be aggregated for stats.
        """
        if isinstance(metrics, BatchStatistics):
            metrics = asdict(metrics)

        self._batch_count += 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._metrics[key].append(value)

    def log_delivered_requests(self, requests: List[Dict[str, Any]]) -> None:
        """
        Log detailed stats for requests delivered in this batch.
        Calculates and logs aggregate metrics (averages) for the batch.
        """
        # Calculate aggregates for this batch
        count = len(requests)
        avg_delay = 0.0
        avg_wait = 0.0
        
        if count > 0:
            total_delay = sum(r["total_delay"] for r in requests)
            total_wait = sum(r["wait_time"] for r in requests)
            avg_delay = total_delay / count
            avg_wait = total_wait / count
            
        # Log to batch metrics
        # We append directly to keep aligned with log_batch calls
        self._metrics["realized_avg_delay"].append(avg_delay)
        self._metrics["realized_avg_wait_time"].append(avg_wait)
        self._metrics["delivered_count"].append(count)

    def log_expired_requests(self, requests: List[Dict[str, Any]]) -> None:
        """
        Log stats for expired requests in this batch.
        """
        count = len(requests)
        avg_wait = 0.0
        
        if count > 0:
            total_wait = sum(r["wait_time"] for r in requests)
            avg_wait = total_wait / count
            
        self._metrics["expired_count"].append(count)
        self._metrics["avg_expired_wait_time"].append(avg_wait)

    def log_scalar_stat(self, key: str, value: Any) -> None:
        """
        Log a single scalar statistic that summarizes the entire simulation
        (e.g. total unique requests).
        """
        self._scalars[key] = value

    def print_summary(self) -> None:
        """
        Prints a summary table of collected statistics to the log.
        """
        if self._batch_count == 0:
            logger.info("No batches processed. No statistics to report.")
            return

        logger.info("=" * 60)
        logger.info(f"SIMULATION STATISTICS SUMMARY (Total Batches: {self._batch_count})")
        logger.info("=" * 60)
        
        # --- Scalars ---
        if self._scalars:
            for key, value in sorted(self._scalars.items()):
                 logger.info(f"{key:<30} : {value}")
            logger.info("-" * 60)

        logger.info(f"{'Metric':<30} | {'Mean':<10} | {'Min':<10} | {'Max':<10}")
        logger.info("-" * 66)

        for key, values in sorted(self._metrics.items()):
            if not values:
                continue
            
            try:
                avg = statistics.mean(values)
                min_val = min(values)
                max_val = max(values)
                
                # Format based on magnitude
                if "time" in key.lower() or "duration" in key.lower():
                     # Times in seconds
                    fmt = "{:.4f}s"
                else:
                    # Counts or sizes
                    fmt = "{:.2f}"
                
                # Additional formatting for integers if strict
                
                logger.info(
                    f"{key:<30} | {fmt.format(avg):<10} | {fmt.format(min_val):<10} | {fmt.format(max_val):<10}"
                )
            except Exception:
                logger.warning(f"Could not calculate stats for {key}")

        logger.info("=" * 60)

    def export_to_csv(self, filepath: str) -> None:
        """
        Exports the collected metrics to a CSV file.
        Each row represents a batch.
        """
        import csv
        import os

        if not self._metrics:
            logger.warning("No metrics to export.")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        fieldnames = sorted(self._metrics.keys())
        num_batches = len(self._metrics[fieldnames[0]])

        try:
            with open(filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for i in range(num_batches):
                    row = {key: self._metrics[key][i] for key in fieldnames}
                    writer.writerow(row)
            
            logger.info(f"Statistics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
