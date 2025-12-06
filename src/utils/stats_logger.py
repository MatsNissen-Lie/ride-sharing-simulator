import statistics
from collections import defaultdict
from typing import Dict, List, Any

from config import get_logger

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

    def log_batch(self, metrics: Dict[str, Any]) -> None:
        """
        Log a set of metrics for a single batch.
        
        Args:
            metrics: Dictionary containing metric names and values for this batch.
                     Numerical values will be aggregated for stats.
        """
        self._batch_count += 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._metrics[key].append(value)

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
