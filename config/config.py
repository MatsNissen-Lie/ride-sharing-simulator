"""Project-wide constant values."""

# Graph
OMEGA = 3 * 60  # 3 minutes in seconds
DELTA = 5 * 60  # 5 minutes in seconds
""" Default QoS constraints for requests."""


# SIM
BATCH_DURATION = 30  # 30 seconds
# data
NUMBER_OF_VEHICLES = 3000

# max passengers to consider in route planning
MAX_PASSENGERS = 4
MAX_REQUESTS = 4
MAX_REQUESTS_ABOVE_CAPACITY = 2

PLOT_RV_GRAPH = False
PLOT_RTV_GRAPH = False


__all__ = [
    "OMEGA",
    "DELTA",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
]
