"""Project-wide constant values."""

# Graph
OMEGA = 2 * 60  # 3 minutes in seconds
DELTA = 4 * 60  # 5 minutes in seconds
""" Default QoS constraints for requests."""


# SIM
BATCH_DURATION = 30  # 30 seconds
# data
NUMBER_OF_VEHICLES = 20

# max passengers to consider in route planning
MAX_PASSENGERS = 4
MAX_REQUESTS = 4
MAX_REQUESTS_ABOVE_CAPACITY = 2


# simulation settings
PLOT_RV_GRAPH = True
PLOT_RTV_GRAPH = True

MIN_START_TIME = 0  # 6 * 3600  # 6 AM in seconds
MIN_REQUESTS_TO_PROCESS = 20


__all__ = [
    "OMEGA",
    "DELTA",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
]
