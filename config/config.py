"""Project-wide constant values."""

# Graph
OMEGA = 2 * 60  # 3 minutes in seconds
DELTA = 4 * 60  # 5 minutes in seconds
""" Default QoS constraints for requests."""


# SIM
BATCH_DURATION = 30  # 30 seconds
# data
NUMBER_OF_VEHICLES = 3000
VEHICLE_CAPACITY = 4  # vehicle capacity

# max passengers to consider in route planning
VEHICLE_CAPACITY = 6

MAX_REQUESTS_ABOVE_CAPACITY = 4


# simulation settings
PLOT_RV_GRAPH = False
PLOT_RTV_GRAPH = False

MIN_START_TIME = 3 * 24 * 3600  # after 3 days
MIN_REQUESTS_TO_PROCESS = 400


MAX_BATCHES = 1

__all__ = [
    "OMEGA",
    "DELTA",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
    "MAX_BATCHES",
]
