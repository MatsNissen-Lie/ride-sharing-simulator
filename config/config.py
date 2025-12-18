"""Project-wide constant values."""

# Graph
OMEGA = 3 * 60  # 3 minutes in seconds
DELTA = 5 * 60  # 5 minutes in seconds
""" Default QoS constraints for requests."""


# SIM
BATCH_DURATION = 30  # 30 seconds
# data
NUMBER_OF_VEHICLES = 25
VEHICLE_CAPACITY = 4  # vehicle capacity

# max passengers to consider in route planning
VEHICLE_CAPACITY = 3

MAX_REQUESTS_ABOVE_CAPACITY = 0


# simulation settings
PLOT_RV_GRAPH = True
PLOT_RTV_GRAPH = True

MIN_START_TIME = 3 * 24 * 3600  # after 3 days
MIN_START_TIME = 0  # after 3 days
MIN_REQUESTS_TO_PROCESS = 30


MAX_BATCHES = 1

##### Otherwise use mega-batching ######
MEGA_BATCH_MODE = True
MEGA_BATCH_MODE = False
MEGA_BATCH_SIZE = 250

# WARMUP
WARMUP_DURATION = 15 * 60  # 15 minutes

__all__ = [
    "OMEGA",
    "DELTA",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
    "MAX_BATCHES",
    "MEGA_BATCH_MODE",
    "MEGA_BATCH_SIZE",
    "WARMUP_DURATION",
]
