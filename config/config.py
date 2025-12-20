"""Project-wide constant values."""

# Graph
OMEGA = 3 * 60  # 3 minutes in seconds
DELTA = 5 * 60  # 5 minutes in seconds
""" Default QoS constraints for requests."""


# SIM
BATCH_DURATION = 20  # 30 seconds
# data
NUMBER_OF_VEHICLES = 2500
VEHICLE_CAPACITY = 4  # vehicle capacity

# max passengers to consider in route planning
VEHICLE_CAPACITY = 4

MAX_REQUESTS_ABOVE_CAPACITY = 4


# simulation settings
PLOT_RV_GRAPH = False
PLOT_RTV_GRAPH = False

MIN_START_TIME = 0  # after 3 days
MIN_START_TIME = (
    2 * 24 * 3600 + 11 * 3600 + 7 * 60 + 35 - 10 * 60 - 30 - 30
)  # after 2 days 15 hours 7 minutes 35 seconds
WARMUP_DURATION = 0
WARMUP_DURATION = 15 * 60
END_TIME = MIN_START_TIME + WARMUP_DURATION + 3600 * 16


MAX_BATCHES = 5  # no limit by default
MAX_BATCHES = 1e10  # no limit by default

MIN_REQUESTS_TO_PROCESS = 0  # this should be zero for normal runs


MEGA_BATCH_MODE = False
MEGA_BATCH_SIZE = 250

__all__ = [
    "OMEGA",
    "DELTA",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
    "MAX_BATCHES",
    "MEGA_BATCH_MODE",
    "MEGA_BATCH_SIZE",
    "WARMUP_DURATION",
    "END_TIME",
]
