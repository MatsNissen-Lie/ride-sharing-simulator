"""Project-wide constant values."""

# Graph
OMEGA = 3 * 60  # 3 minutes in seconds
DELTA = 5 * 60  # 5 minutes in seconds
""" Default QoS constraints for requests."""


# SIM
BATCH_DURATION = 30  # 30 seconds
# data
NUMBER_OF_VEHICLES = 300


__all__ = [
    "OMEGA",
    "DELTA",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
]
