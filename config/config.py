"""Project-wide constant values."""

# Quality-of-service constraints (seconds)
OMEGA = 3 * 60  # maximum waiting time
DELTA = 5 * 60  # maximum delay (waiting + in-vehicle), relative to t*

# Simulation
BATCH_DURATION = 10  # seconds between assignments
NUMBER_OF_VEHICLES = 1500
VEHICLE_CAPACITY = 4

# Trips may hold more requests than free seats when riders are served in sequence.
MAX_REQUESTS_ABOVE_CAPACITY = 4

PLOT_RV_GRAPH = False
PLOT_RTV_GRAPH = False

# Simulation window, in seconds after the first request in the rider data.
# December 2025 runs started 2 days 10:56:35 into January 2025.
MIN_START_TIME = 2 * 24 * 3600 + 10 * 3600 + 56 * 60 + 35
WARMUP_DURATION = 15 * 60
END_TIME = MIN_START_TIME + WARMUP_DURATION + 2 * 24 * 3600

__all__ = [
    "OMEGA",
    "DELTA",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
    "WARMUP_DURATION",
    "END_TIME",
]
