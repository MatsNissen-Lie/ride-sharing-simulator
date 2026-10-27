"""Project-wide constant values."""

# Quality-of-service constraints (seconds)
OMEGA = 3 * 60  # maximum waiting time
DELTA = 5 * 60  # maximum delay (waiting + in-vehicle), relative to t*

# Traffic: simulated travel time = free-flow (OSM speed limit) time x this factor.
# Calibrated on Jan 2025 Manhattan trips (actual / free-flow time, median):
# ~1.9 at 04:00, 2.3 at 07:00, 2.7-3.0 from 08:00 to 19:00 (2.91 at 18:00), so 2.9
# matches the weekday evening peak used in the experiments. 1.0 = free flow (runs
# before 2026-10-27). Recompute: scripts/analysis/travel_time_factor.py
TRAVEL_TIME_FACTOR = 2.9

# Economics (USD). Solo fare model fitted on Manhattan solo HVFHS trips, Jan 2025:
# fare = base + per_km * km + per_min * min (scripts/analysis/fare_model.py).
FARE_BASE = 5.41
FARE_PER_KM = 0.93
FARE_PER_MIN = 0.79
# Riders who accept pooling pay (1 - discount) up front, matched or not (median in the data).
POOLING_DISCOUNT = 0.24
# Variable operating cost per vehicle-km; scenarios 0.10 / 0.25 / 0.50.
COST_PER_KM = 0.25

# Simulation
BATCH_DURATION = 10  # seconds between assignments
NUMBER_OF_VEHICLES = 1500
VEHICLE_CAPACITY = 4

# Trips may hold more requests than free seats when riders are served in sequence.
MAX_REQUESTS_ABOVE_CAPACITY = 4

PLOT_RV_GRAPH = False
PLOT_RTV_GRAPH = False

# Simulation window, in seconds after the first request in the rider data.
# The default start is 2 days 10:56:35 after the first request (3 January 2025 for January data).
MIN_START_TIME = 2 * 24 * 3600 + 10 * 3600 + 56 * 60 + 35
WARMUP_DURATION = 15 * 60
END_TIME = MIN_START_TIME + WARMUP_DURATION + 2 * 24 * 3600

__all__ = [
    "OMEGA",
    "DELTA",
    "TRAVEL_TIME_FACTOR",
    "BATCH_DURATION",
    "NUMBER_OF_VEHICLES",
    "WARMUP_DURATION",
    "END_TIME",
]
