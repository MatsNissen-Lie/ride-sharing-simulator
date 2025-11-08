"""Filesystem-level constants for the project."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RIDER_DIR = DATA_DIR / "riders"
RIDER_PATTERN = "*_enriched.parquet"

# vehicle sample path
VEHICLE_SAMPLE_PATH = DATA_DIR / "vehicles" / "vehicle_samples.parquet"

__all__ = ["BASE_DIR", "DATA_DIR", "RIDER_DIR", "RIDER_PATTERN", "VEHICLE_SAMPLE_PATH"]
