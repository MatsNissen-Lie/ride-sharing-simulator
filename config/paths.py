"""Filesystem-level constants for the project."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
]
