"""Filesystem-level constants for the project."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


__all__ = ["BASE_DIR"]
