import sys
from pathlib import Path

# add workspace to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.simple_db import DuckDBPathCache
from config import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Clearing database...")
    cache = DuckDBPathCache()
    cache.clear_all(force=True)
    logger.info("Database cleared.")

if __name__ == "__main__":
    main()
