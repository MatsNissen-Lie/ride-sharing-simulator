import os
import datetime
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SimulationOutputManager:
    def __init__(self, base_dir: str = "reports", run_name: str = "sim"):
        self.base_dir = base_dir
        self.run_name = run_name
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self._create_run_dir()

    def _create_run_dir(self) -> str:
        """Create a new timestamped directory for this run."""
        dir_name = f"{self.timestamp}_{self.run_name}"
        full_path = os.path.abspath(os.path.join(self.base_dir, dir_name))
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"📁 Created run directory: {full_path}")
        return full_path

    def get_run_dir(self) -> str:
        """Return the absolute path to the current run directory."""
        return self.run_dir

    def save_metadata(self, config_module: Any) -> None:
        """
        Save configuration variables to metadata.json.
        Excludes variables starting with 'MIN_' or containing 'min' if requested,
        but typically we just grab upper case ones.
        """
        metadata = {}
        # Extract all UPPERcase variables from the config module
        for key in dir(config_module):
            if key.isupper():
                # Filter out MIN_ variables as per user feedback
                if key.startswith("MIN_"):
                    continue
                metadata[key] = getattr(config_module, key)

        metadata_path = os.path.join(self.run_dir, "metadata.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"💾 Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
