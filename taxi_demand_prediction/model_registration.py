"""
model_registration.py

Registers a previously logged MLflow model (stored in `run_information.json`)
under a given model name, then transitions it to a specified stage
(e.g., "Staging" or "Production").
"""

from pathlib import Path
from typing import Dict, Optional

import json
import mlflow
import dagshub
from loguru import logger
from mlflow.client import MlflowClient
from mlflow.entities import model_registry

from taxi_demand_prediction.config import (
    DAGSHUB_REPO_OWNER,
    DAGSHUB_REPO_NAME,
    MLFLOW_TRACKING_URI,
    RUN_INFO_PATH,
    REGISTERED_MODEL_NAME,
)

# Initialize Dagshub / MLflow tracking once at import time
dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ModelRegistrar:
    """
    Handles model registration and stage transition based on run info JSON.
    """

    def __init__(
        self,
        run_info_path: Optional[Path] = None,
        model_name: str = REGISTERED_MODEL_NAME,
        model_stage: str = "Staging",
    ) -> None:
        """Initializes a `ModelRegistrar` object.

        Args:
            run_info_path (Optional[Path], optional): Path to 'run_information.json'.
                Defaults to None.
            model_name (str, optional): Desired registered-model name. Defaults
                to REGISTERED_MODEL_NAME.
            model_stage (str, optional): Target stage (e.g., "Staging", "Production",
                etc.). Defaults to "Staging".
        """
        self.run_info_path: Path = run_info_path if run_info_path else RUN_INFO_PATH
        self.model_name: str = model_name
        self.model_stage: str = model_stage
        self.client = MlflowClient()

    def load_run_info(self) -> Dict[str, str]:
        """Loads the 'run_information.json' file.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If an error occurs while loading the file.

        Returns:
            Dict[str, str]: A dictionary with run_id, artifact_path, and
                model_uri.
        """
        logger.info(f"Loading run information from {self.run_info_path}...")
        if not self.run_info_path.exists():
            raise FileNotFoundError(f"Run-info JSON not found: {self.run_info_path}.")
        try:
            with open(self.run_info_path, "r") as f:
                info: Dict[str, str] = json.load(f)
            logger.info("Run information loaded.")
            return info
        except json.JSONDecodeError as e:
            logger.exception("Invalid JSON.")
            raise RuntimeError("Could not decode run-info JSON.") from e
        except Exception as e:
            logger.exception("Error loading run info.")
            raise

    def register_model(self, model_uri: str) -> model_registry.ModelVersion:
        """Registers the model in MLFlow's Model Registry.

        Args:
            model_uri (str): The model_uri.

        Raises:
            RuntimeError: If an error occurs while registering the model.

        Returns:
            model_registry.ModelVersion: A `ModelVersion` object.
        """
        logger.info(f"Registering model URI '{model_uri}' as '{self.model_name}'...")
        try:
            mv = mlflow.register_model(model_uri, self.model_name)
            logger.info(f"Model registered → name: {mv.name} | version: {mv.version}")
            return mv
        except Exception as e:
            logger.exception("Model registration failed.")
            raise RuntimeError("Failed to register model.") from e

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = False,
    ) -> model_registry.ModelVersion:
        """Transitions a model version to a new stage.

        Args:
            name (str): Registered-model name.
            version (str): Model version.
            stage (str): Target stage ("Staging", "Production", etc.).
            archive_existing (bool, optional): Whether to archive existing versions
                in that stage. Defaults to False.

        Raises:
            RuntimeError: If an error occurs while transitioning.

        Returns:
            model_registry.ModelVersion: Updated `ModelVersion` object.
        """
        logger.info(f"Moving model '{name}' v{version} → stage '{stage}'...")
        try:
            mv = self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info(
                f"Transition complete → {mv.name}:{mv.version} is now in '{mv.current_stage}'."
            )
            return mv
        except Exception as e:
            logger.exception("Stage transition failed.")
            raise RuntimeError("Failed to transition model stage.") from e

    def run_registration(self) -> None:
        """
        Orchestrates:
          1. Load run-info JSON to get model_uri
          2. Register the model under desired name
          3. Transition it to the configured stage
        """
        logger.info("Starting model registration pipeline...")
        try:
            info = self.load_run_info()
            mv = self.register_model(info["model_uri"])
            self.transition_stage(mv.name, mv.version, self.model_stage)
            logger.success(
                f"Model '{mv.name}' v{mv.version} registered & moved to '{self.model_stage}'."
            )
        except Exception as e:
            logger.error("Model registration pipeline failed.")
            raise


def main() -> None:
    """
    Runs the model-registration workflow.
    """
    logger.info("Launching main() for model registration...")
    registrar = ModelRegistrar()
    registrar.run_registration()
    logger.success("Model registration process completed.")


if __name__ == "__main__":
    main()
