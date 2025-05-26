"""
model_evaluation.py

Evaluates a trained regression model on the test dataset, logs metrics &
artifacts to MLflow / Dagshub, and stores run information in JSON.

Steps
-----
1. Read train & test CSVs
2. Load encoder & model (joblib)
3. Encode X_test
4. Predict & compute metrics (MAPE)
5. Log params, metrics, datasets, and model to MLflow
6. Persist the run information to JSON
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import json
import joblib
import mlflow
import dagshub
import pandas as pd
from loguru import logger
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from mlflow.models import infer_signature

from taxi_demand_prediction.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    ENCODER_PATH,
    MODEL_PATH,
    RUN_INFO_PATH,
    DAGSHUB_REPO_OWNER,
    DAGSHUB_REPO_NAME,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

# Ensure sklearn transformers output pandas
set_config(transform_output="pandas")


class ModelEvaluator:
    """
    Encapsulates logic for evaluating a trained model, logging to MLflow,
    and saving run information.
    """

    def __init__(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        encoder_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        run_info_path: Optional[Path] = None,
    ):
        """Initializes a `ModelTrainer` object with configurable file paths.

        Args:
            train_path (Optional[Path], optional): Path to the training data.
                Defaults to None.
            test_path (Optional[Path], optional): Path to the test data.
                Defaults to None.
            encoder_path (Optional[Path], optional): Path to the saved encoder
                object. Defaults to None.
            model_path (Optional[Path], optional): Path to the saved model object.
                Defaults to None.
            run_info_path (Optional[Path], optional): Path to the 'run_information.json'
                file. Defaults to None.
        """
        self.train_path = train_path if train_path else TRAIN_DATA_PATH
        self.test_path = test_path if test_path else TEST_DATA_PATH
        self.encoder_path = encoder_path if encoder_path else ENCODER_PATH
        self.model_path = model_path if model_path else MODEL_PATH
        self.run_info_path = run_info_path if run_info_path else RUN_INFO_PATH

        # MLflow / Dagshub initialisation
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Lazy-loaded objects
        self.encoder: Optional[ColumnTransformer] = None
        self.model: Optional[LinearRegression] = None

    def read_csv(self, path: Path) -> pd.DataFrame:
        """Reads and returns a CSV file.

        Args:
            path (Path): Path of the CSV file.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            RuntimeError: If an error occurs while reading the CSV file.

        Returns:
            pd.DataFrame: Data in the CSV file.
        """
        logger.info(f"Reading data from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}.")
        try:
            df = pd.read_csv(path, parse_dates=["tpep_pickup_datetime"])
            logger.info("Data read successfully.")
            return df
        except Exception as e:
            logger.exception("Failed to read CSV.")
            raise RuntimeError(f"Could not read {path}.") from e

    def get_train_test(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reads and returns the training and test CSV files as DataFrames.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
        """
        train_df = self.read_csv(self.train_path)
        test_df = self.read_csv(self.test_path)
        return train_df, test_df

    def load_encoder(self) -> ColumnTransformer:
        """Loads and returns an encoder object.

        Raises:
            RuntimeError: If an error occurs while loading the object.

        Returns:
            ColumnTransformer: Encoder object.
        """
        logger.info(f"Loading encoder from {self.encoder_path}...")
        try:
            encoder: ColumnTransformer = joblib.load(self.encoder_path)
            logger.info("Encoder loaded.")
            return encoder
        except Exception as e:
            logger.exception("Failed to load encoder.")
            raise RuntimeError(f"Could not load encoder at {self.encoder_path}.") from e

    def load_model(self) -> LinearRegression:
        """Loads and returns a model object.

        Raises:
            RuntimeError: If an error occurs while loading the object.

        Returns:
            LinearRegression: Model object.
        """
        logger.info(f"Loading regression model from {self.model_path}...")
        try:
            model: LinearRegression = joblib.load(self.model_path)
            logger.info("Model loaded.")
            return model
        except Exception as e:
            logger.exception("Failed to load model.")
            raise RuntimeError(f"Could not load model at {self.model_path}.") from e

    @staticmethod
    def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Creates X and y using the full data.

        Args:
            df (pd.DataFrame): The full data.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: X and y.
        """
        df = df.copy()
        df.set_index("tpep_pickup_datetime", inplace=True, drop=True)
        X = df.drop(columns=["total_pickups"])
        y = df["total_pickups"]
        return X, y

    @staticmethod
    def compute_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Computes the mean absolute percentage error.

        Args:
            y_true (pd.Series): Ground truth.
            y_pred (pd.Series): Prediction.

        Returns:
            float: Mean absolute percentage error.
        """
        return mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)

    def log_to_mlflow(
        self,
        model: LinearRegression,
        X_test_encoded: pd.DataFrame,
        y_pred: pd.Series,
        y_test: pd.Series,
        loss: float,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Logs parameters, metrics, datasets, and model to MLFlow.

        Args:
            model (LinearRegression): Model to log.
            X_test_encoded (pd.DataFrame): Encoded X_test to infer model signature.
            y_pred (pd.Series): Test predictions.
            y_test (pd.Series): Test ground truth.
            loss (float): Mean absolute percentage error.
            train_df (pd.DataFrame): Training data.
            test_df (pd.DataFrame): Test data.

        Returns:
            Dict[str, Any]: A dictionary containing run_id, artifact_path, and
                model_uri.
        """
        with mlflow.start_run(run_name="LinearRegression") as run:
            # 1. parameters
            mlflow.log_params(model.get_params())

            # 2. metric
            mlflow.log_metric("MAPE", loss)

            # 3. datasets
            train_dataset = mlflow.data.from_pandas(
                train_df.set_index("tpep_pickup_datetime"),
                targets="total_pickups",
            )
            test_dataset = mlflow.data.from_pandas(
                test_df.set_index("tpep_pickup_datetime"),
                targets="total_pickups",
            )
            mlflow.log_input(train_dataset, "training")
            mlflow.log_input(test_dataset, "validation")

            # 4. model
            signature = infer_signature(X_test_encoded, y_pred)
            logged_model = mlflow.sklearn.log_model(
                model,
                "taxi_demand_prediction",
                signature=signature,
                pip_requirements="requirements.txt",
            )

        logger.info("MLflow logging completed.")
        return {
            "run_id": logged_model.run_id,
            "artifact_path": logged_model.artifact_path,
            "model_uri": logged_model.model_uri,
        }

    @staticmethod
    def save_run_info(info: Dict[str, Any], path: Path) -> None:
        """Saves run_id, artifact_path, and model_uri in the 'run_information.json'
            file.

        Args:
            info (Dict[str, Any]): A dictionary containing run_id, artifact_path,
                and model_uri.
            path (Path): Path of the 'run_information.json' file.

        Raises:
            RuntimeError: If an error occurs while saving the information.
        """
        logger.info(f"Saving run info to {path}")
        try:
            with open(path, "w") as f:
                json.dump(info, f, indent=4)
            logger.info("Run info saved.")
        except Exception as e:
            logger.exception("Failed to save run info.")
            raise RuntimeError(f"Could not write run info JSON to {path}.") from e

    def run_evaluation(self) -> None:
        """
        Executes the full evaluation & logging workflow.
        """
        logger.info("Starting model evaluation pipeline...")
        try:
            # Loading the data
            train_df, test_df = self.get_train_test()
            X_train, y_train = self.prepare_X_y(train_df)
            X_test, y_test = self.prepare_X_y(test_df)

            # Loading the encoder & model
            self.encoder = self.load_encoder()
            self.model = self.load_model()

            # Encoding the X_test
            X_test_encoded = self.encoder.transform(X_test)

            # Prediction & metric
            y_pred = self.model.predict(X_test_encoded)
            loss = self.compute_mape(y_test, y_pred)
            logger.info(f"Mean Absolute Percentage Error (MAPE): {loss:.4f}")

            # Logging to MLflow
            run_info = self.log_to_mlflow(
                self.model,
                X_test_encoded,
                y_pred,
                y_test,
                loss,
                train_df,
                test_df,
            )

            # Saving run info to JSON
            self.save_run_info(run_info, self.run_info_path)

            logger.success("Model evaluation pipeline completed successfully.")

        except Exception as e:
            logger.error("Model evaluation pipeline failed.")
            raise


def main() -> None:
    """
    Runs the model evaluation pipeline.
    """
    logger.info("Launching main() for model evaluation...")
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
    logger.success("Model evaluation finished.")


if __name__ == "__main__":
    main()
