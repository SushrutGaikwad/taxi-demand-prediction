"""
extract_features.py

A module that handles feature extraction on the intermediate dataset (df_without_outliers.csv).
It trains a scaler and a MiniBatchKMeans model in partial mode, resamples the data, and saves
the final dataset.
"""

from pathlib import Path
from typing import Optional, Union, List

import joblib
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from yaml import safe_load
from pandas.io.parsers import TextFileReader

from taxi_demand_prediction.config import (
    DF_WITHOUT_OUTLIERS_CSV,
    SCALER_OBJ_PATH,
    KMEANS_OBJ_PATH,
    RESAMPLED_DATA_PATH,
    DEFAULT_CHUNK_SIZE,
    CLUSTER_COLS,
    LOCATION_SUBSET,
    EPSILON_VAL,
    PARAMS_FILE,
    PARSE_DATES,
    DATETIME_COL_NAME,
)


class FeatureExtractor:
    """
    A class to handle feature extraction, including:
        - Reading chunked data
        - Training a scaler
        - Training a MiniBatchKMeans model
        - Predicting clusters
        - Resampling data
        - Saving final outputs
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        params_path: Optional[Path] = None,
    ) -> None:
        """Initializes the `FeatureExtractor` object with paths to the data and
        params file.

        Args:
            data_path (Optional[Path], optional): Path to the data without outliers.
                Defaults to None.
            params_path (Optional[Path], optional): Path to the params.yaml file.
                Defaults to None.
        """
        self.data_path = data_path if data_path else DF_WITHOUT_OUTLIERS_CSV
        self.params_path = params_path if params_path else PARAMS_FILE

    def read_params(self) -> dict:
        """Reads pipeline parameters from a YAML file.

        Raises:
            FileNotFoundError: If the file is missing.
            RuntimeError: If an error occurs during reading.

        Returns:
            dict: Pipeline parameters.
        """
        logger.info(f"Reading parameters from {self.params_path}...")
        if not self.params_path.exists():
            raise FileNotFoundError(f"Parameter file not found at {self.params_path}.")

        try:
            with open(self.params_path, "r") as file:
                params = safe_load(file)
            logger.info("Parameters read successfully.")
            return params
        except Exception as e:
            logger.exception("Failed to read parameters file.")
            raise RuntimeError("Error reading parameters YAML.") from e

    def read_cluster_input(
        self,
        chunksize: int = DEFAULT_CHUNK_SIZE,
        usecols: Optional[List[str]] = None,
    ) -> TextFileReader:
        """Reads a CSV file in chunks for cluster training.

        Args:
            chunksize (int, optional): Number of rows per chunk in partial reading.
                Defaults to DEFAULT_CHUNK_SIZE.
            usecols (Optional[List[str]], optional): Columns to use for partial reading.
                Defaults to None.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            RuntimeError: If an error occurs during partial reading.

        Returns:
            TextFileReader: An iterator of DataFrames.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}.")

        if usecols is None:
            usecols = CLUSTER_COLS

        logger.info(f"Reading data in chunks from {self.data_path}...")
        try:
            df_reader = pd.read_csv(
                self.data_path,
                chunksize=chunksize,
                usecols=usecols,
            )
            logger.info("Data read successfully.")
            return df_reader
        except Exception as e:
            logger.exception("Failed to read data in chunks.")
            raise RuntimeError(f"Error reading chunked data from {self.data_path}") from e

    def save_model(self, model: BaseEstimator, save_path: Path) -> None:
        """Saves a model (e.g., `StandardScaler` or `MiniBatchKMeans`) to disk.

        Args:
            model (BaseEstimator): The model to save (e.g., `StandardScaler` or
                `MiniBatchKMeans`).
            save_path (Path): Path to save the model.

        Raises:
            RuntimeError: If an error occurs during saving.
        """
        logger.info(f"Saving model to {save_path}...")
        try:
            joblib.dump(model, save_path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.exception("Failed to save model.")
            raise RuntimeError(f"Error saving model to {save_path}.") from e

    def train_scaler(
        self,
        chunks: TextFileReader,
        scaler_obj_path: Path = SCALER_OBJ_PATH,
    ) -> StandardScaler:
        """
        Trains a StandardScaler in partial mode using chunked data. Saves the
        trained scaler to disk.

        Args:
            chunks (TextFileReader): An iterator of DataFrames (e.g., from
                `read_cluster_input`).
            scaler_obj_path (Path, optional): Path to save the trained scaler.
                Defaults to SCALER_OBJ_PATH.

        Raises:
            RuntimeError: If an error occurs during training or saving.

        Returns:
            StandardScaler: The trained `StandardScaler` instance.
        """
        logger.info("Training a `StandardScaler` in partial mode...")
        scaler = StandardScaler()
        try:
            for chunk in tqdm(chunks, desc="Fitting scaler"):
                scaler.partial_fit(chunk)
            self.save_model(scaler, scaler_obj_path)
            return scaler
        except Exception as e:
            logger.exception("Failed to train scaler.")
            raise RuntimeError("Error during partial fit of `StandardScaler`.") from e

    def train_kmeans(
        self,
        chunks: TextFileReader,
        scaler: StandardScaler,
        kmeans_obj_path: Path = KMEANS_OBJ_PATH,
        mini_batch_params: dict = None,
    ) -> MiniBatchKMeans:
        """
        Trains a `MiniBatchKMeans` model in partial mode using chunked data.
        Saves the trained model to disk.

        Args:
            chunks (TextFileReader): An iterator of DataFrames for partial fitting.
            scaler (StandardScaler): A previously fitted `StandardScaler` to
                transform each chunk.
            kmeans_obj_path (Path, optional): Path to save the trained
                `MiniBatchKMeans`. Defaults to KMEANS_OBJ_PATH.
            mini_batch_params (dict, optional): Hyperparameters for
                `MiniBatchKMeans`. Defaults to None.

        Raises:
            RuntimeError: If an error occurs during training or saving the
                `MiniBatchKMeans` model.

        Returns:
            MiniBatchKMeans: The trained `MiniBatchKMeans` instance.
        """
        logger.info("Training `MiniBatchKMeans` in partial mode...")
        if mini_batch_params is None:
            mini_batch_params = {}

        kmeans = MiniBatchKMeans(**mini_batch_params)
        try:
            for chunk in tqdm(chunks, desc="Fitting `MiniBatchKMeans`"):
                scaled_chunk = scaler.transform(chunk)
                kmeans.partial_fit(scaled_chunk)
            self.save_model(kmeans, kmeans_obj_path)
            return kmeans
        except Exception as e:
            logger.exception("Failed to train `MiniBatchKMeans`.")
            raise RuntimeError("Error during partial fit of `MiniBatchKMeans`.") from e

    def read_full_data(self) -> pd.DataFrame:
        """
        Reads the entire dataset (not chunked) for cluster prediction and
        further processing.

        Raises:
            FileNotFoundError: If the dataset is not found.
            RuntimeError: If there is an error reading the dataset.

        Returns:
            pd.DataFrame: The entire dataset.
        """
        logger.info(f"Reading full data from {self.data_path}...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}.")

        try:
            df = pd.read_csv(self.data_path, parse_dates=PARSE_DATES)
            logger.info("Full data read successfully.")
            return df
        except Exception as e:
            logger.exception("Failed to read full data.")
            raise RuntimeError("Error reading full dataset in `read_full_data()`.") from e

    def predict_clusters(
        self, df: pd.DataFrame, scaler: StandardScaler, kmeans: MiniBatchKMeans
    ) -> pd.DataFrame:
        """
        Predicts clusters using the trained `StandardScaler` and
        `MiniBatchKMeans` model.

        Args:
            df (pd.DataFrame): Full data.
            scaler (StandardScaler): Trained `StandardScaler` instance.
            kmeans (MiniBatchKMeans): Trained `MiniBatchKMeans` instance.

        Raises:
            RuntimeError: If an error occurs during prediction.

        Returns:
            pd.DataFrame: Full data updated with the 'region' column.
        """
        logger.info("Predicting clusters on full data...")
        try:
            location_subset = df.loc[:, LOCATION_SUBSET]
            scaled_locations = scaler.transform(location_subset)
            cluster_predictions = kmeans.predict(scaled_locations)

            df["region"] = cluster_predictions
            logger.info("Cluster predictions added to data.")
            return df
        except Exception as e:
            logger.exception("Failed to predict clusters.")
            raise RuntimeError("Error during cluster prediction.") from e

    def resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops latitude and longitude columns, resamples the data into 15-minute
        intervals, replaces zeros, and computes an EWMA-based "avg_pickups"
        column.

        Args:
            df (pd.DataFrame): Full data with "region" column.

        Raises:
            RuntimeError: If an error occurs while executing this method.

        Returns:
            pd.DataFrame: Resampled data with "total_pickups" and "avg_pickups".
        """
        logger.info("Resampling data into 15-minute intervals...")
        try:
            # Drop lat/lon columns
            df = df.drop(columns=LOCATION_SUBSET, errors="ignore")
            logger.info("Latitude and longitude columns dropped.")

            # Set datetime as index
            df.set_index(DATETIME_COL_NAME, inplace=True, drop=True)

            # Group by region and resample
            region_grp = df.groupby("region")
            resampled_data = region_grp["region"].resample("15min").count()
            resampled_data.name = "total_pickups"
            logger.info("Data resampled at 15-minute intervals.")

            # Convert to DataFrame
            resampled_data = resampled_data.reset_index(level=0)

            # Replace zeros
            resampled_data.replace({"total_pickups": {0: EPSILON_VAL}}, inplace=True)

            # Read EWMA parameters from YAML
            params = self.read_params()
            ewma_params = params["extract_features"]["ewma"]
            logger.info(f"EWMA parameters: {ewma_params}.")

            # Calculate avg pickups using EWMA
            resampled_data["avg_pickups"] = (
                resampled_data.groupby("region")["total_pickups"]
                .ewm(**ewma_params)
                .mean()
                .round()
                .values
            )
            logger.info("Average pickups calculated using EWMA.")
            return resampled_data
        except Exception as e:
            logger.exception("Failed to resample data.")
            raise RuntimeError("Error during data resampling and transformation.") from e

    def save_final_data(
        self,
        df: pd.DataFrame,
        save_path: Path = RESAMPLED_DATA_PATH,
    ) -> None:
        logger.info(f"Saving resampled data to {save_path}...")
        try:
            df.to_csv(save_path, index=True)
            logger.info("Resampled data saved successfully.")
        except Exception as e:
            logger.exception("Failed to save resampled data.")
            raise RuntimeError(f"Error writing final data to {save_path}.") from e

    def run_feature_extraction(self) -> None:
        logger.info("Starting feature extraction pipeline...")

        try:
            # 1) Train the scaler in partial mode
            df_reader = self.read_cluster_input()
            scaler = self.train_scaler(df_reader)

            # 2) Train the MiniBatchKMeans model in partial mode
            params = self.read_params()
            kmeans_params = params["extract_features"]["mini_batch_kmeans"]
            logger.info(f"`MiniBatchKMeans` parameters: {kmeans_params}.")

            df_reader = self.read_cluster_input()
            kmeans = self.train_kmeans(
                df_reader,
                scaler,
                mini_batch_params=kmeans_params,
            )

            # 3) Predict clusters on the full dataset
            df = self.read_full_data()
            df_with_clusters = self.predict_clusters(df, scaler, kmeans)

            # 4) Resample the data
            df = self.resample_data(df_with_clusters)

            # 5) Save final dataset
            self.save_final_data(df)

            logger.info("Feature extraction pipeline completed successfully.")
        except Exception as e:
            logger.error("Feature extraction pipeline failed.")
            raise


def main() -> None:
    logger.info("Launching `main()` for feature extraction...")

    extractor = FeatureExtractor()
    extractor.run_feature_extraction()

    logger.success("Feature extraction completed successfully.")


if __name__ == "__main__":
    main()
