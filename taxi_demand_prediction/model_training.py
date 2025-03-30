"""
model_training.py

A module that trains a regression model (LinearRegression) on the processed train data.
It also applies a ColumnTransformer (OneHotEncoder) to categorical columns, and saves
both the encoder and the trained model to disk.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import joblib
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from taxi_demand_prediction.config import (
    TRAIN_DATA_PATH,
    ENCODER_PATH,
    MODEL_PATH,
    PARSE_DATES,
    DATETIME_COL_NAME,
    DROP_COLS,
    COLS_TO_ENCODE,
)

# Enable pandas output for transformations
set_config(transform_output="pandas")


class ModelTrainer:
    """
    A class to handle the end-to-end model training process, including:
        - Reading training data
        - Creating and saving an encoder (ColumnTransformer)
        - Encoding features
        - Training a LinearRegression model
        - Saving the final model
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        encoder_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        """
        Initializes a `ModelTrainer` object with configurable file paths.

        Args:
            data_path (Optional[Path], optional): Path to the training CSV.
                Defaults to None.
            encoder_path (Optional[Path], optional): Path to save the fitted encoder.
                Defaults to None.
            model_path (Optional[Path], optional): Path to save the trained model.
                Defaults to None.
        """
        self.data_path = data_path if data_path else TRAIN_DATA_PATH
        self.encoder_path = encoder_path if encoder_path else ENCODER_PATH
        self.model_path = model_path if model_path else MODEL_PATH
        self.encoder: Optional[ColumnTransformer] = None
        self.model: Optional[LinearRegression] = None

    def read_training_data(self) -> pd.DataFrame:
        """
        Reads the training data from CSV.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            RuntimeError: If an error occurs while reading the CSV file.

        Returns:
            pd.DataFrame: Training data.
        """
        logger.info(f"Reading training data from {self.data_path}...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data file not found at {self.data_path}.")

        try:
            df = pd.read_csv(self.data_path, parse_dates=PARSE_DATES)
            logger.info("Training data read successfully.")
            return df
        except Exception as e:
            logger.exception("Error reading training data.")
            raise RuntimeError("Failed to read training CSV.") from e

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sets the datetime column as index, then separates features from the target.

        Args:
            df (pd.DataFrame): The input training data with the "total_pickups" column.

        Raises:
            RuntimeError: If an error occurs while preparing the training data.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: The feature df and target ds.
        """
        logger.info("Preparing training data by setting index and splitting features/target...")
        try:
            # Set datetime as index
            df.set_index(DATETIME_COL_NAME, inplace=True, drop=True)

            # X_train and y_train
            X_train = df.drop(columns=DROP_COLS)
            y_train = df[DROP_COLS[0]]
            logger.info("Training data prepared successfully.")
            return X_train, y_train
        except Exception as e:
            logger.exception("Error preparing training data.")
            raise RuntimeError("Failed to prepare training data (features/target).") from e

    def build_transformer(self) -> ColumnTransformer:
        """
        Creates a `ColumnTransformer` for encoding categorical variables.

        Raises:
            RuntimeError: If an error occurs during building a transformer.

        Returns:
            ColumnTransformer: The `ColumnTransformer` to encode features.
        """
        logger.info("Building the `ColumnTransformer` (`OneHotEncoder`)...")
        try:
            # Encoding 'region' and 'day_of_week', then passthrough the rest
            encoder = ColumnTransformer(
                [
                    (
                        "ohe",
                        OneHotEncoder(drop="first", sparse_output=False),
                        COLS_TO_ENCODE,
                    )
                ],
                remainder="passthrough",
                n_jobs=-1,
                force_int_remainder_cols=False,
            )
            logger.info("`ColumnTransformer` built successfully.")
            return encoder
        except Exception as e:
            logger.exception("Error building `ColumnTransformer`.")
            raise RuntimeError("Failed to build encoder for training.") from e

    def save_encoder(self, encoder: ColumnTransformer) -> None:
        """
        Saves the fitted encoder to disk.

        Args:
            encoder (ColumnTransformer): The fitted `ColumnTransformer`.

        Raises:
            RuntimeError: If an error occurs during saving.
        """
        logger.info(f"Saving encoder to {self.encoder_path}...")
        try:
            joblib.dump(encoder, self.encoder_path)
            logger.info("Encoder saved successfully.")
        except Exception as e:
            logger.exception("Error saving encoder.")
            raise RuntimeError(f"Failed to save encoder at {self.encoder_path}") from e

    def encode_data(
        self,
        encoder: ColumnTransformer,
        X_train: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Applies the fitted transformer to the training features.

        Args:
            encoder (ColumnTransformer): The fitted `ColumnTransformer`.
            X_train (pd.DataFrame): The raw training features.

        Raises:
            RuntimeError: If an error occurs during encoding.

        Returns:
            pd.DataFrame: Encoded training features.
        """
        logger.info("Encoding the training data using the fitted encoder...")
        try:
            X_train_encoded = encoder.transform(X_train)
            logger.info("Data encoded successfully.")
            return X_train_encoded
        except Exception as e:
            logger.exception("Error encoding training data.")
            raise RuntimeError("Failed to transform the training features.") from e

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> LinearRegression:
        """
        Fits a `LinearRegression` model on the encoded training data.

        Args:
            X (pd.DataFrame): Encoded training features.
            y (pd.Series): Training target.

        Raises:
            RuntimeError: If an error occurs during training.

        Returns:
            LinearRegression: Fitted `LinearRegression` model.
        """
        logger.info("Training `LinearRegression` model...")
        try:
            reg = LinearRegression()
            reg.fit(X, y)
            logger.info("Model trained successfully.")
            return reg
        except Exception as e:
            logger.exception("Error training `LinearRegression` model.")
            raise RuntimeError("Failed to train the regression model.") from e

    def save_model(self, model: LinearRegression) -> None:
        """
        Saves the trained model to disk.

        Args:
            model (LinearRegression): The fitted `LinearRegression` model.

        Raises:
            RuntimeError: If an error occurs during saving.
        """
        logger.info(f"Saving model to {self.model_path}...")
        try:
            joblib.dump(model, self.model_path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.exception("Error saving model.")
            raise RuntimeError(f"Failed to save model at {self.model_path}") from e

    def run_training(self) -> None:
        """
        Orchestrates the entire training pipeline:
            1. Read train data,
            2. Prepare features/target,
            3. Build and fit an encoder,
            4. Save encoder,
            5. Encode the training data,
            6. Train a `LinearRegression` model,
            7. Save the model.
        """
        logger.info("Starting the model training pipeline...")
        try:
            # 1) Read the CSV
            df_train = self.read_training_data()

            # 2) Prepare X, y
            X_train, y_train = self.prepare_data(df_train)

            # 3) Build and fit encoder
            encoder = self.build_transformer()
            encoder.fit(X_train)

            # 4) Save fitted encoder
            self.save_encoder(encoder)

            # 5) Encode training data
            X_train_encoded = self.encode_data(encoder, X_train)

            # 6) Train LinearRegression
            model = self.train_model(X_train_encoded, y_train)

            # 7) Save trained model
            self.save_model(model)

            logger.info("Model training pipeline completed successfully.")
        except Exception as e:
            logger.error("Model training pipeline failed.")
            raise


def main() -> None:
    """
    Main entry point to run the model training pipeline.
    """
    logger.info("Launching model training `main()`...")
    trainer = ModelTrainer()
    trainer.run_training()
    logger.success("Model training process completed.")


if __name__ == "__main__":
    main()
