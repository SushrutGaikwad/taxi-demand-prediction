"""
feature_engineering.py

A module for feature engineering on the resampled data. This includes:
  - reading the final resampled data (resampled_data.csv)
  - extracting datetime features (day_of_week, month)
  - creating lag features
  - splitting into train/test sets
  - saving final outputs
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from taxi_demand_prediction.config import (
    RESAMPLED_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_MONTHS,
    TEST_MONTHS,
    PARSE_DATES,
    DATETIME_COL_NAME,
    MAX_LAGS,
)


class FeatureEngineer:
    """
    A class to handle final feature engineering steps on the resampled data.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        train_data_path: Optional[Path] = None,
        test_data_path: Optional[Path] = None,
    ) -> None:
        """
        Initializes `FeatureEngineer` object with paths to the main data, and
        training and test data.

        Args:
            data_path (Optional[Path], optional): Path to the resampled data.
                Defaults to None.
            train_data_path (Optional[Path], optional): Path to the training data.
                Defaults to None.
            test_data_path (Optional[Path], optional): Path to the test data.
                Defaults to None.
        """
        self.data_path: Path = data_path if data_path else RESAMPLED_DATA_PATH
        self.train_data_path: Path = train_data_path if train_data_path else TRAIN_DATA_PATH
        self.test_data_path: Path = test_data_path if test_data_path else TEST_DATA_PATH

    def read_data(self) -> pd.DataFrame:
        """
        Reads the resampled CSV data.

        Raises:
            FileNotFoundError: If the resampled CSV file is not found.
            RuntimeError: If an error occurs while reading the data.

        Returns:
            pd.DataFrame: Resampled CSV data.
        """
        logger.info(f"Reading the resampled data from {self.data_path}...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Resampled data file not found at {self.data_path}.")
        try:
            df = pd.read_csv(self.data_path, parse_dates=PARSE_DATES)
            logger.info("Resampled data read successfully.")
            return df
        except Exception as e:
            logger.exception("Error reading resampled data.")
            raise RuntimeError("Failed to read resampled data.") from e

    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts datetime features from the data, and sets the datetime
        column as index.

        Args:
            df (pd.DataFrame): Resampled data.

        Raises:
            RuntimeError: If an error occurs while extracting datetime features.

        Returns:
            pd.DataFrame: Data after extracting datetime features.
        """
        logger.info("Extracting datetime features (`day_of_week`, `month`)...")
        try:
            df["day_of_week"] = df[DATETIME_COL_NAME].dt.day_of_week
            df["month"] = df[DATETIME_COL_NAME].dt.month
            logger.info("Datetime features extracted successfully.")

            df.set_index(DATETIME_COL_NAME, inplace=True, drop=True)
            logger.info("Datetime column set as index successfully.")
            return df
        except Exception as e:
            logger.exception("Error extracting datetime features.")
            raise RuntimeError("Failed to extract datetime features.") from e

    def create_lag_features(
        self,
        df: pd.DataFrame,
        max_lags: int = MAX_LAGS,
    ) -> pd.DataFrame:
        """
        Creates lag features of the "total_pickups" column for each region group
        and drops missing values generated due to lags.

        Args:
            df (pd.DataFrame): The input DataFrame, indexed by datetime with
                "region" and "total_pickups" columns.
            max_lags (int, optional): Number of lag features to generate.
                Defaults to MAX_LAGS.

        Raises:
            RuntimeError: If an error occurs during the generation.

        Returns:
            pd.DataFrame: DataFrame with added lag features.
        """
        logger.info(f"Generating lag features up to {max_lags} periods...")
        try:
            region_grp = df.groupby("region")["total_pickups"]

            # Each shifted series gets a unique name, e.g., lag_1, lag_2...
            lag_series_list = []
            for i in range(1, max_lags + 1):
                col = region_grp.shift(i)
                col.name = f"lag_{i}"  # Assign a distinct name
                lag_series_list.append(col)

            # Concatenate the original df plus each named lag series
            df_lagged = pd.concat([df] + lag_series_list, axis=1)

            # Drop rows with NaNs introduced by shifting
            df_lagged.dropna(inplace=True)
            logger.info("Dropped missing values after generating lags.")

            # Done: no need for a separate rename step, since each col is already named
            logger.info("Lag features created successfully.")
            return df_lagged

        except Exception as e:
            logger.exception("Error creating lag features.")
            raise RuntimeError("Failed to create lag features.") from e

    def split_data(
        self,
        df: pd.DataFrame,
        train_months: Optional[list] = None,
        test_months: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and test sets based on the "month" column.

        Args:
            df (pd.DataFrame): Input data, which must have a "month" column.
            train_months (Optional[list], optional): List of integer months in the training set.
                Defaults to None.
            test_months (Optional[list], optional): List of integer months in the test set.
                Defaults to None.

        Raises:
            RuntimeError: If there is no "month" column in the data.
            RuntimeError: If an error occurs during splitting.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training set and the test set.
        """
        logger.info("Splitting the data into training and test sets...")
        if "month" not in df.columns:
            raise RuntimeError("No 'month' column found for data splitting.")

        if train_months is None:
            train_months = TRAIN_MONTHS
        if test_months is None:
            test_months = TEST_MONTHS

        try:
            train = df.loc[df["month"].isin(train_months)]
            test = df.loc[df["month"].isin(test_months)]
            logger.info(f"Training set months: {train_months}, test set months: {test_months}.")

            # Remove the "month" column
            train.drop(columns=["month"], inplace=True)
            test.drop(columns=["month"], inplace=True)

            logger.info("Data split successfully.")
            return train, test
        except Exception as e:
            logger.exception("Error splitting data.")
            raise RuntimeError("Failed to split train/test sets.") from e

    def save_dataframes(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Saves the training and test sets to a CSV.

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            test_df (pd.DataFrame): Test DataFrame.

        Raises:
            RuntimeError: If an error occurs while saving the training set.
            RuntimeError: If an error occurs while saving the test set.
        """
        logger.info(f"Saving training data to {self.train_data_path}...")
        try:
            train_df.to_csv(self.train_data_path, index=True)
            logger.info("Training data saved successfully.")
        except Exception as e:
            logger.exception("Failed to save training data.")
            raise RuntimeError(f"Error saving training data to {self.train_data_path}.") from e

        logger.info(f"Saving test data to {self.test_data_path}...")
        try:
            test_df.to_csv(self.test_data_path, index=True)
            logger.info("Test data saved successfully.")
        except Exception as e:
            logger.exception("Failed to save test data.")
            raise RuntimeError(f"Error saving test data to {self.test_data_path}") from e

    def run_feature_engineering(self) -> None:
        logger.info("Starting feature engineering pipeline...")
        try:
            # 1) Read the resampled data
            df = self.read_data()

            # 2) Extract datetime features (day_of_week, month)
            df = self.extract_datetime_features(df)

            # 3) Create lag features
            df = self.create_lag_features(df)

            # 4) Split into train/test sets
            train_df, test_df = self.split_data(df)

            # 5) Save the CSVs
            self.save_dataframes(train_df, test_df)

            logger.info("Feature engineering pipeline completed successfully.")
        except Exception as e:
            logger.error("Feature engineering pipeline failed.")
            raise


def main() -> None:
    logger.info("Launching `main()` for feature engineering...")

    engineer = FeatureEngineer()
    engineer.run_feature_engineering()

    logger.success("Feature engineering process completed.")


if __name__ == "__main__":
    main()
