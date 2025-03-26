"""
dataset.py

This module provides a TaxiDataProcessor class that uses an orchestrator pattern
to read, process, and save taxi data using Dask.
"""

from pathlib import Path
from typing import List, Optional, Union

import dask.dataframe as dd
import pandas as pd
from loguru import logger
from tqdm import tqdm

from taxi_demand_prediction.config import (
    # Project paths
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    # Outlier removal constants
    MIN_LATITUDE,
    MAX_LATITUDE,
    MIN_LONGITUDE,
    MAX_LONGITUDE,
    MIN_FARE_AMOUNT,
    MAX_FARE_AMOUNT,
    MIN_TRIP_DISTANCE,
    MAX_TRIP_DISTANCE,
    # Data reading config
    INPUT_FILES,
    READ_COLUMNS,
    PARSE_DATES,
    COLUMNS_TO_DROP,
)


class TaxiDataProcessor:
    """
    A class to handle data ingestion and outlier removal using Dask.

    This class implements multiple methods for each stage of the data pipeline:
      1) read_data
      2) combine_data
      3) remove_outliers
      4) drop_unused_columns
      5) compute_ddf
      6) run_pipeline (the orchestrator)
    """

    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        min_fare: float,
        max_fare: float,
        min_dist: float,
        max_dist: float,
    ) -> None:
        """Initializes the `TaxiDataProcessor` object with numeric bounds to be
        used in outlier removal.

        Args:
            min_lat (float): Minimum latitude bound.
            max_lat (float): Maximum latitude bound.
            min_lon (float): Minimum longitude bound.
            max_lon (float): Maximum longitude bound.
            min_fare (float): Minimum fare bound.
            max_fare (float): Maximum fare bound.
            min_dist (float): Minimum distance bound.
            max_dist (float): Maximum distance bound.
        """
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_fare = min_fare
        self.max_fare = max_fare
        self.min_dist = min_dist
        self.max_dist = max_dist

    def read_data(
        self,
        input_files: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        parse_dates: Optional[List[str]] = None,
    ) -> List[dd.DataFrame]:
        """Reads multiple CSV files and returns a list of Dask DataFrames.

        Args:
            input_files (Optional[List[str]], optional): List of filenames
                relative to RAW_DATA_DIR. Defaults to None.
            columns (Optional[List[str]], optional): List of columns to select.
                Defaults to None.
            parse_dates (Optional[List[str]], optional): List of datetime columns
                to parse. Defaults to None.

        Raises:
            FileNotFoundError: If any of the files do not exist.
            RuntimeError: If data ingestion fails.

        Returns:
            List[dd.DataFrame]: List of Dask DataFrames.
        """
        logger.info("Reading the data...")
        if input_files is None:
            input_files = INPUT_FILES
        if columns is None:
            columns = READ_COLUMNS
        if parse_dates is None:
            parse_dates = PARSE_DATES

        # Check if all files exist before attempting to read
        missing_files = [f for f in input_files if not (RAW_DATA_DIR / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"The following input files are missing: {missing_files}.")

        dfs: List[dd.DataFrame] = []
        try:
            for fname in tqdm(input_files, desc="Reading CSV files"):
                path: Path = RAW_DATA_DIR / fname
                logger.info(f"Reading data from {path}...")
                df: dd.DataFrame = dd.read_csv(
                    path,
                    usecols=columns,
                    parse_dates=parse_dates,
                )
                dfs.append(df)
                logger.info(f"Successfully read {path}.")
            logger.info("Successfully read all the data.")
        except Exception as e:
            logger.exception(f"Error reading {path}: {e}")
            raise RuntimeError(f"Data ingestion failed while reading {fname}.") from e
        return dfs

    def combine_data(self, dfs: List[dd.DataFrame]) -> Optional[dd.DataFrame]:
        """Concatenates a list of Dask DataFrames into a single DataFrame.

        Args:
            dfs (List[dd.DataFrame]): List of Dask DataFrames to combine.

        Raises:
            ValueError: If input list is empty.
            RuntimeError: If concatenation fails.

        Returns:
            Optional[dd.DataFrame]: Combined Dask DataFrame.
        """
        logger.info("Combining data...")
        if not dfs:
            logger.error(f"`combine_data` failed: Empty input list.")
            raise ValueError("No DataFrames provided for combination.")

        try:
            df_final: dd.DataFrame = dd.concat(dfs, axis=0)
            logger.info("Successfully combined all DataFrames.")
            return df_final
        except Exception as e:
            logger.exception("`combine_data` failed during concatenation.")
            raise RuntimeError("Failed to concatenate DataFrames.") from e

    def remove_outliers(self, df: dd.DataFrame) -> dd.DataFrame:
        """Removes spatial and numerical outliers from the Dask DataFrame.

        Args:
            df (dd.DataFrame): Input Dask DataFrame.

        Returns:
            dd.DataFrame: Dask DataFrame with outliers removed.
        """
        logger.info("Removing outliers...")
        try:
            df = df.loc[
                (df["pickup_latitude"].between(self.min_lat, self.max_lat, inclusive="both"))
                & (df["pickup_longitude"].between(self.min_lon, self.max_lon, inclusive="both"))
                & (df["dropoff_latitude"].between(self.min_lat, self.max_lat, inclusive="both"))
                & (df["dropoff_longitude"].between(self.min_lon, self.max_lon, inclusive="both"))
                & (df["fare_amount"].between(self.min_fare, self.max_fare, inclusive="both"))
                & (df["trip_distance"].between(self.min_dist, self.max_dist, inclusive="both"))
            ]
            logger.info("Outliers removed successfully.")
            return df
        except Exception as e:
            logger.exception(f"Error removing outliers: {e}.")
            return df

    def drop_unused_columns(
        self, df: dd.DataFrame, cols_to_drop: Optional[List[str]] = None
    ) -> dd.DataFrame:
        """Drops unused columns from the DataFrame.

        Args:
            df (dd.DataFrame): Input Dast DataFrame.
            cols_to_drop (Optional[List[str]], optional): List of columns to drop.
                Defaults to None.

        Raises:
            ValueError: If any specified column is still present after dropping.

        Returns:
            dd.DataFrame: Dask DataFrame with specified columns removed.
        """
        if cols_to_drop is None:
            cols_to_drop = COLUMNS_TO_DROP

        try:
            logger.info(f"Dropping columns: {cols_to_drop}...")
            df = df.drop(cols_to_drop, axis=1)

            # Make sure that all `cols_to_drop` are actually dropped
            remaining_cols: List[str] = [col for col in cols_to_drop if col in df.columns]
            if remaining_cols:
                raise ValueError(f"Failed to drop columns: {remaining_cols}.")

            logger.info("All specified columns dropped successfully.")
            return df
        except Exception as e:
            logger.exception("Failed to drop unused columns.")
            raise

    def compute_ddf(self, df: dd.DataFrame) -> pd.DataFrame:
        """Materializes the Dask DataFrame as a Pandas DataFrame.

        Args:
            df (dd.DataFrame): Input Dask DataFrame.

        Returns:
            pd.DataFrame: Pandas DataFrame.
        """
        try:
            logger.info("Computing the final Dask DataFrame...")
            return df.compute()
        except Exception as e:
            logger.exception(f"Error computing DataFrame: {e}.")
            raise

    def run_pipeline(
        self,
        input_files: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if output_path is None:
            output_path = INTERIM_DATA_DIR / "df_without_outliers.csv"
        else:
            output_path = Path(output_path)

        logger.info("Starting the orchestrator pipeline...")

        try:
            dfs = self.read_data(input_files=input_files)
            df = self.combine_data(dfs)
            df = self.remove_outliers(df)
            df = self.drop_unused_columns(df)
            df = self.compute_ddf(df)

            logger.info(f"Saving the final DataFrame to {output_path}...")
            df.to_csv(output_path, index=False)
            logger.info("DataFrame saved successfully.")
        except Exception as e:
            logger.error("Pipeline failed.")
            raise


def main() -> None:
    """
    Main entry point for running data ingestion pipeline.
    """
    logger.info("Launching `main()` for data ingestion...")

    data_processor = TaxiDataProcessor(
        min_lat=MIN_LATITUDE,
        max_lat=MAX_LATITUDE,
        min_lon=MIN_LONGITUDE,
        max_lon=MAX_LONGITUDE,
        min_fare=MIN_FARE_AMOUNT,
        max_fare=MAX_FARE_AMOUNT,
        min_dist=MIN_TRIP_DISTANCE,
        max_dist=MAX_TRIP_DISTANCE,
    )

    data_processor.run_pipeline()
    logger.success("Dataset ingestion and processing completed.")


if __name__ == "__main__":
    main()
