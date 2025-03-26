"""
config.py

This module defines paths, logging setup, and all constants used by the project.
"""

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from loguru import logger

# ------------------------------------------------------------------------------
# Load environment variables from a .env file if it exists
# ------------------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------------------
# We assume this file is at: <proj_root>/taxi_demand_prediction/config.py
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ------------------------------------------------------------------------------
# If tqdm is installed, configure loguru with tqdm.write so the logs show nicely
# alongside any progress bars.
# ------------------------------------------------------------------------------
try:
    from tqdm import tqdm

    # Remove the default loguru handler and replace it
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ------------------------------------------------------------------------------
# Outlier Constants (Latitude, Longitude, Fare, Trip Distance)
# ------------------------------------------------------------------------------
MIN_LATITUDE: float = 40.60
MAX_LATITUDE: float = 40.85
MIN_LONGITUDE: float = -74.05
MAX_LONGITUDE: float = -73.70

MIN_FARE_AMOUNT: float = 0.50
MAX_FARE_AMOUNT: float = 81.0
MIN_TRIP_DISTANCE: float = 0.25
MAX_TRIP_DISTANCE: float = 24.43

# ------------------------------------------------------------------------------
# Input File Constants
# ------------------------------------------------------------------------------
INPUT_FILES: List[str] = [
    "yellow_tripdata_2016-01.csv",
    "yellow_tripdata_2016-02.csv",
    "yellow_tripdata_2016-03.csv",
]

# ------------------------------------------------------------------------------
# Columns for Reading Data
# ------------------------------------------------------------------------------
READ_COLUMNS: List[str] = [
    "trip_distance",
    "tpep_pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "fare_amount",
]

# ------------------------------------------------------------------------------
# Columns to Parse as Datetime
# ------------------------------------------------------------------------------
PARSE_DATES: List[str] = ["tpep_pickup_datetime"]

# ------------------------------------------------------------------------------
# Columns to Drop after Processing
# ------------------------------------------------------------------------------
COLUMNS_TO_DROP: List[str] = [
    "trip_distance",
    "dropoff_longitude",
    "dropoff_latitude",
    "fare_amount",
]
