import joblib
import pytest
import mlflow
import dagshub

import pandas as pd

from pathlib import Path
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error

from taxi_demand_prediction.utils.common import read_run_info

set_config(transform_output="pandas")

# Paths
CURRENT_PATH = Path(__file__)
ROOT_PATH = CURRENT_PATH.parent.parent
TRAIN_DATA_PATH = ROOT_PATH / "data" / "processed" / "train.csv"
TEST_DATA_PATH = ROOT_PATH / "data" / "processed" / "test.csv"
ENCODER_PATH = ROOT_PATH / "models" / "encoder.joblib"

# Threshold
THRESHOLD: float = 0.1

# DagsHub and MLFlow
dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="taxi-demand-prediction",
    mlflow=True,
)
URI = "https://dagshub.com/SushrutGaikwad/taxi-demand-prediction.mlflow"
mlflow.set_tracking_uri(URI)

# Loading the encoder
encoder = joblib.load(ENCODER_PATH)

# Getting the model path
model_path = read_run_info("run_information.json")["model_uri"]

# Loading the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# Building the model pipeline
model_pipeline = Pipeline(
    steps=[
        ("encoder", encoder),
        ("regressor", model),
    ]
)


# Test function
@pytest.mark.parametrize(
    argnames="data_path, threshold",
    argvalues=[
        (TRAIN_DATA_PATH, THRESHOLD),
        (TEST_DATA_PATH, THRESHOLD),
    ],
)
def test_performance(data_path: Path, threshold: float):
    # Loading the data
    data = pd.read_csv(
        data_path,
        parse_dates=["tpep_pickup_datetime"],
    ).set_index("tpep_pickup_datetime")

    # Making X and y
    X = data.drop(columns=["total_pickups"])
    y = data["total_pickups"]

    # Doing predictions
    y_pred = model_pipeline.predict(X)

    # Calculating the loss
    loss = mean_absolute_percentage_error(y, y_pred)

    # Checking the performance
    msg = f"Model failed to pass the MAPE performance threshold of {threshold}."
    assert loss <= threshold, msg
