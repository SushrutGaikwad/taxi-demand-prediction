import mlflow
import dagshub

from pathlib import Path

from taxi_demand_prediction.utils.common import read_run_info

# DagsHub and MLFlow
dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="taxi-demand-prediction",
    mlflow=True,
)
URI = "https://dagshub.com/SushrutGaikwad/taxi-demand-prediction.mlflow"
mlflow.set_tracking_uri(URI)

# Getting the model path
model_path = read_run_info("run_information.json")["model_uri"]

# Loading the latest model from model registry
model = mlflow.sklearn.load_model(model_path)


# Testing
def test_load_model_from_registry():
    assert model is not None, "Failed to load model from the registry."
