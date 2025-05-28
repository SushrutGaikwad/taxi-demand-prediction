import mlflow
import dagshub

from mlflow import MlflowClient

# DagsHub and MLFlow
dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="taxi-demand-prediction",
    mlflow=True,
)
URI = "https://dagshub.com/SushrutGaikwad/taxi-demand-prediction.mlflow"
mlflow.set_tracking_uri(URI)

# Getting model name
REGISTERED_MODEL_NAME = "taxi_demand_prediction_model"
CURRENT_STAGE = "Staging"

# Promotion stage
PROMOTION_STAGE = "Production"

# Getting the latest version of the model from the staging stage
client = MlflowClient()
latest_versions = client.get_latest_versions(
    name=REGISTERED_MODEL_NAME,
    stages=[CURRENT_STAGE],
)
latest_version_of_model_in_staging = latest_versions[0].version

production_model = client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=latest_version_of_model_in_staging,
    stage=PROMOTION_STAGE,
    archive_existing_versions=True,
)

production_model_version = production_model.version
new_stage = production_model.current_stage

print(f"Model is moved to the {new_stage} stage, with version {production_model_version}.")
