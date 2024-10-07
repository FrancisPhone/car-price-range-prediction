import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
os.environ["LOGNAME"] = os.getenv('LOGNAME')
stage = os.getenv('stage')
model_name = os.getenv('APP_MODEL_NAME')

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(name=model_name, version=1, stage=stage)

model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
if not os.path.exists(model_name):
    mlflow.sklearn.save_model(
        model,
        model_name,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )