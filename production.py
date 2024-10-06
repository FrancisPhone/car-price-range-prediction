import mlflow
import pickle
import os

tracking_uri = "https://mlflow.ml.brain.cs.ait.ac.th/"
mlflow.set_tracking_uri(tracking_uri)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
os.environ["LOGNAME"] = "st124973"
stage = 'Production'
model_name = 'st124973-a3-model'

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(name=model_name, version=1, stage=stage)

model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
with open('st124973-a3-model', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)