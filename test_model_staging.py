import numpy as np
import mlflow
import pickle
import pytest
import os

tracking_uri = "https://mlflow.ml.brain.cs.ait.ac.th/"
mlflow.set_tracking_uri(tracking_uri)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
os.environ["LOGNAME"] = "st124973"
stage = 'Staging'
model_name = 'st124973-a3-model'


client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(name=model_name, version=1, stage=stage)


def test_load_model():
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
    with open('lgr_scaler.pickle', 'rb') as handle:
        scaler = pickle.load(handle)
    return model, scaler


@pytest.mark.depends(on=['test_load_model'])
def test_model_input():
    model, scaler = test_load_model()
    sample = np.array([[2018, 100]])
    sample = scaler.transform(sample)
    intercept = np.ones((sample.shape[0], 1))
    sample = np.concatenate((intercept, sample), axis=1)
    pred = model.predict(sample)
    assert sample.shape == (1, 3)


@pytest.mark.depends(on=['test_model_input'])
def test_model_output():
    model, scaler = test_load_model()
    sample = np.array([[2018, 100]])
    sample = scaler.transform(sample)
    intercept = np.ones((sample.shape[0], 1))
    sample = np.concatenate((intercept, sample), axis=1)
    pred = model.predict(sample)
    assert pred.shape == (1, ), f"{pred.shape=}"