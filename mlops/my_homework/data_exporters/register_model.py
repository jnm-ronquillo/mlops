if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import os
import pickle

EXPERIMENT_NAME = "yellow-models"

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
dest_path="output"

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    dv, lr = data
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)
    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))

    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, artifact_path="models")
        mlflow.log_artifact("output/dv.pkl", artifact_path="preprocessor")
        

