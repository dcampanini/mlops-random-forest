import time
from datetime import datetime
from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component, pipeline)
from kfp.v2.google.client import AIPlatformClient

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
import os
import subprocess


# Get your Google Cloud project ID from gcloud
if not os.getenv("IS_TESTING"):
    shell_output = subprocess.check_output(
            "gcloud config list --format 'value(core.project)' 2>/dev/null",
            shell=True, text=True).strip()
    PROJECT_ID = shell_output
    print("Project ID: ", PROJECT_ID)

@component(
    packages_to_install=[
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "fsspec==2021.10.0",  # Add fsspec = interface to access in this case GCS
        "gcsfs==2021.10.0"     # Add gcsfs for Google Cloud Storage access
    ],
)
def read_and_preprocessing(
    input_bucket: str,
    scaled_dataset: Output[Dataset],
):
    """

    Args:
        input_bucket: bucket where the dataset is stored
        scaled_dataset: Output[Dataset] is a special type of parameter called an artifact. 
                        It tells Vertex AI Pipelines that this component will output a dataset file.
                        typically as a .csv, .json, .parquet

    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    # Debugging: Print the path you're trying to access
    gs_uri = f"gs://{input_bucket}/pima-indians-diabetes.csv"
    print(f"Reading from: {gs_uri}")

    df = pd.read_csv(gs_uri)
    # normalize data
    # Separate the input features and the target variable
    X = df.iloc[:, :-1]  # All columns except the last one (features)
    y = df.iloc[:, -1]   # Last column (target variable)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Normalize the features
    X_normalized = scaler.fit_transform(X)

    # Create a new DataFrame with the normalized features and the target variable
    df_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    df_normalized['target'] = y
    
    # guardamos el archivo como CSV
    # scaled_dataset.path : is a GCS folder created by Vertex AI
    # Vertex make it available for future pipeline components, you can reuse it 
    df_normalized.to_csv(scaled_dataset.path, index=False)
    
    print ("Prerpocessing Task Completed")


@component(
    packages_to_install=[
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "joblib==0.15.1",
    ],
)
def train(
    scaled_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Trains a model on tabular data."""
    import pandas as pd
    import joblib
    import os
    from sklearn.metrics import (accuracy_score, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split    
    from sklearn.ensemble import RandomForestClassifier

    # read the data
    df = pd.read_csv(scaled_dataset.path)
    X = df.iloc[:,0:8] # features columns
    Y = df.iloc[:,8] # label column
    
    # split data into train and test sets
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    score = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    print('accuracy:   %0.3f' % score)
    print('auc:   %0.3f' % auc)
    
    # Save Model
    metrics.log_metric("accuracy", (score * 100.0))
    metrics.log_metric("algorithm", "Random Forest")
    metrics.log_metric("dataset_size", len(df))
    metrics.log_metric("AUC", auc)

    
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(clf, os.path.join(model.path, "model.joblib"))


    
@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys  model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name="rf-diabetes-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    )
    print("Model uploaded to Vertex AI model registry")
    endpoint = deployed_model.deploy(machine_type="n1-standard-2")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name
    


@dsl.pipeline(
    name="mlops-rf-diabetes",
)
def pipeline():
    
    read_and_preprocessing_task = read_and_preprocessing(input_bucket="mlops_landing")
    
    
    train_task =  (
        train(
            scaled_dataset=read_and_preprocessing_task.outputs["scaled_dataset"],
        )
        .after(read_and_preprocessing_task)
        .set_caching_options(False)
    )

    _ = deploy_model(
        project_id=PROJECT_ID,
        model=train_task.outputs["model"],
    )

    
if __name__ == '__main__':
    
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="classification_pipeline.json"
    )
    print('Pipeline compilado exitosamente')