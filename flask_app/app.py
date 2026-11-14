import logging
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, render_template, request
from mlflow.tracking import MlflowClient

from .preprocessing_utility import normalize_text


logger = logging.getLogger(__name__)


# DagsHub credentials and MLflow configuration
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "samarjeetsinghpatil93-jpg"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)


# Initialize Flask application
app = Flask(__name__)


def resolve_model_uri(model_name: str) -> str:
    """
    Resolve the MLflow model URI.

    Prefer an explicitly configured model URI.
    Otherwise, use the current Production model version.
    """

    configured_uri = os.getenv(
        "MLFLOW_MODEL_URI", ""
    ).strip()

    if configured_uri:
        return configured_uri

    # Compatibility fallback for the current DagsHub registry.
    # MLflow stage APIs are deprecated.
    client = MlflowClient()

    versions = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    if not versions:
        raise RuntimeError(
            f"No Production version found for model "
            f"{model_name!r}. Promote a reviewed model "
            f"or configure MLFLOW_MODEL_URI."
        )

    latest = max(
        versions,
        key=lambda version: int(version.version)
    )

    return f"models:/{model_name}/{latest.version}"


# Load model from MLflow Model Registry
model_name = "my_model"

model_uri = resolve_model_uri(model_name)

logger.info(
    "Loading MLflow model from %s",
    model_uri
)

model = mlflow.pyfunc.load_model(model_uri)


# Load vectorizer
vectorizer_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "vectorizer.pkl"
)

with vectorizer_path.open("rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Home page
@app.route("/")
def home():
    return render_template(
        "index.html",
        result=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    # Get input text
    text = request.form["text"]

    # Normalize text
    text = normalize_text(text)

    # Convert text to sparse Bag-of-Words features
    features = vectorizer.transform([text])

    # Predict directly using the sparse matrix
    prediction = model.predict(features)

    return render_template(
        "index.html",
        result=prediction[0]
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )