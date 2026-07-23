"""Smoke: scoring script init/run locally, healthy + anomalous + garbage inputs."""

import json
from pathlib import Path

import pytest

DATA_PRESENT = Path("src/serving/deploy_assets/reference_history.csv").exists()
pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="deploy assets not built")


def test_score_healthy_and_anomalous(tmp_path, monkeypatch):
    import mlflow
    import mlflow.lightgbm

    # stage the registered model locally the way AZUREML_MODEL_DIR would
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    from src.config.settings import settings
    from src.models.evaluate import MODEL_NAME

    c = MLClient(
        DefaultAzureCredential(),
        settings.subscription_id,
        settings.resource_group,
        settings.workspace_name,
    )
    mlflow.set_tracking_uri(c.workspaces.get(settings.workspace_name).mlflow_tracking_uri)
    mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/3", dst_path=str(tmp_path))
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))

    from src.serving import score

    score.init()

    healthy = score.run(
        json.dumps({"device_id": "BEARING_4", "timestamp": "2003-10-25 12:00:00", "value": 0.11})
    )
    assert healthy["verdict"] == "normal", healthy

    anomalous = score.run(
        json.dumps({"device_id": "BEARING_4", "timestamp": "2003-11-25 23:00:00", "value": 0.45})
    )
    assert anomalous["verdict"] == "anomaly", anomalous

    garbage = score.run(json.dumps({"device_id": "BEARING_4"}))
    assert "error" in garbage
