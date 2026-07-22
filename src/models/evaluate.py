"""P9 evaluation gate: the registered model must clear the floor and never regress.

Exit 0 = blessed. Exit 1 = rejected (CI goes red).
Run:  python -m src.models.evaluate
"""

import sys

import mlflow.lightgbm
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from src.config.settings import settings
from src.models.train import build_features, recall_at_k

RECALL_FLOOR = 0.90
MODEL_NAME = "bearing-anomaly-model"
TEST_DEVICES = ["BEARING_4"]


def load_test_data() -> pd.DataFrame:
    sensor = pd.read_csv(settings.sensor_data_path)
    labels = pd.read_csv(settings.labels_path)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    feats = build_features(sensor)
    data = feats.merge(labels, on=["timestamp", "device_id"], how="left")
    data["is_anomaly"] = data["is_anomaly"].fillna(0).astype(int)
    return data[data["device_id"].isin(TEST_DEVICES)]


def score_version(model_uri: str, test: pd.DataFrame) -> float:
    model = mlflow.lightgbm.load_model(model_uri)
    feature_cols = ["rms_value", "rolling_mean_1h", "rolling_std_1h", "deviation_from_baseline"]
    scores = pd.Series(model.predict_proba(test[feature_cols])[:, 1], index=test.index)
    return recall_at_k(test["is_anomaly"], scores, 0.10)


def main() -> int:
    ml_client = MLClient(
        DefaultAzureCredential(),
        settings.subscription_id,
        settings.resource_group,
        settings.workspace_name,
    )
    mlflow.set_tracking_uri(ml_client.workspaces.get(settings.workspace_name).mlflow_tracking_uri)

    versions = [int(m.version) for m in ml_client.models.list(name=MODEL_NAME)]
    latest = max(versions)
    previous = max((v for v in versions if v != latest), default=None)

    test = load_test_data()
    latest_recall = score_version(f"models:/{MODEL_NAME}/{latest}", test)
    print(f"version {latest}: recall@10% = {latest_recall:.3f}")
    absolute_floor_passed = latest_recall >= RECALL_FLOOR
    regression_passed = True
    if previous is not None:
        previous_recall = score_version(f"models:/{MODEL_NAME}/{previous}", test)
        print(f"version {previous}: recall@10% = {previous_recall:.3f}")
        regression_passed = latest_recall >= previous_recall

    if not absolute_floor_passed:
        print(f"FAIL: version {latest} did not meet the absolute floor of {RECALL_FLOOR:.2f}")
        return 1
    if not regression_passed:
        print(
            f"FAIL: version {latest} ({latest_recall:.3f}) regressed "
            f"from version {previous} ({previous_recall:.3f})"
        )
        return 1
    print(f"PASS: version {latest} cleared the floor and did not regress")

    return 0


if __name__ == "__main__":
    sys.exit(main())
