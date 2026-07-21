"""Smoke test: the full training pipeline runs end-to-end on real data.

Catches code/dtype/import errors locally before any cloud submission.
Skips (like the contract tests) when P7 data is absent (CI runners).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import settings
from src.models.train import build_features, recall_at_k

DATA_PRESENT = Path(settings.sensor_data_path).exists() and Path(settings.labels_path).exists()
pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="P7 data files not on this machine")


def test_training_pipeline_end_to_end() -> None:
    import lightgbm as lgb

    sensor = pd.read_csv(settings.sensor_data_path)
    labels = pd.read_csv(settings.labels_path)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    feats = build_features(sensor)
    data = feats.merge(labels, on=["timestamp", "device_id"], how="left")
    data["is_anomaly"] = data["is_anomaly"].fillna(0).astype(int)

    assert list(data.columns).count("timestamp") == 1  # merge keys aligned
    assert not data["rolling_mean_1h"].isna().any()

    model = lgb.LGBMClassifier(n_estimators=10, random_state=42)  # tiny: speed
    feature_cols = ["rms_value", "rolling_mean_1h", "rolling_std_1h", "deviation_from_baseline"]
    model.fit(data[feature_cols], data["is_anomaly"])

    scores = pd.Series(model.predict_proba(data[feature_cols])[:, 1], index=data.index)
    r = recall_at_k(data["is_anomaly"], scores, 0.10)
    assert 0.0 <= r <= 1.0
