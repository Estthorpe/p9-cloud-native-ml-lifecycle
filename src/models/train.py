"""P9 cloud training: bearing anomaly model (LightGBM).

Reads registered data assets, computes point-in-time-correct features
(same definitions as the feature store's Spark transformer), trains,
evaluates with recall@K, and logs everything to MLflow.
"""

import argparse

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd


def build_features(sensor: pd.DataFrame) -> pd.DataFrame:
    """Pandas mirror of the feature store transformer. PIT-correct:
    every window ends at the current row - no future data."""
    df = sensor.sort_values(["device_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    grp = df.groupby("device_id")["value"]

    df["rms_value"] = df["value"]
    df["rolling_mean_1h"] = grp.transform(lambda s: s.rolling("1h").mean())
    df["rolling_std_1h"] = grp.transform(lambda s: s.rolling("1h").std())
    df["deviation_from_baseline"] = grp.transform(lambda s: s - s.expanding().mean())
    df["rolling_std_1h"] = df["rolling_std_1h"].fillna(0)  # first row per device
    return df.reset_index()


def recall_at_k(y_true: pd.Series, scores: pd.Series, k_frac: float = 0.10) -> float:
    """Of all true anomalies, what fraction sits in the top-k% riskiest rows?"""
    k = max(1, int(len(scores) * k_frac))
    top_k_idx = scores.nlargest(k).index
    caught = y_true.loc[top_k_idx].sum()
    total = y_true.sum()
    return float(caught / total) if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-data", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    args = parser.parse_args()

    sensor = pd.read_csv(args.sensor_data)
    labels = pd.read_csv(args.labels)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    feats = build_features(sensor)
    data = feats.merge(labels, on=["timestamp", "device_id"], how="left")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["is_anomaly"] = data["is_anomaly"].fillna(0).astype(int)

    # Group-aware split: learn failure signature on one bearing,
    # prove generalization on a bearing the model has never seen.
    # (Time split starves training: anomalies cluster at end-of-life.)
    train_devices = ["BEARING_1", "BEARING_2", "BEARING_3"]
    test_devices = ["BEARING_4"]
    train = data[data["device_id"].isin(train_devices)]
    test = data[data["device_id"].isin(test_devices)]

    feature_cols = ["rms_value", "rolling_mean_1h", "rolling_std_1h", "deviation_from_baseline"]

    mlflow.lightgbm.autolog(log_models=False)  # we log manually to register the model
    with mlflow.start_run():
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, scale_pos_weight=50, random_state=42
        )
        model.fit(train[feature_cols], train["is_anomaly"])

        scores = pd.Series(model.predict_proba(test[feature_cols])[:, 1], index=test.index)
        r_at_10 = recall_at_k(test["is_anomaly"], scores, 0.10)

        mlflow.log_metric("recall_at_10pct", r_at_10)
        mlflow.log_metric("test_anomaly_count", int(test["is_anomaly"].sum()))
        mlflow.lightgbm.log_model(model, "model", registered_model_name="bearing-anomaly-model")
        print(f"recall@10%: {r_at_10:.3f}")


if __name__ == "__main__":
    main()
