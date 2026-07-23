"""P9 scoring script - managed online endpoint (init/run contract).

Request:  {"device_id": "BEARING_4", "timestamp": "2003-11-20 12:00:00", "value": 0.19}
Response: {"device_id": ..., "risk_score": 0.94, "verdict": "anomaly", "model_version": "3"}
"""

import json
import os
from pathlib import Path

import mlflow.lightgbm
import pandas as pd

model = None
history = None
FEATURE_COLS = ["rms_value", "rolling_mean_1h", "rolling_std_1h", "deviation_from_baseline"]
THRESHOLD = 0.50


def init():
    global model, history
    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])
    mlflow_dir = next(model_dir.rglob("MLmodel")).parent
    model = mlflow.lightgbm.load_model(str(mlflow_dir))
    ref = Path(__file__).parent / "deploy_assets" / "reference_history.csv"
    history = pd.read_csv(ref)
    history["timestamp"] = pd.to_datetime(history["timestamp"])


def run(raw_data: str) -> dict:
    req = json.loads(raw_data)

    # 1. Validate the form before touching the contents (never crash on bad input)
    for field in ("device_id", "timestamp", "value"):
        if field not in req:
            return {"error": f"missing required field: {field}"}
    try:
        value = float(req["value"])
        timestamp = pd.to_datetime(req["timestamp"])
    except (ValueError, TypeError) as exc:
        return {"error": f"bad field value: {exc}"}
    if value < 0:
        return {"error": f"value must be non-negative, got {value}"}

    # 2. Append the incoming reading to the reference history and rebuild features
    incoming = pd.DataFrame(
        [
            {
                "device_id": str(req["device_id"]),
                "timestamp": timestamp,
                "sensor_type": "vibration",
                "value": value,
            }
        ]
    )
    df = pd.concat([history, incoming], ignore_index=True)
    df = df.sort_values(["device_id", "timestamp"]).set_index("timestamp")

    grp = df.groupby("device_id")["value"]
    df["rms_value"] = df["value"]
    df["rolling_mean_1h"] = grp.transform(lambda s: s.rolling("1h").mean())
    df["rolling_std_1h"] = grp.transform(lambda s: s.rolling("1h").std()).fillna(0)
    df["deviation_from_baseline"] = grp.transform(lambda s: s - s.expanding().mean())
    df = df.reset_index()

    # 3. Score only the incoming reading (it sorted to the row matching its timestamp;
    #    select it explicitly rather than assuming it's last)
    mask = (df["device_id"] == str(req["device_id"])) & (df["timestamp"] == timestamp)
    last_row = df[mask].iloc[[-1]]
    risk = float(model.predict_proba(last_row[FEATURE_COLS])[:, 1][0])

    # 4. The reply
    return {
        "device_id": str(req["device_id"]),
        "timestamp": str(req["timestamp"]),
        "risk_score": round(risk, 4),
        "verdict": "anomaly" if risk >= THRESHOLD else "normal",
        "model_version": os.environ.get("MODEL_VERSION", "unknown"),
    }
