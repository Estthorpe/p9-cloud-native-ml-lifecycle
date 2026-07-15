"""Data contracts for P9 ingestion — validation gate before any upload/registration.

Run directly:  python -m src.ingestion.contracts
Exits non-zero on any contract violation (fail closed).
"""

import sys

import pandas as pd
import structlog

from src.config.logging_config import configure_logging
from src.config.settings import settings

logger = structlog.get_logger()


class ContractViolation(Exception):
    """Raised when a dataset breaks its contract."""


def validate_sensor_data(df: pd.DataFrame) -> None:
    """Contract for sensor_data_real.csv. Raises ContractViolation on breach."""
    # Rule 1: required columns present
    required = {"timestamp", "device_id", "sensor_type", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ContractViolation(f"sensor data missing columns: {missing}")

    # Rule 2: row count in sane band (known good: 8,624)
    if not 8000 <= len(df) <= 9500:
        raise ContractViolation(f"sensor row count {len(df)} outside band [8000, 9500]")

    # Rule 3: value column — no nulls, no negatives
    if df["value"].isna().any() or (df["value"] < 0).any():
        raise ContractViolation("sensor 'value' column contains nulls or negatives")


def validate_labels(labels: pd.DataFrame, sensor: pd.DataFrame) -> None:
    """Contract for anomaly_labels_real.csv. Raises ContractViolation on breach."""
    # Rule 4: required columns present
    required = {"timestamp", "device_id", "is_anomaly"}
    missing = required - set(labels.columns)
    if missing:
        raise ContractViolation(f"labels missing columns: {missing}")

    # Rule 5: anomaly rate within expected band (known good: ~0.8%)
    rate = labels["is_anomaly"].mean()
    if not 0.004 <= rate <= 0.02:
        raise ContractViolation(f"anomaly rate {rate:.3%} outside band [0.4%, 2.0%]")

    # Rule 6: no orphan labels — every label timestamp exists in sensor data
    if not labels["timestamp"].isin(sensor["timestamp"]).all():
        raise ContractViolation("labels contain timestamps not present in sensor data")


def main() -> int:
    configure_logging()
    sensor = pd.read_csv(settings.sensor_data_path)
    labels = pd.read_csv(settings.labels_path)
    try:
        validate_sensor_data(sensor)
        validate_labels(labels, sensor)
    except ContractViolation as exc:
        logger.error("contract_violation", detail=str(exc))
        return 1
    logger.info("contracts_passed", sensor_rows=len(sensor), label_rows=len(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
