"""Contract tests — re-assert data contracts in CI against the local dataset copies.

Skips (rather than fails) when the P7 data files are not present, so CI on
GitHub runners (which have no local data) stays green while any machine with
the data enforces the full contract.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import settings
from src.ingestion.contracts import validate_labels, validate_sensor_data

DATA_PRESENT = Path(settings.sensor_data_path).exists() and Path(settings.labels_path).exists()

pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="P7 data files not on this machine")


def test_sensor_contract_passes() -> None:
    validate_sensor_data(pd.read_csv(settings.sensor_data_path))


def test_labels_contract_passes() -> None:
    sensor = pd.read_csv(settings.sensor_data_path)
    labels = pd.read_csv(settings.labels_path)
    validate_labels(labels, sensor)


def test_contract_catches_bad_anomaly_rate() -> None:
    """Prove the gate actually closes: a 12% anomaly rate must raise."""
    sensor = pd.read_csv(settings.sensor_data_path)
    labels = pd.read_csv(settings.labels_path)
    poisoned = labels.copy()
    poisoned.loc[: int(len(poisoned) * 0.12), "is_anomaly"] = 1
    with pytest.raises(Exception, match="anomaly rate"):
        validate_labels(poisoned, sensor)
