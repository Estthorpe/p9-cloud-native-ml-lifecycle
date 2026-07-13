"""Typed configuration for P9 — validated at startup via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """P9 runtime settings. Values load from environment variables or .env."""

    model_config = SettingsConfigDict(env_prefix="P9_", env_file=".env")

    subscription_id: str = "ef005542-7ef3-4995-8b0e-830d1cfe3335"
    resource_group: str = "rg-clariv-foundation"
    workspace_name: str = "mlw-clariv-p9"
    location: str = "uksouth"
    sensor_data_path: str = (
        "C:/Users/enuzo/Documents/ai_eng_projs/predictive-maintenance-triage/data/"
        "sensor_data_real.csv"
    )
    labels_path: str = (
        "C:/Users/enuzo/Documents/ai_eng_projs/predictive-maintenance-triage/data/"
        "anomaly_labels_real.csv"
    )


settings = Settings()
