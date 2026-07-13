"""Smoke test: typed settings load and contain sane values."""

from src.config.settings import Settings


def test_settings_load() -> None:
    s = Settings()
    assert s.resource_group == "rg-clariv-foundation"
    assert s.workspace_name == "mlw-clariv-p9"
    assert s.location == "uksouth"
