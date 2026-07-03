"""Layer 6 — model registry metadata/versioning (offline, filesystem only)."""
from __future__ import annotations

from src.models.registry import ModelRegistry


def _registry(tmp_path):
    reg = ModelRegistry()
    reg.dir = tmp_path
    reg.index_path = tmp_path / "registry.json"
    return reg


def test_register_and_metadata(tmp_path):
    reg = _registry(tmp_path)
    reg.register("NVDA", "downside_xgb", "1.0.0", "fs-1.0.0", metrics=None)
    assert reg.is_trained("NVDA", "downside_xgb")
    md = reg.metadata("NVDA", "downside_xgb")
    assert md["model_version"] == "1.0.0"
    assert md["feature_set_version"] == "fs-1.0.0"
    assert md["training_date"] is not None


def test_unregistered_model_synthesizes_metadata(tmp_path):
    reg = _registry(tmp_path)
    # GARCH is fit on-demand (never registered) -> metadata synthesized from config.
    md = reg.metadata("NVDA", "vol_garch")
    assert md["model_name"] == "vol_garch"
    assert "model_version" in md and "feature_set_version" in md


def test_is_trained_false_when_absent(tmp_path):
    assert _registry(tmp_path).is_trained("TSLA", "downside_xgb") is False
