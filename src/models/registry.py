"""Model registry (spec §11) — versioned artifacts + metadata index.

Artifacts live on disk (``models_store/``); a JSON index tracks model_name,
version, training_date, feature_set_version per ticker so every prediction can
be traced to an exact, reproducible model. Training metrics are also recorded
to the ``model_performance`` DB table.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.config import get_model_config, project_root
from src.core.logging import get_logger
from src.core.time import now_utc
from src.db import repositories as repo
from src.db.session import get_session

log = get_logger("models.registry")


class ModelRegistry:
    def __init__(self):
        cfg = get_model_config()
        self.dir = project_root() / cfg.get("registry", {}).get("artifacts_dir", "models_store")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "registry.json"

    def artifact_path(self, ticker: str, model_name: str) -> Path:
        return self.dir / f"{ticker.upper()}_{model_name}"

    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {}

    def register(self, ticker: str, model_name: str, model_version: str,
                 feature_set_version: str, training_date: datetime | None = None,
                 metrics: dict[str, float] | None = None, mode: str = "strict") -> dict[str, Any]:
        """Record a trained model's metadata + optional metrics."""
        training_date = training_date or now_utc()
        index = self._load_index()
        meta = {
            "model_name": model_name, "model_version": model_version,
            "feature_set_version": feature_set_version,
            "training_date": training_date.isoformat(),
            "artifact": str(self.artifact_path(ticker, model_name).name),
        }
        index.setdefault(ticker.upper(), {})[model_name] = meta
        self.index_path.write_text(json.dumps(index, indent=2))

        if metrics:
            with get_session() as s:
                repo.record_model_performance(s, [{
                    "model_name": model_name, "model_version": model_version,
                    "training_date": training_date, "feature_set_version": feature_set_version,
                    "metric_name": k, "metric_value": float(v), "window": "walk_forward", "mode": mode,
                } for k, v in metrics.items() if v == v])  # skip NaN
        log.info("model_registered", ticker=ticker, model=model_name, version=model_version)
        return meta

    def metadata(self, ticker: str, model_name: str) -> dict[str, Any]:
        entry = self._load_index().get(ticker.upper(), {}).get(model_name)
        if entry:
            return entry
        # Unregistered (e.g. on-demand GARCH): synthesize from config.
        cfg = get_model_config()
        block = {"vol_garch": cfg.get("volatility_model", {})}.get(model_name, {})
        return {
            "model_name": model_name,
            "model_version": block.get("version", "0.0.0"),
            "feature_set_version": cfg.get("feature_set_version", "fs-1.0.0"),
            "training_date": None,
        }

    def is_trained(self, ticker: str, model_name: str) -> bool:
        return model_name in self._load_index().get(ticker.upper(), {})
