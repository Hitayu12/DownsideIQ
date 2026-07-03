"""Domain contracts for model predictions (spec §11 model governance)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ModelPrediction(BaseModel):
    """One model's output with full versioning metadata."""

    model_name: str
    model_version: str
    training_date: datetime | None
    feature_set_version: str
    prediction_timestamp: datetime
    ticker: str
    output: dict[str, Any]

    model_config = {"protected_namespaces": ()}   # allow model_* field names


class CouncilOutputs(BaseModel):
    """The three Council-2 model outputs for one prediction timestamp."""

    ticker: str
    timestamp: datetime
    classifier: ModelPrediction
    volatility: ModelPrediction
    quantile: ModelPrediction

    model_config = {"protected_namespaces": ()}

    def as_decision_inputs(self) -> dict[str, Any]:
        """Flatten into the shape the ensemble/decision engine expects."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "classifier": {"p_downside": self.classifier.output.get("p_downside"),
                           "top_features": self.classifier.output.get("top_features", [])},
            "garch": self.volatility.output,
            "quantile": self.quantile.output,
        }
