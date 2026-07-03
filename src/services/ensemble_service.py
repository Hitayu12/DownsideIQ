"""Ensemble service (spec §1, §5) — assemble Council-2 + news overlay inputs.

Merges the live news overlay aggregates into the feature row and hands the
combined ``model_outputs`` to the risk engine. The scoring math itself
(weighted ensemble, capped log-odds overlay, agreement, regime, uncertainty)
lives in the validated ``ensemble_engine`` and is invoked by the risk engine.
"""
from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.domain.features import FeatureSnapshot
from src.domain.predictions import CouncilOutputs

log = get_logger("services.ensemble")

_OVERLAY_DEFAULTS = {
    "company_news_risk_score": 0.0, "macro_risk_score": 0.0,
    "negative_catalyst_score": 0.0, "positive_catalyst_score": 0.0,
    "abnormal_news_volume_flag": 0.0, "company_specificity_score": 0.0,
}


class EnsembleService:
    def assemble(self, council: CouncilOutputs, snapshot: FeatureSnapshot,
                 overlay_features: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build the ``model_outputs`` dict consumed by the risk engine."""
        feature_row = dict(snapshot.features)
        # News overlay defaults ensure the gate is well-defined even in price-only mode.
        for k, v in _OVERLAY_DEFAULTS.items():
            feature_row.setdefault(k, v)
        if overlay_features:
            feature_row.update(overlay_features)

        model_outputs = council.as_decision_inputs()
        model_outputs["feature_row"] = feature_row
        return model_outputs
