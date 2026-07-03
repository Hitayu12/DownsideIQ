"""Domain contracts for ingestion + features (typed service interfaces)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.core.errors import DegradedMode


@dataclass
class IngestionResult:
    """Raw, validated inputs for one analysis run (DataFrames + raw payloads)."""

    ticker: str
    as_of: datetime
    bar_size: str
    prices: pd.DataFrame
    context: dict[str, pd.DataFrame] = field(default_factory=dict)
    raw_macro_news: list[dict] = field(default_factory=list)
    raw_company_news: list[dict] = field(default_factory=list)
    fundamentals: dict[str, Any] = field(default_factory=dict)
    degraded: DegradedMode = field(default_factory=DegradedMode)


class FeatureSnapshot(BaseModel):
    """One leakage-safe feature row for a prediction timestamp."""

    ticker: str
    ts: datetime
    bar_size: str
    horizon: str
    feature_set_version: str
    features: dict[str, Any]
    data_confidence_score: float = Field(ge=0.0, le=1.0)
    missing_data_flag: bool = False

    def numeric_features(self) -> dict[str, float]:
        out = {}
        for k, v in self.features.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(v)
        return out
