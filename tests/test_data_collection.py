"""Phase 2 tests: data loader + agents (offline parts only)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.agents.market_context_agent import context_assets_for
from src.utils import data_loader
from src.utils.data_loader import _yf_interval


def test_yf_interval_mapping():
    assert _yf_interval("1d") == "1d"
    assert _yf_interval("1h") == "1h"
    with pytest.raises(ValueError):
        _yf_interval("3y")


def test_context_assets_dedup_and_order():
    assets = context_assets_for("NVDA")
    # market ETFs come first, no duplicates, VIX proxy included.
    assert assets[0] in {"SPY", "QQQ"}
    assert len(assets) == len(set(assets))
    assert "^VIX" in assets
    assert "AMD" in assets


def test_json_roundtrip(tmp_path, monkeypatch):
    # Redirect data_dir to a temp location for an isolated I/O test.
    monkeypatch.setattr(data_loader, "data_dir", lambda: tmp_path)
    obj = {"a": 1, "ts": pd.Timestamp("2026-05-30", tz="UTC")}
    data_loader.save_json(obj, "macro", "unit_test")
    loaded = data_loader.load_json("macro", "unit_test")
    assert loaded["a"] == 1


def test_frame_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "data_dir", lambda: tmp_path)
    df = pd.DataFrame(
        {"close": [1.0, 2.0], "volume": [10, 20]},
        index=pd.to_datetime(["2026-05-28", "2026-05-29"], utc=True),
    )
    data_loader.save_frame(df, "prices", "UNIT")
    loaded = data_loader.load_frame("prices", "UNIT")
    assert list(loaded["close"]) == [1.0, 2.0]
