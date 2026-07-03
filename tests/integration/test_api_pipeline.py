"""Layer 10/12 — API + full-pipeline integration (deterministic, no network).

Ingestion is stubbed at its boundary with deterministic SAMPLE price data;
everything downstream (real feature build, real model training on the sample,
ensemble, risk engine, ledger persistence, API async path) runs for real.
Model artifacts are redirected to a temp dir so the test never pollutes
``models_store``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.domain.features import IngestionResult


def _sample_prices(n=420, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2026-05-29", periods=n, freq="B", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "adj_close": close, "volume": 1e6}, index=idx)


class _FakeIngestion:
    def ingest(self, ticker, bar_size=None, *, as_of=None, persist=True):
        from src.core.time import now_utc
        return IngestionResult(ticker=ticker.upper(), as_of=now_utc(), bar_size="1d",
                               prices=_sample_prices(), context={}, raw_macro_news=[],
                               raw_company_news=[], fundamentals={})


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Redirect model artifacts to a temp dir (no models_store pollution).
    import src.models.registry as registry_mod
    cfg = registry_mod.get_model_config()
    patched = {**cfg, "registry": {"artifacts_dir": str(tmp_path)}}
    monkeypatch.setattr(registry_mod, "get_model_config", lambda: patched)

    # Stub ingestion (network boundary) with deterministic sample data.
    import src.pipeline.orchestrator as orch
    monkeypatch.setattr(orch, "IngestionService", _FakeIngestion)

    from src.providers import gemini_provider
    monkeypatch.setattr(gemini_provider, "available", lambda: False)

    from src.api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert set(body["providers"]) >= {"tavily", "gemini", "alpha_vantage", "finnhub"}


def test_analyze_async_path_and_persistence(client):
    job = client.post("/analyze/NVDA", params={"mode": "strict"}).json()
    assert job["status"] == "queued"

    res = client.get(f"/jobs/{job['job_id']}").json()
    assert res["state"] == "SUCCESS" and res["ready"] is True
    decision = res["result"]
    assert decision["decision"] in ("SHORT", "WATCH", "NO TRADE")
    assert decision["data_quality"] in ("ok", "degraded", "blocked")
    sid = decision["signal_id"]
    assert sid

    # Signal persisted + retrievable with full governance.
    latest = client.get("/signals/latest", params={"ticker": "NVDA", "mode": "strict"}).json()
    assert latest["signal_id"] == sid
    gov = client.get(f"/predictions/{sid}").json()["governance"]
    assert "gates" in gov and "data_quality" in gov and "scores" in gov


def test_invalid_mode_rejected(client):
    assert client.post("/analyze/NVDA", params={"mode": "bogus"}).status_code == 400


def test_unknown_signal_404(client):
    assert client.get("/predictions/does-not-exist").status_code == 404
