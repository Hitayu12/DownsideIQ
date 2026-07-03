"""Phase 4 tests: target builder, walk-forward splits, classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.walk_forward import walk_forward_splits
from src.models.downside_classifier import DownsideClassifier
from src.models.target_builder import build_targets


def _prices_with_drop():
    idx = pd.date_range("2024-01-01", periods=40, freq="B", tz="UTC")
    close = np.full(40, 100.0)
    close[20] = 100.0
    close[21] = 90.0   # -10% next-session move at bar 20
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "adj_close": close, "volume": 1e6},
        index=idx,
    )


def test_target_label_on_sharp_drop():
    t = build_targets(_prices_with_drop())
    # Bar 20 has a large negative forward return -> downside_label == 1.
    assert t["downside_label"].iloc[20] == 1.0


def test_last_bar_target_is_nan():
    t = build_targets(_prices_with_drop())
    assert np.isnan(t["future_return"].iloc[-1])
    assert np.isnan(t["downside_label"].iloc[-1])


def test_walk_forward_is_time_ordered_and_nonoverlapping():
    splits = walk_forward_splits(n=1000, n_splits=5, min_train=252)
    seen_test = set()
    for sp in splits:
        # Train strictly precedes test (no look-ahead across the split).
        assert sp.train_idx.max() < sp.test_idx.min()
        # Test blocks don't overlap.
        assert not (set(sp.test_idx.tolist()) & seen_test)
        seen_test |= set(sp.test_idx.tolist())


def test_walk_forward_expands():
    splits = walk_forward_splits(n=1000, n_splits=5, min_train=252)
    sizes = [len(sp.train_idx) for sp in splits]
    assert sizes == sorted(sizes)          # expanding window grows monotonically


def test_classifier_learns_separable_signal(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, 0.3, n) > 0).astype(int)   # separable-ish
    X = pd.DataFrame({"feat": x, "noise": rng.normal(0, 1, n)})
    y = pd.Series(y)

    clf = DownsideClassifier(params={"n_estimators": 80}).fit(X.iloc[:450], y.iloc[:450])
    p = clf.predict_proba(X.iloc[450:])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y.iloc[450:], p) > 0.8

    # Save / load roundtrip.
    clf.save(tmp_path / "m")
    clf2 = DownsideClassifier.load(tmp_path / "m")
    p2 = clf2.predict_proba(X.iloc[450:])
    assert np.allclose(p, p2, atol=1e-6)
