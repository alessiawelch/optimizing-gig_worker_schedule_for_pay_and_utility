from __future__ import annotations
from typing import List
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from single_driver_simulated_environment import ml_models as M

def _mk_stub(feats: List[str], value):
    mdl = SimpleNamespace(predict=lambda X: np.full(len(X), value))
    return mdl, feats


@pytest.fixture(autouse=True)
def _clear_model_caches():
    if hasattr(M, "_loaded"):
        M._loaded.clear()
    # Clear the lru_cache on _price_bundle, if present
    if hasattr(M, "_price_bundle") and hasattr(M._price_bundle, "cache_clear"):
        M._price_bundle.cache_clear()
    yield
    if hasattr(M, "_loaded"):
        M._loaded.clear()
    if hasattr(M, "_price_bundle") and hasattr(M._price_bundle, "cache_clear"):
        M._price_bundle.cache_clear()


def test_prep_column_order() -> None:
    feats: List[str] = ["a", "b", "c"]
    df = pd.DataFrame({"c": [3], "a": [1], "b": [2]})
    out = M._prep(df, feats)
    assert list(out.columns) == feats


def test_predict_distance(monkeypatch: pytest.MonkeyPatch):
    feats = ["x", "y"]
    monkeypatch.setattr(M, "_load", lambda key: _mk_stub(feats, 7.5))  

    df = pd.DataFrame({"y": [0.2, 0.3, 0.4], "x": [0.1, 0.4, 0.9]})
    out = M.predict_distance(df)

    assert isinstance(out, np.ndarray)
    assert out.shape == (len(df),)
    assert np.allclose(out, 7.5)


def test_predict_duration_ratio(monkeypatch: pytest.MonkeyPatch):
    feats = ["f1", "f2"]
    monkeypatch.setattr(M, "_load", lambda key: _mk_stub(feats, 1.25))

    df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    out = M.predict_duration_ratio(df)

    assert isinstance(out, np.ndarray)
    assert out.shape == (len(df),)
    assert np.allclose(out, 1.25)


def test_predict_price_blend(monkeypatch: pytest.MonkeyPatch):
    # Blend weights
    meta = {"blend_w": {"lgbm": 0.6, "xgb": 0.4}}
    feats = ["foo", "bar"]

    # Stub models that output log1p(price)
    class _LGBM:
        def predict(self, X):
            # log1p(20.0) for each row
            return np.full(len(X), np.log1p(20.0))

    class _XGB:
        def predict(self, X):
            # log1p(30.0) for each row
            return np.full(len(X), np.log1p(30.0))

    # Monkeypatch the bundle loader to return our stubs
    monkeypatch.setattr(M, "_price_bundle", lambda: (meta, feats, _LGBM(), _XGB()))

    # Single-row df → function should return a scalar float
    df = pd.DataFrame({"foo": [7], "bar": [3]})
    out = M.predict_price(df)

    # Expected: 0.6*20 + 0.4*30 = 24.0
    assert isinstance(out, float)
    assert pytest.approx(out, rel=1e-12) == 24.0
