from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb 
import json
from functools import lru_cache

class _SilentLogger:
    def info(self, msg):     
        pass
    def warning(self, msg):  
        pass

lgb.register_logger(_SilentLogger())

THIS_DIR   = Path(__file__).resolve().parent     
BASE_DIR  = THIS_DIR.parent.parent / "driver_data" / "models"
PRICE_MODEL_DIR =  BASE_DIR / "price" / "model_artifacts" / "price_v4_original"

MODELS = {
    "distance":  BASE_DIR / "distance" / "model_artifacts" / "distance_v3_1_original",
    "duration":  BASE_DIR / "duration" / "model_artifacts" /  "duration_v4_original",
    "price":     PRICE_MODEL_DIR,
}

_loaded = {}  

def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252") as f:
            return json.load(f)

def _to_float(x) -> float:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(()))
    raise ValueError(f"Expected scalar prediction, got shape {arr.shape}")    


def _load(model_key: str):
    if model_key in _loaded:
        return _loaded[model_key]

    d = MODELS[model_key]
    mdl: lgb.Booster | lgb.LGBMRegressor = joblib.load(d / f"{model_key}_lgbm.joblib")

    # read feature order
    with open(d / f"{model_key}_feature_order.json", "r") as f:
        feats: list[str] = json.load(f)

    _loaded[model_key] = (mdl, feats)
    return _loaded[model_key]

@lru_cache(maxsize=1)
def _price_bundle():
    """Load/caches price metadata, feature order, and both models once."""
    meta  = _load_json(PRICE_MODEL_DIR / "metadata.json")
    feats = _load_json(PRICE_MODEL_DIR / "price_feature_order.json")
    mdl_lgb = joblib.load(PRICE_MODEL_DIR / "price_lgbm_log.joblib")
    mdl_xgb = joblib.load(PRICE_MODEL_DIR / "price_xgb_log.joblib")
    return meta, feats, mdl_lgb, mdl_xgb

def _prep(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    return df.reindex(columns=feats)

def predict_distance(df: pd.DataFrame) -> np.ndarray:
    mdl, feats = _load("distance")
    return mdl.predict(_prep(df, feats)) # predict straight

def predict_duration_ratio(df: pd.DataFrame) -> np.ndarray:
    mdl, feats = _load("duration")
    return mdl.predict(_prep(df, feats)) #predict ratio

def predict_price(df: pd.DataFrame) -> float:
    meta, feats, mdl_lgb, mdl_xgb = _price_bundle()
    w = meta["blend_w"]

    X = _prep(df, feats)
    pred_lgb = np.expm1(mdl_lgb.predict(X))
    pred_xgb = np.expm1(mdl_xgb.predict(X))

    blended = w["lgbm"] * pred_lgb + w["xgb"] * pred_xgb
    return _to_float(blended)  