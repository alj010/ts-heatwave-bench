import time
import logging
import numpy as np
import pandas as pd

from models.metrics import compute_metrics

log = logging.getLogger(__name__)

_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "return_1d", "return_5d",
    "sma_5", "sma_10", "sma_20",
    "std_5", "std_10", "std_20",
    "hl_spread", "vol_change_1d",
]


def _prepare(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df  = pd.read_parquet("data/processed/test.parquet")

    def make_xy(df: pd.DataFrame):
        parts = []
        for _, grp in df.groupby("ticker"):
            grp = grp.sort_index().copy()
            grp["target"] = grp["close"].shift(-1)
            parts.append(grp.dropna(subset=["target"]))
        combined = pd.concat(parts)
        feats = [c for c in _FEATURE_COLS if c in combined.columns]
        return combined[feats].values, combined["target"].values

    X_tr, y_tr = make_xy(train_df)
    X_te, y_te = make_xy(test_df)
    return X_tr, y_tr, X_te, y_te


def run_xgboost(cfg: dict) -> dict:
    try:
        import xgboost as xgb
    except ImportError:
        return {"model": "xgboost", "error": "xgboost not installed"}

    try:
        p      = cfg["models"]["xgboost"]
        n_runs = cfg["benchmark"]["n_latency_runs"]
        X_tr, y_tr, X_te, y_te = _prepare(cfg)

        model = xgb.XGBRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            preds = model.predict(X_te)
            latencies.append(time.perf_counter() - t0)

        result = compute_metrics(y_te, preds, latencies)
        result["model"] = "xgboost"
        log.info(f"XGBoost — MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
        return result

    except Exception as e:
        log.error(f"XGBoost failed: {e}")
        return {"model": "xgboost", "error": str(e)}


def run_lightgbm(cfg: dict) -> dict:
    try:
        import lightgbm as lgb
    except ImportError:
        return {"model": "lightgbm", "error": "lightgbm not installed"}

    try:
        p      = cfg["models"]["lightgbm"]
        n_runs = cfg["benchmark"]["n_latency_runs"]
        X_tr, y_tr, X_te, y_te = _prepare(cfg)

        model = lgb.LGBMRegressor(
            n_estimators=p["n_estimators"],
            num_leaves=p["num_leaves"],
            learning_rate=p["learning_rate"],
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            preds = model.predict(X_te)
            latencies.append(time.perf_counter() - t0)

        result = compute_metrics(y_te, preds, latencies)
        result["model"] = "lightgbm"
        log.info(f"LightGBM — MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
        return result

    except Exception as e:
        log.error(f"LightGBM failed: {e}")
        return {"model": "lightgbm", "error": str(e)}
