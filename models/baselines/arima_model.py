import time
import logging
import numpy as np
import pandas as pd

from models.metrics import compute_metrics

log = logging.getLogger(__name__)


def run_arima(cfg: dict) -> dict:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return {"model": "arima", "error": "statsmodels not installed"}

    order = tuple(cfg["models"]["arima"]["order"])
    n_runs = cfg["benchmark"]["n_latency_runs"]

    try:
        train_df = pd.read_parquet("data/processed/train.parquet")
        test_df  = pd.read_parquet("data/processed/test.parquet")

        all_actual, all_pred, latencies = [], [], []

        for ticker in train_df["ticker"].unique():
            train_close = train_df[train_df["ticker"] == ticker]["close"].sort_index().values
            test_close  = test_df[test_df["ticker"] == ticker]["close"].sort_index().values

            fit = ARIMA(train_close, order=order).fit()

            for _ in range(max(1, n_runs // len(train_df["ticker"].unique()))):
                t0 = time.perf_counter()
                fc = fit.forecast(steps=len(test_close))
                latencies.append(time.perf_counter() - t0)

            all_actual.append(test_close)
            all_pred.append(fc)

        actual    = np.concatenate(all_actual)
        predicted = np.concatenate(all_pred)

        result = compute_metrics(actual, predicted, latencies)
        result["model"] = "arima"
        log.info(f"ARIMA — MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
        return result

    except Exception as e:
        log.error(f"ARIMA failed: {e}")
        return {"model": "arima", "error": str(e)}
