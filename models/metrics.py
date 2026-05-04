import numpy as np


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, latencies: list[float]) -> dict:
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0

    if len(actual) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100)
    else:
        dir_acc = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "dir_accuracy": dir_acc,
        "mean_latency_ms": float(np.mean(latencies) * 1000),
    }
