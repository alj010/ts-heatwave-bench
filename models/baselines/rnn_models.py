import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.metrics import compute_metrics

log = logging.getLogger(__name__)


def _make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _prepare(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lookback = cfg["dataset"]["lookback"]
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df  = pd.read_parquet("data/processed/test.parquet")

    X_tr_parts, y_tr_parts = [], []
    X_te_parts, y_te_parts = [], []

    for ticker in train_df["ticker"].unique():
        train_close = train_df[train_df["ticker"] == ticker]["close"].sort_index().values
        test_close  = test_df[test_df["ticker"] == ticker]["close"].sort_index().values

        mu, sigma = train_close.mean(), train_close.std() + 1e-8
        train_norm = (train_close - mu) / sigma
        test_norm  = (test_close  - mu) / sigma

        Xt, yt = _make_sequences(train_norm, lookback)
        Xv, yv = _make_sequences(test_norm,  lookback)
        X_tr_parts.append(Xt);  y_tr_parts.append(yt)
        X_te_parts.append(Xv);  y_te_parts.append(yv)

    # Shape: (N, lookback, 1)
    X_train = np.concatenate(X_tr_parts)[:, :, np.newaxis]
    y_train = np.concatenate(y_tr_parts)
    X_test  = np.concatenate(X_te_parts)[:, :, np.newaxis]
    y_test  = np.concatenate(y_te_parts)
    return X_train, y_train, X_test, y_test


def _train_eval(rnn_type: str, cfg: dict) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"model": rnn_type, "error": "torch not installed"}

    class RNNModel(nn.Module):
        def __init__(self, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = cls(
                input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    p      = cfg["models"][rnn_type]
    n_runs = cfg["benchmark"]["n_latency_runs"]
    X_train, y_train, X_test, y_test = _prepare(cfg)

    requested = cfg.get("device", "auto")
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"  [{rnn_type.upper()}] using device: {device}")

    loss_log_path = Path("results") / f"{rnn_type}_loss.jsonl"
    loss_log_path.parent.mkdir(exist_ok=True)
    loss_log_path.write_text("")  # signal that training has started

    X_tr = torch.tensor(X_train).to(device)
    y_tr = torch.tensor(y_train).to(device)
    X_te = torch.tensor(X_test).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=p["batch_size"], shuffle=True,
                        pin_memory=False)

    model     = RNNModel(p["hidden_size"], p["num_layers"], p["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(p["epochs"]):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        with loss_log_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "loss": epoch_loss}) + "\n")
        log.info(f"  [{rnn_type.upper()}] epoch {epoch + 1}/{p['epochs']}  loss={epoch_loss:.6f}")

    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            preds = model(X_te).cpu().numpy()
            latencies.append(time.perf_counter() - t0)

    result = compute_metrics(y_test, preds, latencies)
    result["model"] = rnn_type
    log.info(f"{rnn_type.upper()} — MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
    return result


def run_lstm(cfg: dict) -> dict:
    try:
        return _train_eval("lstm", cfg)
    except Exception as e:
        log.error(f"LSTM failed: {e}")
        return {"model": "lstm", "error": str(e)}


def run_gru(cfg: dict) -> dict:
    try:
        return _train_eval("gru", cfg)
    except Exception as e:
        log.error(f"GRU failed: {e}")
        return {"model": "gru", "error": str(e)}
