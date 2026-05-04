import os
import time
import logging
import numpy as np
import pandas as pd

from models.metrics import compute_metrics

log = logging.getLogger(__name__)

_TABLE = "ts_ohlcv"
_MODEL = "ts_forecast_model"

_COLS = [
    "date", "ticker",
    "open", "high", "low", "close", "volume",
    "return_1d", "return_5d",
    "sma_5", "sma_10", "sma_20",
    "std_5", "std_10", "std_20",
    "hl_spread", "vol_change_1d",
]


def run_heatwave(cfg: dict) -> dict:
    try:
        import mysql.connector
    except ImportError:
        return {"model": "heatwave", "error": "mysql-connector-python not installed"}

    n_runs = cfg["benchmark"]["n_latency_runs"]
    target = cfg["dataset"]["target_col"]

    try:
        conn = mysql.connector.connect(
            host=os.environ["HW_HOST"],
            port=int(os.environ.get("HW_PORT", 3306)),
            user=os.environ["HW_USER"],
            password=os.environ["HW_PASSWORD"],
            database=os.environ.get("HW_DATABASE", "bench"),
        )
        cur = conn.cursor()

        _ensure_table(cur)
        _load_data(cur, conn)
        _ensure_model(cur, conn, target)

        test_df = pd.read_parquet("data/processed/test.parquet")
        actual, preds, latencies = _score(cur, test_df, target, n_runs)

        result = compute_metrics(actual, preds, latencies)
        result["model"] = "heatwave"
        log.info(f"HeatWave — MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")

        cur.close()
        conn.close()
        return result

    except Exception as e:
        log.error(f"HeatWave failed: {e}")
        return {"model": "heatwave", "error": str(e)}


def _ensure_table(cur) -> None:
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS `{_TABLE}` (
            id            BIGINT AUTO_INCREMENT PRIMARY KEY,
            date          DATE         NOT NULL,
            ticker        VARCHAR(10)  NOT NULL,
            open          DOUBLE,
            high          DOUBLE,
            low           DOUBLE,
            close         DOUBLE,
            volume        BIGINT,
            return_1d     DOUBLE,
            return_5d     DOUBLE,
            sma_5         DOUBLE,
            sma_10        DOUBLE,
            sma_20        DOUBLE,
            std_5         DOUBLE,
            std_10        DOUBLE,
            std_20        DOUBLE,
            hl_spread     DOUBLE,
            vol_change_1d DOUBLE,
            INDEX idx_ticker_date (ticker, date)
        )
    """)


def _load_data(cur, conn) -> None:
    cur.execute(f"SELECT COUNT(*) FROM `{_TABLE}`")
    if cur.fetchone()[0] > 0:
        log.info(f"Table '{_TABLE}' already populated, skipping load.")
        return

    df = pd.read_csv("data/processed/heatwave_import.csv")
    available = [c for c in _COLS if c in df.columns]
    placeholders = ", ".join(["%s"] * len(available))
    col_list = ", ".join(f"`{c}`" for c in available)

    rows = df[available].where(pd.notna(df[available]), None).values.tolist()
    cur.executemany(
        f"INSERT INTO `{_TABLE}` ({col_list}) VALUES ({placeholders})", rows
    )
    conn.commit()
    log.info(f"Loaded {len(rows):,} rows into '{_TABLE}'.")


def _ensure_model(cur, conn, target: str) -> None:
    try:
        cur.execute(f"CALL sys.ML_MODEL_LOAD('{_MODEL}', NULL)")
        log.info(f"HeatWave model '{_MODEL}' already loaded.")
        return
    except Exception:
        pass

    log.info(f"Training HeatWave AutoML model '{_MODEL}' ...")
    cur.execute(f"""
        CALL sys.ML_TRAIN(
            '{_TABLE}',
            '{target}',
            JSON_OBJECT(
                'task',              'forecasting',
                'datetime_index',    'date',
                'forecast_horizon',  1
            ),
            '{_MODEL}'
        )
    """)
    conn.commit()
    cur.execute(f"CALL sys.ML_MODEL_LOAD('{_MODEL}', NULL)")
    log.info(f"HeatWave model '{_MODEL}' trained and loaded.")


def _score(cur, test_df: pd.DataFrame, target: str, n_runs: int) -> tuple:
    feature_json = (
        "JSON_OBJECT("
        "'open', open, 'high', high, 'low', low, 'close', close, "
        "'volume', volume, 'return_1d', return_1d, 'sma_5', sma_5"
        ")"
    )

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        cur.execute(f"""
            SELECT sys.ML_PREDICT_ROW({feature_json}, '{_MODEL}', NULL) AS prediction
            FROM `{_TABLE}`
            ORDER BY date DESC
            LIMIT 1000
        """)
        rows = cur.fetchall()
        latencies.append(time.perf_counter() - t0)

    preds  = np.array([float(r[0]) if r[0] is not None else np.nan for r in rows])
    actual = test_df[target].values

    n = min(len(actual), len(preds))
    preds = np.nan_to_num(preds[:n], nan=float(np.nanmean(preds)))
    return actual[:n], preds, latencies
