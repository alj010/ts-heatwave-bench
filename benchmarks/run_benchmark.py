"""
benchmarks/run_benchmark.py
----------------------------
Orchestrates the full benchmark:
  1. Python baselines (ARIMA, XGBoost, LightGBM, LSTM, GRU)
  2. MySQL HeatWave (skipped gracefully if not configured)
  3. Saves results to JSON + prints a summary table

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --skip-heatwave
    python benchmarks/run_benchmark.py --models xgboost lightgbm
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "configs/benchmark.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_data():
    required = ["data/processed/train.parquet", "data/processed/test.parquet"]
    missing  = [p for p in required if not Path(p).exists()]
    if missing:
        log.error(f"Missing data files: {missing}")
        log.error("Run first: python scripts/fetch_data.py")
        sys.exit(1)


def run_all(cfg: dict, selected: list[str], skip_heatwave: bool) -> list[dict]:
    results = []

    if "arima" in selected:
        log.info("── ARIMA ─────────────────────────────────────")
        from models.baselines.arima_model import run_arima
        results.append(run_arima(cfg))

    if "xgboost" in selected:
        log.info("── XGBoost ───────────────────────────────────")
        from models.baselines.gbm_models import run_xgboost
        results.append(run_xgboost(cfg))

    if "lightgbm" in selected:
        log.info("── LightGBM ──────────────────────────────────")
        from models.baselines.gbm_models import run_lightgbm
        results.append(run_lightgbm(cfg))

    if "lstm" in selected:
        log.info("── LSTM ──────────────────────────────────────")
        from models.baselines.rnn_models import run_lstm
        results.append(run_lstm(cfg))

    if "gru" in selected:
        log.info("── GRU ───────────────────────────────────────")
        from models.baselines.rnn_models import run_gru
        results.append(run_gru(cfg))

    if "heatwave" in selected and not skip_heatwave:
        if not os.environ.get("HW_HOST"):
            log.warning("HW_HOST not set — skipping HeatWave. "
                        "Copy .env.example → .env and fill in credentials.")
        else:
            log.info("── HeatWave AutoML ───────────────────────────")
            from models.heatwave.hw_model import run_heatwave
            results.append(run_heatwave(cfg))

    return results


def save_results(results: list[dict], out_dir: str = "results") -> Path:
    Path(out_dir).mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"benchmark_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {path}")
    return path


def print_summary(results: list[dict]) -> None:
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["model"], "ERROR", "—", "—", "—", r["error"][:40]])
        else:
            rows.append([
                r["model"],
                f"{r.get('mae',  'N/A'):.4f}",
                f"{r.get('rmse', 'N/A'):.4f}",
                f"{r.get('mape', 'N/A'):.2f}%",
                f"{r.get('dir_accuracy', 'N/A'):.1f}%",
                f"{r.get('mean_latency_ms', 'N/A'):.2f}ms",
            ])

    headers = ["Model", "MAE", "RMSE", "MAPE", "Dir. Acc.", "Latency (mean)"]
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print()


ALL_MODELS = ["arima", "xgboost", "lightgbm", "lstm", "gru", "heatwave"]


def main():
    parser = argparse.ArgumentParser(description="ts-heatwave-bench runner")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS,
                        default=ALL_MODELS, help="Which models to run")
    parser.add_argument("--skip-heatwave", action="store_true",
                        help="Skip HeatWave even if configured")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_data()

    log.info(f"Running benchmark for models: {args.models}")
    results = run_all(cfg, args.models, args.skip_heatwave)

    save_results(results, cfg["benchmark"]["results_dir"])
    print_summary(results)


if __name__ == "__main__":
    main()
