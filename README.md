# ts-heatwave-bench

A benchmarking study comparing **in-database ML forecasting** with MySQL HeatWave against external Python baselines, evaluating tradeoffs in latency, accuracy, and scalability for time series prediction tasks.

## Overview

This project constructs an end-to-end time series forecasting pipeline inside MySQL HeatWave and benchmarks it against equivalent models built with standard Python ML libraries. The goal is to systematically quantify the practical tradeoffs of database-native ML inference versus external compute.

## Benchmarks

### In-Database
- MySQL HeatWave ML (AutoML forecasting)

### External Python Baselines
- XGBoost / LightGBM
- LSTM / GRU (PyTorch)
- ARIMA / statsmodels

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE / RMSE | Forecast accuracy |
| Inference latency | End-to-end prediction time |
| Throughput | Predictions per second |
| Scalability | Performance under increasing data volume |

## Stack

- MySQL HeatWave
- Python, Pandas, NumPy
- PyTorch, XGBoost, LightGBM
- statsmodels
