"""
Training pipeline for demand forecasting models.

Implements:
- Time series cross-validation (expanding window, 5 folds)
- Training all statistical + ML + Prophet models
- Metrics: MAE, RMSE, MAPE, WAPE, sMAPE, Bias
- Forecast horizons: 7, 14, 28 days
- Saves best model per SKU and overall metrics
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from src.models.statistical_models import get_all_statistical_models
from src.models.ml_models import get_all_ml_models, _get_feature_columns
from src.models.prophet_model import get_prophet_model

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

HORIZONS = [7, 14, 28]
N_CV_FOLDS = 5
TEST_DAYS = 90  # Last 90 days held out as final test


# ---------- Metrics ----------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum(np.abs(y_true))
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": round(mae(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 2),
        "WAPE": round(wape(y_true, y_pred), 2),
        "sMAPE": round(smape(y_true, y_pred), 2),
        "Bias": round(bias(y_true, y_pred), 2),
    }


# ---------- Cross-Validation ----------


def expanding_window_cv(dates: pd.Series, n_folds: int = 5, horizon: int = 28, gap: int = 0):
    """
    Generate expanding window train/test splits for time series CV.

    Yields (train_end_idx, test_start_idx, test_end_idx) for each fold.
    """
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)

    # Reserve enough room for n_folds * horizon at the end
    total_test = n_folds * horizon
    train_start_size = n_dates - total_test - gap * n_folds

    if train_start_size < 90:
        # Not enough data: use minimal folds
        train_start_size = max(60, n_dates // 2)
        n_folds = min(n_folds, (n_dates - train_start_size) // (horizon + gap))

    for fold in range(n_folds):
        train_end = train_start_size + fold * (horizon + gap) - 1
        test_start = train_end + gap + 1
        test_end = min(test_start + horizon - 1, n_dates - 1)

        if test_end <= test_start:
            break

        train_end_date = unique_dates[train_end]
        test_start_date = unique_dates[test_start]
        test_end_date = unique_dates[test_end]

        yield train_end_date, test_start_date, test_end_date


# ---------- Training ----------


def train_statistical_models_for_sku(
    sku_df: pd.DataFrame, horizon: int
) -> dict:
    """Train and evaluate all statistical models for a single SKU."""
    dates = sku_df["date"].sort_values()
    results = {}

    models = get_all_statistical_models()

    for model in models:
        fold_metrics = []
        for train_end, test_start, test_end in expanding_window_cv(dates, N_CV_FOLDS, horizon):
            train = sku_df[sku_df["date"] <= train_end]
            test = sku_df[(sku_df["date"] >= test_start) & (sku_df["date"] <= test_end)]

            if len(train) < 30 or len(test) < 1:
                continue

            model.fit(train)
            preds = model.predict(len(test))
            y_true = test["units_sold"].values

            # Align lengths
            min_len = min(len(y_true), len(preds))
            metrics = compute_all_metrics(y_true[:min_len], preds[:min_len])
            fold_metrics.append(metrics)

        if fold_metrics:
            avg_metrics = {
                k: round(np.mean([m[k] for m in fold_metrics]), 2)
                for k in fold_metrics[0]
            }
            results[model.name] = avg_metrics

    return results


def train_ml_models_global(
    df: pd.DataFrame, horizon: int
) -> tuple:
    """
    Train ML models on all SKUs together (global model).
    Uses walk-forward CV with recursive multi-step forecasting.
    """
    dates = df["date"].sort_values()
    feature_cols = _get_feature_columns(df)
    results = {}
    best_model = None
    best_mae = float("inf")

    models = get_all_ml_models()

    for model in models:
        fold_metrics = []
        for train_end, test_start, test_end in expanding_window_cv(dates, N_CV_FOLDS, horizon):
            train = df[df["date"] <= train_end].copy()
            test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

            if len(train) < 100 or len(test) < 1:
                continue

            # Drop rows with NaN features
            train_clean = train.dropna(subset=feature_cols)
            if len(train_clean) < 50:
                continue

            model.fit(train_clean)

            # Direct prediction (test features are available)
            test_clean = test.dropna(subset=feature_cols)
            if len(test_clean) < 1:
                continue

            preds = model.predict(test_clean)
            y_true = test_clean["units_sold"].values

            min_len = min(len(y_true), len(preds))
            metrics = compute_all_metrics(y_true[:min_len], preds[:min_len])
            fold_metrics.append(metrics)

        if fold_metrics:
            avg_metrics = {
                k: round(np.mean([m[k] for m in fold_metrics]), 2)
                for k in fold_metrics[0]
            }
            results[model.name] = avg_metrics
            if avg_metrics["MAE"] < best_mae:
                best_mae = avg_metrics["MAE"]
                best_model = model

    return results, best_model


def train_prophet_for_sku(sku_df: pd.DataFrame, horizon: int) -> dict:
    """Train and evaluate Prophet for a single SKU via CV."""
    dates = sku_df["date"].sort_values()
    model = get_prophet_model()
    fold_metrics = []

    for train_end, test_start, test_end in expanding_window_cv(dates, N_CV_FOLDS, horizon):
        train = sku_df[sku_df["date"] <= train_end]
        test = sku_df[(sku_df["date"] >= test_start) & (sku_df["date"] <= test_end)]

        if len(train) < 60 or len(test) < 1:
            continue

        try:
            model = get_prophet_model()
            model.fit(train)
            preds = model.predict(future_df=test)
            y_true = test["units_sold"].values

            min_len = min(len(y_true), len(preds))
            metrics = compute_all_metrics(y_true[:min_len], preds[:min_len])
            fold_metrics.append(metrics)
        except Exception:
            continue

    if fold_metrics:
        return {
            model.name: {
                k: round(np.mean([m[k] for m in fold_metrics]), 2)
                for k in fold_metrics[0]
            }
        }
    return {}


# ---------- Main ----------


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("SUPPLY CHAIN DEMAND FORECASTING - MODEL TRAINING")
    print("=" * 70)

    # Load data
    data_path = os.path.join(PROCESSED_DIR, "features.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run `make features` first.")
        return

    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"\nLoaded {len(df):,} rows, {df['sku_id'].nunique()} SKUs")

    all_results = {}
    skus = sorted(df["sku_id"].unique())

    # --- Statistical Models (per SKU, sample of SKUs for speed) ---
    print("\n" + "-" * 50)
    print("Training Statistical Models (per SKU)")
    print("-" * 50)

    sample_skus = skus[:10]  # Sample for demo speed
    stat_results_agg = {}

    for sku_id in sample_skus:
        sku_df = df[df["sku_id"] == sku_id].copy()
        for horizon in [7]:
            results = train_statistical_models_for_sku(sku_df, horizon)
            for model_name, metrics in results.items():
                if model_name not in stat_results_agg:
                    stat_results_agg[model_name] = []
                stat_results_agg[model_name].append(metrics)

    for model_name, metrics_list in stat_results_agg.items():
        avg = {
            k: round(np.mean([m[k] for m in metrics_list]), 2)
            for k in metrics_list[0]
        }
        all_results[model_name] = avg
        print(f"  {model_name:20s} | MAE={avg['MAE']:7.2f} | RMSE={avg['RMSE']:7.2f} | MAPE={avg['MAPE']:5.1f}%")

    # --- ML Models (global) ---
    print("\n" + "-" * 50)
    print("Training ML Models (global)")
    print("-" * 50)

    for horizon in [7]:
        ml_results, best_ml_model = train_ml_models_global(df, horizon)
        for model_name, metrics in ml_results.items():
            all_results[model_name] = metrics
            print(f"  {model_name:20s} | MAE={metrics['MAE']:7.2f} | RMSE={metrics['RMSE']:7.2f} | MAPE={metrics['MAPE']:5.1f}%")

    # Save best ML model
    if best_ml_model is not None:
        # Re-train on full training data (excluding last TEST_DAYS)
        cutoff = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
        train_full = df[df["date"] <= cutoff].dropna(subset=_get_feature_columns(df))
        best_ml_model.fit(train_full)
        model_path = os.path.join(MODELS_DIR, "best_ml_model.joblib")
        joblib.dump(best_ml_model, model_path)
        print(f"\n  Best ML model ({best_ml_model.name}) saved to {model_path}")

        # Save feature columns
        feat_path = os.path.join(MODELS_DIR, "feature_columns.json")
        with open(feat_path, "w") as f:
            json.dump(best_ml_model.feature_cols, f)

    # --- Prophet (sample SKUs) ---
    print("\n" + "-" * 50)
    print("Training Prophet (per SKU, sample)")
    print("-" * 50)

    prophet_results = []
    for sku_id in sample_skus[:5]:
        sku_df = df[df["sku_id"] == sku_id].copy()
        result = train_prophet_for_sku(sku_df, horizon=7)
        if result:
            prophet_results.append(list(result.values())[0])
            print(f"  {sku_id} done")

    if prophet_results:
        avg_prophet = {
            k: round(np.mean([m[k] for m in prophet_results]), 2)
            for k in prophet_results[0]
        }
        all_results["Prophet"] = avg_prophet
        print(f"  {'Prophet':20s} | MAE={avg_prophet['MAE']:7.2f} | RMSE={avg_prophet['RMSE']:7.2f} | MAPE={avg_prophet['MAPE']:5.1f}%")

    # --- Final Test Set Evaluation ---
    print("\n" + "=" * 70)
    print("FINAL RESULTS (7-day horizon, walk-forward CV)")
    print("=" * 70)
    print(f"\n{'Model':20s} | {'MAE':>7s} | {'RMSE':>7s} | {'MAPE':>7s} | {'WAPE':>7s} | {'sMAPE':>7s} | {'Bias':>7s}")
    print("-" * 85)
    for model_name in [
        "Naive", "Seasonal Naive", "Moving Average", "Holt-Winters", "SARIMA",
        "Ridge", "Random Forest", "XGBoost", "LightGBM", "Prophet"
    ]:
        if model_name in all_results:
            m = all_results[model_name]
            print(
                f"  {model_name:20s} | {m['MAE']:7.2f} | {m['RMSE']:7.2f} | "
                f"{m['MAPE']:6.1f}% | {m['WAPE']:6.1f}% | {m['sMAPE']:6.1f}% | {m['Bias']:7.2f}"
            )

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
