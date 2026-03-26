"""
Evaluation and visualization for demand forecasting models.

Generates:
- Actual vs Predicted line plots (per SKU and aggregate)
- Error distribution histograms
- Model comparison heatmap
- Forecast uncertainty fan charts
- Residual analysis (ACF, normality)
- Demand decomposition plots (trend + seasonality + residual)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib

from src.models.ml_models import _get_feature_columns

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#607D8B"]


def load_data():
    """Load processed data and trained model."""
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "features.csv"), parse_dates=["date"]
    )
    return df


def load_metrics():
    """Load saved metrics."""
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_best_model():
    """Load best ML model."""
    path = os.path.join(MODELS_DIR, "best_ml_model.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def plot_actual_vs_predicted(df: pd.DataFrame, model, sku_id: str):
    """Plot actual vs predicted for a single SKU."""
    sku_df = df[df["sku_id"] == sku_id].copy().sort_values("date")

    # Use last 90 days as test
    cutoff = sku_df["date"].max() - pd.Timedelta(days=90)
    train = sku_df[sku_df["date"] <= cutoff]
    test = sku_df[sku_df["date"] > cutoff]

    if model is None or len(test) == 0:
        return

    feature_cols = _get_feature_columns(df)
    test_clean = test.dropna(subset=feature_cols)
    if len(test_clean) == 0:
        return

    preds = model.predict(test_clean)

    fig, ax = plt.subplots(figsize=(14, 5))
    # Show last 60 days of training + test
    train_tail = train.tail(60)
    ax.plot(train_tail["date"], train_tail["units_sold"], color=COLORS[0], alpha=0.6, label="Train (actual)")
    ax.plot(test_clean["date"], test_clean["units_sold"], color=COLORS[0], linewidth=2, label="Test (actual)")
    ax.plot(test_clean["date"], preds, color=COLORS[1], linewidth=2, linestyle="--", label=f"Predicted ({model.name})")
    ax.axvline(cutoff, color="gray", linestyle=":", alpha=0.7, label="Train/Test split")
    ax.set_title(f"Actual vs Predicted — {sku_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"actual_vs_pred_{sku_id}.png"), dpi=150)
    plt.close()


def plot_aggregate_actual_vs_predicted(df: pd.DataFrame, model):
    """Plot aggregate (total) actual vs predicted."""
    if model is None:
        return

    cutoff = df["date"].max() - pd.Timedelta(days=90)
    test = df[df["date"] > cutoff].copy()
    feature_cols = _get_feature_columns(df)
    test_clean = test.dropna(subset=feature_cols)

    if len(test_clean) == 0:
        return

    preds = model.predict(test_clean)
    test_clean = test_clean.copy()
    test_clean["predicted"] = preds

    agg_actual = test_clean.groupby("date")["units_sold"].sum()
    agg_pred = test_clean.groupby("date")["predicted"].sum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(agg_actual.index, agg_actual.values, color=COLORS[0], linewidth=2, label="Actual")
    ax.plot(agg_pred.index, agg_pred.values, color=COLORS[1], linewidth=2, linestyle="--", label="Predicted")
    ax.set_title("Aggregate Demand: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "aggregate_actual_vs_pred.png"), dpi=150)
    plt.close()


def plot_error_distribution(df: pd.DataFrame, model):
    """Plot error distribution histogram."""
    if model is None:
        return

    cutoff = df["date"].max() - pd.Timedelta(days=90)
    test = df[df["date"] > cutoff].copy()
    feature_cols = _get_feature_columns(df)
    test_clean = test.dropna(subset=feature_cols)

    if len(test_clean) == 0:
        return

    preds = model.predict(test_clean)
    errors = test_clean["units_sold"].values - preds

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(errors, bins=50, color=COLORS[0], alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Forecast Error Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Error (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")

    pct_errors = errors / (test_clean["units_sold"].values + 1) * 100
    axes[1].hist(pct_errors, bins=50, color=COLORS[2], alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Percentage Error Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Error (%)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "error_distribution.png"), dpi=150)
    plt.close()


def plot_model_comparison_heatmap(metrics: dict):
    """Plot model comparison heatmap (metrics x models)."""
    if not metrics:
        return

    metric_names = ["MAE", "RMSE", "MAPE", "WAPE", "sMAPE"]
    model_names = list(metrics.keys())

    data = []
    for model in model_names:
        row = [metrics[model].get(m, np.nan) for m in metric_names]
        data.append(row)

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(4, len(model_names) * 0.6)))
    sns.heatmap(
        data_arr,
        annot=True,
        fmt=".1f",
        xticklabels=metric_names,
        yticklabels=model_names,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Model Comparison Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_heatmap.png"), dpi=150)
    plt.close()


def plot_forecast_fan_chart(df: pd.DataFrame, model, sku_id: str):
    """Plot forecast with uncertainty fan chart."""
    if model is None:
        return

    sku_df = df[df["sku_id"] == sku_id].copy().sort_values("date")
    cutoff = sku_df["date"].max() - pd.Timedelta(days=90)
    train = sku_df[sku_df["date"] <= cutoff]
    test = sku_df[sku_df["date"] > cutoff]

    feature_cols = _get_feature_columns(df)
    test_clean = test.dropna(subset=feature_cols)
    if len(test_clean) == 0:
        return

    preds = model.predict(test_clean)
    residuals = train["units_sold"].values[-len(preds):] - preds[:len(train["units_sold"].values[-len(preds):])] if len(train) > len(preds) else np.zeros(10)

    # Estimate prediction intervals from training residuals
    train_clean = train.dropna(subset=feature_cols)
    if len(train_clean) > 0:
        train_preds = model.predict(train_clean)
        train_residuals = train_clean["units_sold"].values - train_preds
        std = np.std(train_residuals)
    else:
        std = np.std(test_clean["units_sold"].values) * 0.2

    dates = test_clean["date"].values

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.tail(30)["date"], train.tail(30)["units_sold"], color=COLORS[0], alpha=0.5)
    ax.plot(dates, test_clean["units_sold"].values, color=COLORS[0], linewidth=2, label="Actual")
    ax.plot(dates, preds, color=COLORS[1], linewidth=2, label="Forecast")
    ax.fill_between(dates, preds - 1.28 * std, preds + 1.28 * std, alpha=0.3, color=COLORS[1], label="80% CI")
    ax.fill_between(dates, preds - 1.96 * std, preds + 1.96 * std, alpha=0.15, color=COLORS[1], label="95% CI")
    ax.set_title(f"Forecast Fan Chart — {sku_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"fan_chart_{sku_id}.png"), dpi=150)
    plt.close()


def plot_residual_analysis(df: pd.DataFrame, model, sku_id: str):
    """Plot residual analysis: ACF and normality."""
    if model is None:
        return

    sku_df = df[df["sku_id"] == sku_id].copy().sort_values("date")
    cutoff = sku_df["date"].max() - pd.Timedelta(days=90)
    test = sku_df[sku_df["date"] > cutoff]
    feature_cols = _get_feature_columns(df)
    test_clean = test.dropna(subset=feature_cols)

    if len(test_clean) < 10:
        return

    preds = model.predict(test_clean)
    residuals = test_clean["units_sold"].values - preds

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Residuals over time
    axes[0].plot(test_clean["date"].values, residuals, color=COLORS[0], linewidth=1)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residuals Over Time", fontweight="bold")
    axes[0].set_xlabel("Date")

    # ACF of residuals
    plot_acf(residuals, lags=min(30, len(residuals) - 1), ax=axes[1], color=COLORS[2])
    axes[1].set_title("Residual ACF", fontweight="bold")

    # QQ-style histogram
    axes[2].hist(residuals, bins=30, density=True, color=COLORS[4], alpha=0.7, edgecolor="white")
    x = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm
    mu, sigma = norm.fit(residuals)
    axes[2].plot(x, norm.pdf(x, mu, sigma), color="red", linewidth=2)
    axes[2].set_title("Residual Distribution", fontweight="bold")
    axes[2].set_xlabel("Residual")

    plt.suptitle(f"Residual Analysis — {sku_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"residual_analysis_{sku_id}.png"), dpi=150)
    plt.close()


def plot_demand_decomposition(df: pd.DataFrame, sku_id: str):
    """Plot STL-style demand decomposition."""
    sku_df = df[df["sku_id"] == sku_id].copy().sort_values("date")
    ts = sku_df.set_index("date")["units_sold"]

    if len(ts) < 60:
        return

    try:
        decomp = seasonal_decompose(ts, model="additive", period=7, extrapolate_trend="freq")
    except Exception:
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(decomp.observed, color=COLORS[0])
    axes[0].set_ylabel("Observed")
    axes[0].set_title(f"Demand Decomposition — {sku_id}", fontsize=14, fontweight="bold")

    axes[1].plot(decomp.trend, color=COLORS[2])
    axes[1].set_ylabel("Trend")

    axes[2].plot(decomp.seasonal, color=COLORS[3])
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(decomp.resid, color=COLORS[4], alpha=0.7)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"decomposition_{sku_id}.png"), dpi=150)
    plt.close()


def plot_category_comparison(df: pd.DataFrame):
    """Plot average demand by category over time."""
    cat_daily = df.groupby(["date", "category"])["units_sold"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, cat in enumerate(sorted(cat_daily["category"].unique())):
        cat_data = cat_daily[cat_daily["category"] == cat]
        # Weekly rolling average for smoothing
        rolling = cat_data.set_index("date")["units_sold"].rolling(7).mean()
        ax.plot(rolling.index, rolling.values, label=cat, color=COLORS[i % len(COLORS)], linewidth=1.5)

    ax.set_title("Average Daily Demand by Category (7-day rolling)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Units Sold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "category_demand.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("SUPPLY CHAIN DEMAND FORECASTING - EVALUATION")
    print("=" * 70)

    df = load_data()
    metrics = load_metrics()
    model = load_best_model()

    skus = sorted(df["sku_id"].unique())
    sample_skus = skus[:3]

    print("\nGenerating visualizations...")

    # 1. Actual vs Predicted (per SKU)
    for sku in sample_skus:
        plot_actual_vs_predicted(df, model, sku)
        print(f"  [✓] Actual vs Predicted — {sku}")

    # 2. Aggregate actual vs predicted
    plot_aggregate_actual_vs_predicted(df, model)
    print("  [✓] Aggregate Actual vs Predicted")

    # 3. Error distribution
    plot_error_distribution(df, model)
    print("  [✓] Error Distribution")

    # 4. Model comparison heatmap
    plot_model_comparison_heatmap(metrics)
    print("  [✓] Model Comparison Heatmap")

    # 5. Fan charts
    for sku in sample_skus:
        plot_forecast_fan_chart(df, model, sku)
        print(f"  [✓] Fan Chart — {sku}")

    # 6. Residual analysis
    for sku in sample_skus:
        plot_residual_analysis(df, model, sku)
        print(f"  [✓] Residual Analysis — {sku}")

    # 7. Demand decomposition
    for sku in sample_skus:
        plot_demand_decomposition(df, sku)
        print(f"  [✓] Decomposition — {sku}")

    # 8. Category comparison
    plot_category_comparison(df)
    print("  [✓] Category Demand Comparison")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
