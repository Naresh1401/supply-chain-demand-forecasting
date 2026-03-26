"""
FastAPI application for demand forecast serving.

Endpoints:
- POST /forecast          - Generate forecast for SKU + horizon
- GET  /forecast/{sku_id} - Get forecast for a specific SKU
- GET  /skus              - List available SKUs
- GET  /health            - Health check
- GET  /model/performance - Return model metrics
"""

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

app = FastAPI(
    title="Supply Chain Demand Forecasting API",
    description="Multi-horizon demand forecasting for supply chain optimization",
    version="1.0.0",
)

# ---------- Pydantic Models ----------


class ForecastRequest(BaseModel):
    sku_id: str = Field(..., description="SKU identifier, e.g. SKU_000")
    horizon: int = Field(default=7, ge=1, le=90, description="Forecast horizon in days")


class ForecastPoint(BaseModel):
    date: str
    forecast: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class ForecastResponse(BaseModel):
    sku_id: str
    horizon: int
    model_name: str
    forecasts: list[ForecastPoint]


class SKUInfo(BaseModel):
    sku_id: str
    category: str
    avg_daily_demand: float
    last_date: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool


class PerformanceResponse(BaseModel):
    metrics: dict


# ---------- Data / Model Loading ----------


def _load_data():
    """Load processed feature data."""
    path = os.path.join(PROCESSED_DIR, "features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])


def _load_model():
    """Load best trained model."""
    path = os.path.join(MODELS_DIR, "best_ml_model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _load_metrics():
    """Load model performance metrics."""
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# Lazy-loaded globals
_data_cache = None
_model_cache = None


def _get_data():
    global _data_cache
    if _data_cache is None:
        _data_cache = _load_data()
    return _data_cache


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = _load_model()
    return _model_cache


# ---------- Forecast Logic ----------


def _generate_forecast(sku_id: str, horizon: int) -> ForecastResponse:
    """Generate forecast for a given SKU."""
    df = _get_data()
    model = _get_model()

    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Run pipeline first.")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    sku_df = df[df["sku_id"] == sku_id]
    if len(sku_df) == 0:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found.")

    sku_df = sku_df.sort_values("date")

    # Use the last `horizon` rows as "future" (for demo; in prod these would be constructed)
    feature_cols = model.feature_cols
    test_rows = sku_df.tail(horizon).copy()
    test_clean = test_rows.dropna(subset=feature_cols)

    if len(test_clean) == 0:
        raise HTTPException(status_code=422, detail="Insufficient data for forecast.")

    preds = model.predict(test_clean)

    # Estimate uncertainty from recent residuals
    recent = sku_df.tail(90).dropna(subset=feature_cols)
    if len(recent) > 10:
        recent_preds = model.predict(recent)
        residual_std = float(np.std(recent["units_sold"].values - recent_preds))
    else:
        residual_std = float(np.std(sku_df["units_sold"].values)) * 0.2

    forecasts = []
    for i, (_, row) in enumerate(test_clean.iterrows()):
        pred = float(preds[i])
        forecasts.append(
            ForecastPoint(
                date=str(row["date"].date()),
                forecast=round(max(0, pred), 1),
                lower_80=round(max(0, pred - 1.28 * residual_std), 1),
                upper_80=round(max(0, pred + 1.28 * residual_std), 1),
                lower_95=round(max(0, pred - 1.96 * residual_std), 1),
                upper_95=round(max(0, pred + 1.96 * residual_std), 1),
            )
        )

    return ForecastResponse(
        sku_id=sku_id,
        horizon=len(forecasts),
        model_name=model.name,
        forecasts=forecasts,
    )


# ---------- Endpoints ----------


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_get_model() is not None,
        data_loaded=_get_data() is not None,
    )


@app.get("/skus", response_model=list[SKUInfo])
def list_skus():
    """List all available SKUs."""
    df = _get_data()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    skus = []
    for sku_id in sorted(df["sku_id"].unique()):
        sku_df = df[df["sku_id"] == sku_id]
        skus.append(
            SKUInfo(
                sku_id=sku_id,
                category=sku_df["category"].iloc[0],
                avg_daily_demand=round(float(sku_df["units_sold"].mean()), 1),
                last_date=str(sku_df["date"].max().date()),
            )
        )
    return skus


@app.post("/forecast", response_model=ForecastResponse)
def create_forecast(request: ForecastRequest):
    """Generate forecast for a given SKU and horizon."""
    return _generate_forecast(request.sku_id, request.horizon)


@app.get("/forecast/{sku_id}", response_model=ForecastResponse)
def get_forecast(sku_id: str, horizon: int = Query(default=7, ge=1, le=90)):
    """Get forecast for a specific SKU."""
    return _generate_forecast(sku_id, horizon)


@app.get("/model/performance", response_model=PerformanceResponse)
def model_performance():
    """Return model performance metrics."""
    metrics = _load_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found. Run training first.")
    return PerformanceResponse(metrics=metrics)
