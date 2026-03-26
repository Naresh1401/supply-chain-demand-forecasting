"""
Statistical baseline models for demand forecasting.

Models:
- NaiveModel: last observed value
- SeasonalNaiveModel: same day last week
- MovingAverageModel: simple moving average
- HoltWintersModel: exponential smoothing with trend + seasonality
- SARIMAModel: seasonal ARIMA with automatic parameter selection
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class NaiveModel:
    """Forecast = last observed value."""

    def __init__(self):
        self.last_value = None
        self.name = "Naive"

    def fit(self, train_df: pd.DataFrame):
        self.last_value = train_df["units_sold"].iloc[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_value)


class SeasonalNaiveModel:
    """Forecast = same weekday from last week."""

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.last_season = None
        self.name = "Seasonal Naive"

    def fit(self, train_df: pd.DataFrame):
        self.last_season = train_df["units_sold"].iloc[-self.season_length :].values
        return self

    def predict(self, horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / self.season_length))
        tiled = np.tile(self.last_season, reps)
        return tiled[:horizon]


class MovingAverageModel:
    """Forecast = average of last `window` observations."""

    def __init__(self, window: int = 7):
        self.window = window
        self.avg = None
        self.name = "Moving Average"

    def fit(self, train_df: pd.DataFrame):
        self.avg = train_df["units_sold"].iloc[-self.window :].mean()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.avg)


class HoltWintersModel:
    """Exponential Smoothing with trend and seasonal components."""

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted = None
        self.name = "Holt-Winters"

    def fit(self, train_df: pd.DataFrame):
        y = train_df["units_sold"].values.astype(float)
        # Ensure enough data for seasonal decomposition
        if len(y) < 2 * self.seasonal_periods:
            self.model = None
            self._fallback = y[-1]
            return self
        try:
            model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self.fitted = model.fit(optimized=True, use_brute=False)
        except Exception:
            # Fallback: no seasonality
            try:
                model = ExponentialSmoothing(y, trend="add", initialization_method="estimated")
                self.fitted = model.fit(optimized=True)
            except Exception:
                self.fitted = None
                self._fallback = y[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.fitted is None:
            return np.full(horizon, self._fallback)
        forecast = self.fitted.forecast(horizon)
        return np.maximum(0, forecast)


class SARIMAModel:
    """Seasonal ARIMA with fixed parameters for speed."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted = None
        self.name = "SARIMA"

    def fit(self, train_df: pd.DataFrame):
        y = train_df["units_sold"].values.astype(float)
        if len(y) < 30:
            self._fallback = y[-1] if len(y) > 0 else 0
            return self
        try:
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted = model.fit(disp=False, maxiter=100)
        except Exception:
            # Fallback to simpler model
            try:
                model = SARIMAX(
                    y,
                    order=(1, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.fitted = model.fit(disp=False, maxiter=50)
            except Exception:
                self.fitted = None
                self._fallback = y[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.fitted is None:
            return np.full(horizon, self._fallback)
        forecast = self.fitted.forecast(steps=horizon)
        return np.maximum(0, forecast)


def get_all_statistical_models():
    """Return a list of all statistical model instances."""
    return [
        NaiveModel(),
        SeasonalNaiveModel(season_length=7),
        MovingAverageModel(window=7),
        HoltWintersModel(seasonal_periods=7),
        SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)),
    ]
