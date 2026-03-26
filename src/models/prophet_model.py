"""
Facebook Prophet model wrapper for demand forecasting.

Features:
- Custom weekly and annual seasonalities
- Holiday effects (US holidays)
- External regressors: promotion, temperature, price
- Uncertainty intervals at 80% and 95% confidence
"""

import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
import holidays

warnings.filterwarnings("ignore")


class ProphetModel:
    """Prophet model with custom seasonalities and regressors."""

    def __init__(
        self,
        yearly_seasonality: int = 10,
        weekly_seasonality: int = 5,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.95,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.model = None
        self.name = "Prophet"
        self._has_regressors = False

    def _build_prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to Prophet format (ds, y) with regressors."""
        pdf = pd.DataFrame()
        pdf["ds"] = pd.to_datetime(df["date"])
        pdf["y"] = df["units_sold"].values.astype(float)

        # Add regressors if available
        if "promotion_flag" in df.columns:
            pdf["promotion"] = df["promotion_flag"].values.astype(float)
        if "temperature" in df.columns:
            pdf["temperature"] = df["temperature"].values.astype(float)
        if "price" in df.columns:
            pdf["price"] = df["price"].values.astype(float)

        return pdf

    def _build_future_df(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Build future dataframe with regressor values."""
        pdf = pd.DataFrame()
        pdf["ds"] = pd.to_datetime(future_df["date"])

        if "promotion_flag" in future_df.columns:
            pdf["promotion"] = future_df["promotion_flag"].values.astype(float)
        else:
            pdf["promotion"] = 0.0

        if "temperature" in future_df.columns:
            pdf["temperature"] = future_df["temperature"].values.astype(float)
        else:
            pdf["temperature"] = 65.0

        if "price" in future_df.columns:
            pdf["price"] = future_df["price"].values.astype(float)
        else:
            pdf["price"] = 0.0

        return pdf

    def fit(self, train_df: pd.DataFrame):
        """Fit Prophet model."""
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
        )

        # Add US holidays
        self.model.add_country_holidays(country_name="US")

        # Add regressors
        if "promotion_flag" in train_df.columns:
            self.model.add_regressor("promotion", mode="multiplicative")
            self._has_regressors = True
        if "temperature" in train_df.columns:
            self.model.add_regressor("temperature", mode="additive")
            self._has_regressors = True
        if "price" in train_df.columns:
            self.model.add_regressor("price", mode="multiplicative")
            self._has_regressors = True

        pdf = self._build_prophet_df(train_df)
        self.model.fit(pdf)
        return self

    def predict(self, horizon: int = None, future_df: pd.DataFrame = None) -> np.ndarray:
        """Generate point forecasts."""
        forecast = self._forecast(horizon, future_df)
        return np.maximum(0, forecast["yhat"].values)

    def predict_with_intervals(
        self, horizon: int = None, future_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Generate forecasts with confidence intervals."""
        forecast = self._forecast(horizon, future_df)
        result = pd.DataFrame(
            {
                "date": forecast["ds"],
                "forecast": np.maximum(0, forecast["yhat"].values),
                "lower_80": np.maximum(0, forecast["yhat_lower"].values),
                "upper_80": np.maximum(0, forecast["yhat_upper"].values),
            }
        )
        # Approximate 80% interval from 95%
        center = forecast["yhat"].values
        width_95 = forecast["yhat_upper"].values - forecast["yhat_lower"].values
        width_80 = width_95 * (1.28 / 1.96)
        result["lower_80"] = np.maximum(0, center - width_80 / 2)
        result["upper_80"] = np.maximum(0, center + width_80 / 2)
        result["lower_95"] = np.maximum(0, forecast["yhat_lower"].values)
        result["upper_95"] = np.maximum(0, forecast["yhat_upper"].values)
        return result

    def _forecast(self, horizon: int = None, future_df: pd.DataFrame = None) -> pd.DataFrame:
        """Internal forecast method."""
        if future_df is not None:
            future = self._build_future_df(future_df)
        else:
            future = self.model.make_future_dataframe(periods=horizon)
            if self._has_regressors:
                future["promotion"] = 0.0
                future["temperature"] = 65.0
                future["price"] = 0.0
            future = future.tail(horizon)

        forecast = self.model.predict(future)
        return forecast.tail(horizon if horizon else len(future_df))


def get_prophet_model():
    """Return a Prophet model instance."""
    return ProphetModel()
