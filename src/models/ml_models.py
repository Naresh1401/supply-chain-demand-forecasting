"""
Machine learning models for demand forecasting.

Models:
- RidgeModel: L2-regularized linear regression
- RandomForestModel: ensemble of decision trees
- XGBoostModel: gradient boosted trees
- LightGBMModel: gradient boosted trees (histogram-based)

All models use the engineered tabular features and support
recursive multi-step forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Features that should not be used as model input
EXCLUDE_COLS = ["date", "sku_id", "category", "units_sold", "revenue"]

# Lag/rolling features that need recursive updating
LAG_COLS = [c for c in [f"lag_{d}" for d in [1, 7, 14, 28]]]
ROLLING_MEAN_COLS = [f"rolling_mean_{w}" for w in [7, 14, 28]]


def _get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature column names (everything except target and IDs)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


class BaseMLModel:
    """Base class for ML forecasting models."""

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.name = "BaseML"

    def fit(self, train_df: pd.DataFrame):
        """Fit model on training data."""
        self.feature_cols = _get_feature_columns(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["units_sold"].values
        self.model.fit(X, y)
        return self

    def predict_single(self, X_row: np.ndarray) -> float:
        """Predict a single step."""
        return max(0, float(self.model.predict(X_row.reshape(1, -1))[0]))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict on test dataframe (direct, non-recursive)."""
        X = test_df[self.feature_cols].values
        preds = self.model.predict(X)
        return np.maximum(0, preds)

    def predict_recursive(self, last_train_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
        """
        Recursive multi-step forecast.

        Uses the last training observations to initialize lags,
        then predicts one step at a time, updating lag features
        with the predicted values.
        """
        horizon = len(future_df)
        preds = np.zeros(horizon)

        # Build a buffer of recent actuals for lag computation
        recent = last_train_df["units_sold"].values[-28:].tolist()

        for i in range(horizon):
            row = future_df.iloc[i].copy()
            # Update lag features from buffer
            if "lag_1" in self.feature_cols and len(recent) >= 1:
                row["lag_1"] = recent[-1]
            if "lag_7" in self.feature_cols and len(recent) >= 7:
                row["lag_7"] = recent[-7]
            if "lag_14" in self.feature_cols and len(recent) >= 14:
                row["lag_14"] = recent[-14]
            if "lag_28" in self.feature_cols and len(recent) >= 28:
                row["lag_28"] = recent[-28]

            # Update rolling means
            for w in [7, 14, 28]:
                col = f"rolling_mean_{w}"
                if col in self.feature_cols and len(recent) >= w:
                    row[col] = np.mean(recent[-w:])
                col_std = f"rolling_std_{w}"
                if col_std in self.feature_cols and len(recent) >= w:
                    row[col_std] = np.std(recent[-w:])

            # EWM approximation
            for span in [7, 14, 28]:
                col = f"ewm_{span}"
                if col in self.feature_cols and len(recent) >= 1:
                    alpha = 2 / (span + 1)
                    vals = recent[-span:]
                    ewm_val = vals[0]
                    for v in vals[1:]:
                        ewm_val = alpha * v + (1 - alpha) * ewm_val
                    row[col] = ewm_val

            X_row = row[self.feature_cols].values.astype(float)
            pred = self.predict_single(X_row)
            preds[i] = pred
            recent.append(pred)

        return preds

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance (if available)."""
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_)
        else:
            return pd.DataFrame()
        return pd.DataFrame(
            {"feature": self.feature_cols, "importance": imp}
        ).sort_values("importance", ascending=False)


class RidgeModel(BaseMLModel):
    """Ridge regression (L2 regularized)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.model = Ridge(alpha=alpha)
        self.name = "Ridge"


class RandomForestModel(BaseMLModel):
    """Random Forest regressor."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 15, random_state: int = 42):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=random_state,
        )
        self.name = "Random Forest"


class XGBoostModel(BaseMLModel):
    """XGBoost regressor."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 8, learning_rate: float = 0.05, random_state: int = 42):
        super().__init__()
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        )
        self.name = "XGBoost"


class LightGBMModel(BaseMLModel):
    """LightGBM regressor."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 8, learning_rate: float = 0.05, random_state: int = 42):
        super().__init__()
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
        self.name = "LightGBM"


def get_all_ml_models():
    """Return a list of all ML model instances."""
    return [
        RidgeModel(alpha=1.0),
        RandomForestModel(),
        XGBoostModel(),
        LightGBMModel(),
    ]
