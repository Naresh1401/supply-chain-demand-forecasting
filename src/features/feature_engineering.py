"""
Time series feature engineering for demand forecasting.

Creates 50+ features from raw sales data:
- Lag features (1, 7, 14, 28 days)
- Rolling statistics (7d, 14d, 28d mean, std, min, max)
- Exponential weighted moving averages
- Calendar features
- Price features
- Fourier terms for seasonality
- Target encoding for category
"""

import os
import numpy as np
import pandas as pd
import holidays

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def load_raw_data() -> pd.DataFrame:
    """Load raw sales data."""
    df = pd.read_csv(os.path.join(RAW_DIR, "sales_data.csv"), parse_dates=["date"])
    df.sort_values(["sku_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features grouped by SKU."""
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby("sku_id")["units_sold"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window statistics grouped by SKU."""
    for window in ROLLING_WINDOWS:
        grp = df.groupby("sku_id")["units_sold"]
        # Shift by 1 to prevent data leakage (use only past data)
        shifted = grp.shift(1)
        rolling = shifted.rolling(window=window, min_periods=1)
        df[f"rolling_mean_{window}"] = rolling.mean().values
        df[f"rolling_std_{window}"] = rolling.std().values
        df[f"rolling_min_{window}"] = rolling.min().values
        df[f"rolling_max_{window}"] = rolling.max().values
    return df


def add_ewm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add exponential weighted moving average features."""
    for span in [7, 14, 28]:
        shifted = df.groupby("sku_id")["units_sold"].shift(1)
        df[f"ewm_{span}"] = shifted.ewm(span=span, min_periods=1).mean().values
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features."""
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    # Holiday features
    us_holidays = holidays.US(years=df["date"].dt.year.unique().tolist())
    df["is_holiday"] = df["date"].apply(lambda d: 1 if d in us_holidays else 0)

    # Days until next holiday
    holiday_dates = sorted(us_holidays.keys())
    holiday_dates = pd.to_datetime(holiday_dates)

    def _days_until_holiday(d):
        future = holiday_dates[holiday_dates >= d]
        if len(future) == 0:
            return 365
        return (future[0] - d).days

    df["days_until_holiday"] = df["date"].apply(_days_until_holiday)
    df["days_until_holiday"] = df["days_until_holiday"].clip(upper=30)

    # Day of month
    df["day_of_month"] = df["date"].dt.day

    # Is month start / end
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-related features."""
    # Price change percentage (vs previous day for same SKU)
    df["price_lag1"] = df.groupby("sku_id")["price"].shift(1)
    df["price_change_pct"] = ((df["price"] - df["price_lag1"]) / df["price_lag1"]).fillna(0)
    df.drop(columns=["price_lag1"], inplace=True)

    # Discount depth
    df["discount_depth"] = df["promotion_discount"]

    # Price ratio vs competitor
    df["price_ratio_competitor"] = df["price"] / df["competitor_price"].replace(0, np.nan)
    df["price_ratio_competitor"] = df["price_ratio_competitor"].fillna(1.0)

    return df


def add_fourier_features(df: pd.DataFrame, n_terms: int = 4) -> pd.DataFrame:
    """Add Fourier terms to capture multiple seasonality."""
    day_of_year = df["date"].dt.dayofyear
    day_of_week = df["date"].dt.dayofweek

    # Annual seasonality via Fourier
    for k in range(1, n_terms + 1):
        df[f"annual_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        df[f"annual_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    # Weekly seasonality via Fourier
    for k in range(1, 3):
        df[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * day_of_week / 7)
        df[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * day_of_week / 7)

    return df


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add target encoding for category (using expanding mean to avoid leakage)."""
    df.sort_values(["date", "sku_id"], inplace=True)

    # Global expanding mean per category (shifted to avoid leakage)
    df["category_encoded"] = (
        df.groupby("category")["units_sold"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    # Fill NaN (first observation per category) with global mean
    global_mean = df["units_sold"].mean()
    df["category_encoded"] = df["category_encoded"].fillna(global_mean)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    print("  Adding lag features...")
    df = add_lag_features(df)

    print("  Adding rolling statistics...")
    df = add_rolling_features(df)

    print("  Adding exponential weighted moving averages...")
    df = add_ewm_features(df)

    print("  Adding calendar features...")
    df = add_calendar_features(df)

    print("  Adding price features...")
    df = add_price_features(df)

    print("  Adding Fourier features...")
    df = add_fourier_features(df)

    print("  Adding target encoding...")
    df = add_target_encoding(df)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw data...")
    df = load_raw_data()
    print(f"  -> {len(df):,} rows")

    print("Engineering features...")
    df = build_features(df)

    # Drop rows with NaN from lagging (first 28 days per SKU)
    initial_len = len(df)
    df = df.dropna(subset=["lag_28"]).reset_index(drop=True)
    print(f"  -> Dropped {initial_len - len(df):,} rows (warm-up period)")

    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    df.to_csv(out_path, index=False)
    print(f"  -> {len(df):,} rows, {len(df.columns)} columns")
    print(f"  -> Saved to {out_path}")

    # Print feature list
    feature_cols = [
        c
        for c in df.columns
        if c not in ["date", "sku_id", "category", "units_sold", "revenue"]
    ]
    print(f"\nFeatures ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
