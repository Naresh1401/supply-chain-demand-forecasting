"""
Generate realistic synthetic supply chain sales data.

Creates daily sales records for 50 SKUs across 5 product categories
spanning 3 years (2023-01-01 to 2025-12-31) with realistic patterns:
trend, weekly/monthly/annual seasonality, promotions, stockouts,
holiday effects, and new product launches.
"""

import os
import numpy as np
import pandas as pd
import holidays

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

CATEGORIES = {
    "Beverages": {
        "skus": 12,
        "base_demand_range": (80, 200),
        "price_range": (1.50, 5.99),
        "weekend_lift": 1.25,
    },
    "Snacks": {
        "skus": 12,
        "base_demand_range": (60, 180),
        "price_range": (2.00, 7.99),
        "weekend_lift": 1.30,
    },
    "Dairy": {
        "skus": 10,
        "base_demand_range": (50, 150),
        "price_range": (2.50, 8.99),
        "weekend_lift": 1.10,
    },
    "Frozen": {
        "skus": 8,
        "base_demand_range": (30, 120),
        "price_range": (3.00, 12.99),
        "weekend_lift": 1.15,
    },
    "Personal Care": {
        "skus": 8,
        "base_demand_range": (20, 100),
        "price_range": (4.00, 15.99),
        "weekend_lift": 0.95,
    },
}


def _generate_sku_metadata() -> pd.DataFrame:
    """Generate SKU-level metadata."""
    records = []
    sku_idx = 0
    for category, config in CATEGORIES.items():
        for i in range(config["skus"]):
            sku_id = f"SKU_{sku_idx:03d}"
            base_demand = np.random.uniform(*config["base_demand_range"])
            base_price = round(np.random.uniform(*config["price_range"]), 2)
            trend_slope = np.random.choice(
                [1, -1], p=[0.7, 0.3]
            ) * np.random.uniform(0.005, 0.08)
            launch_delay_days = (
                np.random.randint(90, 540) if np.random.rand() < 0.15 else 0
            )
            records.append(
                {
                    "sku_id": sku_id,
                    "category": category,
                    "base_demand": round(base_demand, 1),
                    "base_price": base_price,
                    "trend_slope": round(trend_slope, 4),
                    "weekend_lift": config["weekend_lift"],
                    "launch_delay_days": launch_delay_days,
                }
            )
            sku_idx += 1
    return pd.DataFrame(records)


def _generate_daily_sales(sku_meta: pd.DataFrame) -> pd.DataFrame:
    """Generate daily sales data for all SKUs."""
    date_range = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    us_holidays = holidays.US(years=[2023, 2024, 2025])
    n_days = len(date_range)

    # Shared temperature signal (sinusoidal annual cycle + noise)
    day_of_year = np.array([d.timetuple().tm_yday for d in date_range])
    temperature = 55 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 5, n_days)
    temperature = np.clip(temperature, 10, 105)

    all_records = []
    for _, sku in sku_meta.iterrows():
        sku_id = sku["sku_id"]
        base = sku["base_demand"]
        price = sku["base_price"]
        trend = sku["trend_slope"]
        wknd_lift = sku["weekend_lift"]
        launch_delay = sku["launch_delay_days"]

        # Trend component (linear)
        t = np.arange(n_days)
        trend_component = 1 + trend * t / 365

        # Weekly seasonality
        dow = np.array([d.weekday() for d in date_range])
        weekly = np.ones(n_days)
        weekly[dow >= 5] = wknd_lift  # Weekend
        weekly[dow == 0] *= 0.90      # Monday dip
        weekly[dow == 4] *= 1.05      # Friday bump

        # Monthly seasonality (payday effect around 1st and 15th)
        dom = np.array([d.day for d in date_range])
        monthly = np.ones(n_days)
        monthly[(dom <= 3) | ((dom >= 14) & (dom <= 17))] *= 1.10

        # Annual seasonality (summer bump for beverages/snacks, winter for comfort)
        month = np.array([d.month for d in date_range])
        annual = np.ones(n_days)
        if sku["category"] in ["Beverages", "Snacks"]:
            annual += 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        elif sku["category"] == "Frozen":
            annual += 0.10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        elif sku["category"] == "Dairy":
            annual += 0.05 * np.cos(2 * np.pi * (day_of_year - 172) / 365)

        # Holiday effects
        holiday_flag = np.array([1 if d in us_holidays else 0 for d in date_range])
        holiday_lift = 1 + 0.30 * holiday_flag

        # Promotion schedule (random bursts, ~12% of days)
        promotion_flag = np.zeros(n_days, dtype=int)
        promo_starts = np.random.choice(n_days - 7, size=int(n_days * 0.04), replace=False)
        for ps in promo_starts:
            duration = np.random.randint(3, 8)
            promotion_flag[ps : ps + duration] = 1
        promotion_discount = promotion_flag * np.random.uniform(0.10, 0.35, n_days)
        promotion_discount = np.round(promotion_discount, 2)
        promo_lift = 1 + 1.5 * promotion_discount  # Promotions boost demand

        # Competitor price (fluctuates around SKU price)
        competitor_price = price * (1 + np.cumsum(np.random.normal(0, 0.003, n_days)))
        competitor_price = np.clip(competitor_price, price * 0.7, price * 1.4)
        competitor_price = np.round(competitor_price, 2)

        # Price variations (occasional price changes)
        actual_price = np.full(n_days, price)
        # Apply promotion discounts to price
        actual_price = actual_price * (1 - promotion_discount)
        actual_price = np.round(actual_price, 2)

        # Combine all components
        demand = (
            base
            * trend_component
            * weekly
            * monthly
            * annual
            * holiday_lift
            * promo_lift
        )

        # Temperature effect (category-specific)
        temp_centered = (temperature - 65) / 20
        if sku["category"] == "Beverages":
            demand *= 1 + 0.12 * temp_centered
        elif sku["category"] == "Frozen":
            demand *= 1 + 0.08 * temp_centered
        elif sku["category"] == "Personal Care":
            demand *= 1 + 0.03 * temp_centered

        # Add noise
        noise = np.random.normal(1, 0.08, n_days)
        demand *= noise

        # Stockout events (~2% of days, demand drops to near zero)
        stockout_mask = np.random.rand(n_days) < 0.02
        demand[stockout_mask] *= np.random.uniform(0.0, 0.15, stockout_mask.sum())

        # New product launch: zero demand before launch
        if launch_delay > 0:
            demand[:launch_delay] = 0
            # Ramp up period
            ramp = min(60, n_days - launch_delay)
            ramp_factor = np.linspace(0.3, 1.0, ramp)
            demand[launch_delay : launch_delay + ramp] *= ramp_factor

        # Floor at zero, round to integers
        units_sold = np.maximum(0, np.round(demand)).astype(int)

        # Revenue
        revenue = np.round(units_sold * actual_price, 2)

        # Inventory level (simplified: decays with sales, replenishes periodically)
        inventory = np.zeros(n_days)
        inventory[0] = base * 14  # 2-week supply
        for j in range(1, n_days):
            inventory[j] = inventory[j - 1] - units_sold[j]
            # Replenish every 7 days or when low
            if j % 7 == 0 or inventory[j] < base * 3:
                inventory[j] += base * 10
            inventory[j] = max(0, inventory[j])
        inventory = np.round(inventory).astype(int)

        for j in range(n_days):
            d = date_range[j]
            all_records.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "sku_id": sku_id,
                    "category": sku["category"],
                    "units_sold": units_sold[j],
                    "revenue": revenue[j],
                    "price": actual_price[j],
                    "promotion_flag": promotion_flag[j],
                    "promotion_discount": promotion_discount[j],
                    "holiday_flag": holiday_flag[j],
                    "temperature": round(temperature[j], 1),
                    "day_of_week": d.weekday(),
                    "month": d.month,
                    "is_weekend": int(d.weekday() >= 5),
                    "competitor_price": competitor_price[j],
                    "inventory_level": inventory[j],
                }
            )

    return pd.DataFrame(all_records)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Generating SKU metadata...")
    sku_meta = _generate_sku_metadata()
    sku_meta.to_csv(os.path.join(RAW_DIR, "sku_metadata.csv"), index=False)
    print(f"  -> {len(sku_meta)} SKUs across {sku_meta['category'].nunique()} categories")

    print("Generating daily sales data...")
    sales = _generate_daily_sales(sku_meta)
    sales.to_csv(os.path.join(RAW_DIR, "sales_data.csv"), index=False)
    print(f"  -> {len(sales):,} rows ({sales['sku_id'].nunique()} SKUs × {sales['date'].nunique()} days)")
    print(f"  -> Date range: {sales['date'].min()} to {sales['date'].max()}")
    print(f"  -> Saved to {RAW_DIR}/")


if __name__ == "__main__":
    main()
