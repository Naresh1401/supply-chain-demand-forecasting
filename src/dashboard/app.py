"""
Streamlit dashboard for demand forecasting visualization.

Tabs:
- Forecast Explorer: interactive forecast plot with confidence intervals
- Model Comparison: compare all models across metrics
- Demand Decomposition: trend, seasonality, residual
- Inventory Planning: safety stock calculator
- Category View: aggregate category-level analysis
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide",
)


# ---------- Data Loading ----------

@st.cache_data
def load_data():
    path = os.path.join(PROCESSED_DIR, "features.csv")
    if not os.path.exists(path):
        st.error("Processed data not found. Run `make features` first.")
        st.stop()
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_model():
    path = os.path.join(MODELS_DIR, "best_ml_model.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ---------- App ----------

def main():
    st.title("📦 Supply Chain Demand Forecasting Platform")
    st.markdown("Multi-horizon demand forecasting with statistical, ML, and Prophet models")

    df = load_data()
    metrics = load_metrics()
    model = load_model()

    skus = sorted(df["sku_id"].unique())
    categories = sorted(df["category"].unique())

    # Sidebar
    st.sidebar.header("Controls")
    selected_sku = st.sidebar.selectbox("Select SKU", skus, index=0)
    selected_category = df[df["sku_id"] == selected_sku]["category"].iloc[0]
    st.sidebar.info(f"**Category:** {selected_category}")
    horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 28)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecast Explorer",
        "🏆 Model Comparison",
        "🔍 Demand Decomposition",
        "📦 Inventory Planning",
        "📊 Category View",
    ])

    # ---- Tab 1: Forecast Explorer ----
    with tab1:
        st.subheader(f"Demand Forecast — {selected_sku}")

        sku_df = df[df["sku_id"] == selected_sku].sort_values("date")
        cutoff = sku_df["date"].max() - pd.Timedelta(days=horizon)
        train = sku_df[sku_df["date"] <= cutoff]
        test = sku_df[sku_df["date"] > cutoff]

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=train.tail(90)["date"], y=train.tail(90)["units_sold"],
            name="Historical", mode="lines", line=dict(color="#2196F3", width=1.5),
        ))

        # Actual (test period)
        fig.add_trace(go.Scatter(
            x=test["date"], y=test["units_sold"],
            name="Actual", mode="lines", line=dict(color="#2196F3", width=2.5),
        ))

        # Forecast
        if model is not None:
            feature_cols = model.feature_cols
            test_clean = test.dropna(subset=feature_cols)
            if len(test_clean) > 0:
                preds = model.predict(test_clean)
                residual_std = float(np.std(train.tail(90)["units_sold"].values)) * 0.15

                fig.add_trace(go.Scatter(
                    x=test_clean["date"], y=preds,
                    name=f"Forecast ({model.name})", mode="lines",
                    line=dict(color="#FF5722", width=2.5, dash="dash"),
                ))

                # 80% CI
                fig.add_trace(go.Scatter(
                    x=pd.concat([test_clean["date"], test_clean["date"][::-1]]),
                    y=np.concatenate([preds + 1.28 * residual_std, (preds - 1.28 * residual_std)[::-1]]),
                    fill="toself", fillcolor="rgba(255,87,34,0.15)",
                    line=dict(width=0), name="80% CI", showlegend=True,
                ))

                # 95% CI
                fig.add_trace(go.Scatter(
                    x=pd.concat([test_clean["date"], test_clean["date"][::-1]]),
                    y=np.concatenate([preds + 1.96 * residual_std, (preds - 1.96 * residual_std)[::-1]]),
                    fill="toself", fillcolor="rgba(255,87,34,0.07)",
                    line=dict(width=0), name="95% CI", showlegend=True,
                ))
        else:
            st.warning("No trained model found. Run `make train` first.")

        fig.update_layout(
            title=f"Demand Forecast — {selected_sku} ({selected_category})",
            xaxis_title="Date", yaxis_title="Units Sold",
            height=500, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Daily Demand", f"{sku_df['units_sold'].mean():.0f}")
        col2.metric("Max Daily Demand", f"{sku_df['units_sold'].max():.0f}")
        col3.metric("Days with Promos", f"{sku_df['promotion_flag'].sum():.0f}")
        col4.metric("Std Dev", f"{sku_df['units_sold'].std():.1f}")

    # ---- Tab 2: Model Comparison ----
    with tab2:
        st.subheader("Model Performance Comparison")

        if metrics:
            metric_names = ["MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "Bias"]
            rows = []
            for model_name, vals in metrics.items():
                row = {"Model": model_name}
                row.update({k: vals.get(k, None) for k in metric_names})
                rows.append(row)

            perf_df = pd.DataFrame(rows)
            st.dataframe(perf_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE", "WAPE"], color="#c8e6c9"), use_container_width=True)

            # Grouped bar chart
            fig = go.Figure()
            for metric in ["MAE", "RMSE"]:
                fig.add_trace(go.Bar(
                    x=perf_df["Model"], y=perf_df[metric], name=metric
                ))
            fig.update_layout(
                title="MAE & RMSE by Model", barmode="group",
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # MAPE comparison
            fig2 = px.bar(perf_df, x="Model", y="MAPE", color="MAPE",
                          color_continuous_scale="RdYlGn_r", title="MAPE (%) by Model")
            fig2.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No metrics found. Run `make train` first.")

    # ---- Tab 3: Demand Decomposition ----
    with tab3:
        st.subheader(f"Demand Decomposition — {selected_sku}")

        sku_ts = sku_df.set_index("date")["units_sold"]
        if len(sku_ts) >= 60:
            try:
                decomp = seasonal_decompose(sku_ts, model="additive", period=7, extrapolate_trend="freq")

                fig = go.Figure()
                for name, data, color in [
                    ("Observed", decomp.observed, "#2196F3"),
                    ("Trend", decomp.trend, "#4CAF50"),
                    ("Seasonal", decomp.seasonal, "#FF9800"),
                    ("Residual", decomp.resid, "#9C27B0"),
                ]:
                    fig.add_trace(go.Scatter(x=data.index, y=data.values, name=name, line=dict(color=color)))

                fig.update_layout(
                    title=f"STL Decomposition — {selected_sku}",
                    height=600, template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Decomposition failed: {e}")
        else:
            st.warning("Not enough data for decomposition.")

    # ---- Tab 4: Inventory Planning ----
    with tab4:
        st.subheader("📦 Safety Stock Calculator")

        st.markdown("""
        Calculate recommended safety stock based on forecast uncertainty
        and desired service level.
        """)

        col1, col2 = st.columns(2)
        with col1:
            service_level = st.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)
            lead_time = st.number_input("Lead Time (days)", min_value=1, max_value=30, value=7)

        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z = z_scores.get(service_level, 1.65)

        avg_demand = sku_df["units_sold"].mean()
        std_demand = sku_df["units_sold"].std()

        safety_stock = z * std_demand * np.sqrt(lead_time)
        reorder_point = avg_demand * lead_time + safety_stock

        with col2:
            st.metric("Safety Stock (units)", f"{safety_stock:.0f}")
            st.metric("Reorder Point (units)", f"{reorder_point:.0f}")
            st.metric("Avg Daily Demand", f"{avg_demand:.0f}")
            st.metric("Demand Std Dev", f"{std_demand:.1f}")

        st.markdown("---")
        st.markdown(f"""
        **Formula:**
        - Safety Stock = Z × σ × √(Lead Time) = {z} × {std_demand:.1f} × √{lead_time} = **{safety_stock:.0f}**
        - Reorder Point = (Avg Demand × Lead Time) + Safety Stock = **{reorder_point:.0f}**
        """)

    # ---- Tab 5: Category View ----
    with tab5:
        st.subheader("Category-Level Demand Analysis")

        cat_daily = df.groupby(["date", "category"])["units_sold"].sum().reset_index()
        fig = px.line(
            cat_daily, x="date", y="units_sold", color="category",
            title="Total Daily Demand by Category",
        )
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Category summary table
        cat_summary = df.groupby("category").agg(
            total_skus=("sku_id", "nunique"),
            avg_daily_demand=("units_sold", "mean"),
            total_revenue=("revenue", "sum"),
            promo_pct=("promotion_flag", "mean"),
        ).round(2).reset_index()
        cat_summary["promo_pct"] = (cat_summary["promo_pct"] * 100).round(1)
        cat_summary.columns = ["Category", "SKUs", "Avg Daily Demand", "Total Revenue ($)", "Promo Days (%)"]
        st.dataframe(cat_summary, use_container_width=True)

        # Category selector for drilldown
        sel_cat = st.selectbox("Drill into Category", categories)
        cat_skus = df[df["category"] == sel_cat]["sku_id"].unique()
        cat_sku_data = df[df["sku_id"].isin(cat_skus)].groupby(["date", "sku_id"])["units_sold"].sum().reset_index()

        fig2 = px.line(
            cat_sku_data, x="date", y="units_sold", color="sku_id",
            title=f"Daily Demand — {sel_cat} SKUs",
        )
        fig2.update_layout(height=500, template="plotly_white", showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
