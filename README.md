# 🏭 Supply Chain Demand Forecasting Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-Time%20Series-orange.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)
![Models](https://img.shields.io/badge/Models-12+-purple.svg)

**Multi-horizon demand forecasting system** combining statistical, machine learning, and Prophet models with walk-forward validation and probabilistic predictions for supply chain optimization.

---

## 📋 Overview

This platform forecasts daily demand for 50 SKUs across 5 product categories using an ensemble of 12+ models. It features rigorous time series cross-validation, comprehensive feature engineering, a REST API for serving predictions, and an interactive Streamlit dashboard for business stakeholders.

### Key Capabilities
- **Multi-horizon forecasting**: 7-day, 14-day, and 28-day ahead predictions
- **12+ models**: From naive baselines to gradient boosting and Prophet
- **Walk-forward validation**: 5-fold expanding window CV respecting temporal order
- **Probabilistic forecasts**: Prediction intervals at 80% and 95% confidence
- **Production-ready API**: FastAPI with Pydantic validation
- **Interactive dashboard**: Streamlit with Plotly visualizations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (:8501)                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Forecast  │  │   Model      │  │ Demand   │  │  Inventory    │  │
│  │ Explorer  │  │   Comparison │  │ Decomp.  │  │  Planning     │  │
│  └──────────┘  └──────────────┘  └──────────┘  └───────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      FastAPI Service (:8000)                        │
│  POST /forecast    GET /skus    GET /model/performance    /health   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                        Model Registry                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Stat.   │  │  Ridge  │  │ XGBoost │  │ LightGBM│  │ Prophet │ │
│  │ Models  │  │  + RF   │  │         │  │         │  │         │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    Feature Engineering Pipeline                     │
│  Lags │ Rolling Stats │ Calendar │ Price │ Fourier │ Target Enc.   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     Raw Data (Synthetic)                             │
│         50 SKUs × 5 Categories × 3 Years Daily Sales                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

| Model               | MAE   | RMSE  | MAPE (%) | WAPE (%) | Horizon |
|----------------------|-------|-------|----------|----------|---------|
| Naive                | 28.4  | 36.1  | 22.3     | 19.8     | 7-day   |
| Seasonal Naive       | 19.7  | 25.3  | 15.6     | 13.7     | 7-day   |
| Moving Average       | 21.2  | 27.8  | 16.8     | 14.8     | 7-day   |
| Holt-Winters         | 16.3  | 21.4  | 12.9     | 11.4     | 7-day   |
| SARIMA               | 15.1  | 19.8  | 11.9     | 10.5     | 7-day   |
| Ridge Regression     | 13.8  | 18.2  | 10.9     | 9.6      | 7-day   |
| Random Forest        | 11.5  | 15.3  | 9.1      | 8.0      | 7-day   |
| **XGBoost**          | **9.2** | **12.4** | **7.3** | **6.4** | 7-day   |
| **LightGBM**         | **9.0** | **12.1** | **7.1** | **6.3** | 7-day   |
| Prophet              | 12.8  | 16.7  | 10.1     | 8.9      | 7-day   |

*Results on holdout test set (last 90 days). Best model selected per SKU.*

---

## 🛠️ Tech Stack

| Component          | Technology                                    |
|--------------------|-----------------------------------------------|
| Language           | Python 3.10+                                  |
| Statistical Models | statsmodels, scipy                            |
| ML Models          | scikit-learn, XGBoost, LightGBM               |
| Prophet            | prophet                                       |
| API                | FastAPI, Uvicorn, Pydantic                     |
| Dashboard          | Streamlit, Plotly                              |
| Visualization      | Matplotlib, Seaborn, Plotly                    |
| Data Processing    | Pandas, NumPy                                  |
| Containerization   | Docker, Docker Compose                         |

---

## 📁 Project Structure

```
supply-chain-demand-forecasting/
├── data/
│   ├── raw/                    # Raw synthetic data
│   └── processed/              # Feature-engineered datasets
├── models/                     # Serialized trained models
├── results/
│   ├── figures/                # Evaluation plots
│   └── metrics.json            # Model performance metrics
├── src/
│   ├── data/
│   │   └── generate_data.py    # Synthetic data generation
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── statistical_models.py
│   │   ├── ml_models.py
│   │   ├── prophet_model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── api/
│   │   └── app.py              # FastAPI service
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip or conda

### Quick Start

```bash
# Clone and setup
git clone https://github.com/nareshsampangi/supply-chain-demand-forecasting.git
cd supply-chain-demand-forecasting
make install

# Run full pipeline
make all

# Or step by step
make data        # Generate synthetic data
make features    # Engineer features
make train       # Train all models
make evaluate    # Generate evaluation plots

# Serve
make api         # Start FastAPI on :8000
make dashboard   # Start Streamlit on :8501
```

### Docker

```bash
docker-compose up --build
```

---

## 📈 Methodology

1. **Data Generation**: Realistic synthetic data with trend, seasonality, promotions, holidays, stockouts
2. **Feature Engineering**: 50+ features including lags, rolling stats, calendar, price, Fourier terms
3. **Model Training**: Walk-forward cross-validation with expanding window
4. **Evaluation**: Multiple error metrics with statistical significance testing
5. **Deployment**: REST API + interactive dashboard

---

## 👤 Author

**Naresh Sampangi**

---

## 📄 License

This project is licensed under the MIT License.
