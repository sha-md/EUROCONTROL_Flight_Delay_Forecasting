# ✈️ EUROCONTROL Air Traffic Delay Intelligence Dashboard

This repository presents a machine learning approach for forecasting **EUROCONTROL en-route Air Traffic Flow Management (ATFM) delays** using historical Air Navigation Service Provider (ANSP) delay data. It includes research notebooks exploring statistical and machine learning forecasting techniques, together with an interactive Streamlit dashboard for visualising historical delays and generating future delay forecasts.

**🌐 Live Demo:** https://sha-md-eurocontrol-flight-delay-forecasting-app-lpetac.streamlit.app/

---

# 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Objective](#-business-objective)
- [Why This Project Matters](#-why-this-project-matters)
- [Features](#-features)
- [Dashboard](#-dashboard)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Forecast Workflow](#-forecast-workflow)
- [Technologies Used](#-technologies-used)
- [Dashboard Preview](#-dashboard-preview)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

# 📌 Project Overview

Air traffic delays are a major challenge for airlines, airports, Air Navigation Service Providers (ANSPs), and passengers. Unexpected congestion can lead to operational disruption, increased fuel consumption, crew overtime, and passenger dissatisfaction.

This project analyses historical EUROCONTROL en-route delay data to identify traffic patterns and forecast future daily ATFM delays.

The repository contains:

- Research notebooks for forecasting model development
- A production-ready Streamlit dashboard
- Interactive visualisations
- Machine learning-based forecasting
- Downloadable forecast reports

Potential users include:

- Air Navigation Service Providers (ANSPs)
- Airlines
- Airport operators
- Aviation researchers
- Air Traffic Management (ATM) analysts

---

# 🎯 Business Objective

The objective of this project is to forecast daily en-route ATFM delays to support operational decision-making across the aviation industry.

Accurate forecasting enables stakeholders to:

- Improve controller and workforce planning
- Reduce operational disruption
- Improve airline schedule optimisation
- Better manage airport congestion
- Support proactive capacity planning
- Improve overall network efficiency

---

# ✈️ Why This Project Matters

Flight delays create significant operational and financial challenges across the aviation network.

Reliable delay forecasting helps organisations:

- Detect potential congestion before it occurs
- Allocate operational resources efficiently
- Improve traffic flow management
- Support data-driven decision making
- Reduce delays and improve passenger experience

---

# 🚀 Features

- Interactive Streamlit dashboard
- Automatic preprocessing of EUROCONTROL datasets
- Default 2024 dataset included
- Optional upload of custom CSV/BZ2 datasets
- Historical daily delay analysis
- Monthly delay trend analysis
- Adjustable forecasting horizon (1–30 days)
- XGBoost-based forecasting
- Model evaluation using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Adjusted Mean Absolute Percentage Error (MAPE)
- Feature importance visualisation
- Business insight generation
- Forecast download as CSV

---

# 📊 Dashboard

The Streamlit application consists of three interactive tabs.

## 📊 Dashboard

Provides:

- Dataset overview
- KPI metrics
- Year filter
- Month filter
- Daily delay trend
- Monthly delay trend
- Processed dataset preview

---

## 📈 Forecast

Provides:

- Adjustable forecast horizon (1–30 days)
- Interactive forecast visualisation
- Forecast summary
- Automatically generated business insights
- CSV download of forecast results

---

## 🤖 Model

Displays:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Adjusted Mean Absolute Percentage Error (MAPE)
- XGBoost feature importance

---

# 📂 Dataset

**Source**

EUROCONTROL Performance Review Unit (PRU)

https://ansperformance.eu/data/

## Research Datasets

Model development and experimentation were performed using historical EUROCONTROL datasets covering **2020–2024**.

Datasets used include:

- `ert_dly_ansp_2020.csv.bz2`
- `ert_dly_ansp_2021.csv.bz2`
- `ert_dly_ansp_2022.csv.bz2`
- `ert_dly_ansp_2023.csv.bz2`
- `ert_dly_ansp_2024.csv.bz2`

These datasets were used during exploratory analysis and model development in the Jupyter notebooks.

## Streamlit Dashboard

To keep the dashboard lightweight and responsive, the Streamlit application loads:

- `ert_dly_ansp_2024.csv.bz2`

as the default dataset.

Users may optionally upload their own compatible EUROCONTROL datasets for analysis and forecasting.

Supported formats:

- CSV
- BZ2 compressed CSV

---

# 🧠 Machine Learning Models

Two forecasting approaches were explored during model development.

## SARIMA

A statistical time-series forecasting model was initially developed and evaluated as a baseline.

Although SARIMA captured overall trends, it struggled to model sudden fluctuations and complex delay behaviour.

---

## XGBoost

The final dashboard uses **XGBoost Regressor**, which achieved significantly better forecasting performance.

### Feature Engineering

The model automatically generates:

- Day of week
- Month
- Weekend indicator
- Lag-1 delay
- Lag-7 delay
- 7-day rolling mean
- 7-day rolling standard deviation

These engineered features enable the model to capture temporal dependencies, seasonality, and recent traffic behaviour.

---

# 📈 Forecast Workflow

1. Load the default dataset or upload a custom dataset.
2. Clean and preprocess the data.
3. Aggregate delays into daily totals.
4. Generate time-series features.
5. Train the XGBoost forecasting model.
6. Evaluate model performance.
7. Forecast future daily delays.
8. Visualise historical and forecasted delays.
9. Download forecast results.

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- XGBoost
- Scikit-learn
- Statsmodels

---

# 📷 Dashboard Preview

## Dashboard

![](assets/dashboard.png)

---

## Forecast

![](assets/forecast.png)

---

## Model Evaluation

![](assets/model.png)

---

# 🚀 Future Improvements

- Multi-year forecasting
- Hyperparameter optimisation
- SHAP model explainability
- Weather data integration
- Flight volume prediction
- Comparison with Prophet, LightGBM and LSTM models
- Real-time EUROCONTROL data integration
- Additional operational KPIs and analytics

---

# 👤 Author

**Shabnam Begam Mahammad**  
[LinkedIn](https://www.linkedin.com/in/shabnam-b-mahammad) | [Email](mailto:md.shabnam21@gmail.com) 

“Transforming air traffic data into smarter skies through machine learning and data analytics.”
