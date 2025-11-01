# EUROCONTROL Flight Delay Forecasting — ANSP Dataset (2020–2024)

A machine learning and time-series forecasting project predicting **daily en-route Air Traffic Flow Management (ATFM) delays** across Europe using operational data from **EUROCONTROL ANSP** (Air Navigation Service Providers).  

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Why This Project Matters](#why-this-project-matters)
- [Dataset](#dataset)
- [Exploratory Analysis](#exploratory-analysis)
- [Model Development](#model-development)
- [Model Comparison](#model-comparison)
- [Model Interpretability](#model-interpretability)
- [Cost–Benefit Impact](#costbenefit-impact)
- [Streamlit Web App](#streamlit-web-app)
- [Author](#author)

---

## Project Overview

This project focuses on predicting **daily en-route flight delays** across European airspace using **EUROCONTROL ATFM operational datasets** from 2020–2024.  
The analysis combines both **classical time series forecasting (SARIMA)** and **machine learning (XGBoost)** models to capture delay volatility, seasonal peaks, and operational variations across different ANSPs.

Forecasting air traffic delays supports **strategic decision-making** in network operations, staffing, and congestion management — helping Europe’s aviation ecosystem move toward **data-driven efficiency**.

---

## Business Objective

The aim is to forecast **daily air traffic flow management delays** to support:

1. **Air Navigation Service Providers (ANSPs):** Optimize controller staffing and route management.  
2. **Airlines:** Adjust schedules to reduce cascading delays and fuel waste.  
3. **Airports:** Manage arrival/departure congestion more effectively.  
4. **Policy Makers:** Monitor regional performance for capacity planning.  

Accurate forecasting of delays directly impacts cost efficiency, sustainability, and passenger satisfaction across the aviation value chain.

---

## Why This Project Matters

Unpredictable flight delays create ripple effects across the aviation network — increasing costs, emissions, and workload.  
By forecasting delays with precision, this project provides:

- **Early warning systems** for delay spikes.  
- **Operational foresight** for ANSPs and airlines.  
- **Data-backed policy insights** for European airspace management.  

For instance, a 5% reduction in daily en-route delay minutes could save millions of euros annually across European carriers through reduced fuel burn and crew overtime.

---

## Dataset

- **Source:** [EUROCONTROL Performance Review Unit (PRU)](https://ansperformance.eu/data/)  
- **Files Used:** `ert_dly_ansp_2020.csv.bz2` → `ert_dly_ansp_2024.csv.bz2`  
- **Time Range:** January 2020 – September 2025  
- **Volume:** ~50,000 records across multiple ANSPs  

**Key Features:**
- `FLT_DATE` — Flight date (UTC)  
- `ENTITY_NAME` — Air Navigation Service Provider (ANSP)  
- `FLT_ERT_1_DLY` — Total en-route delay (minutes)  
- Additional delay causes: `DLY_ERT_A_1`, `DLY_ERT_C_1`, etc.

---

## Exploratory Analysis

Exploration revealed strong **seasonality** and **weekday patterns**:
- Delay peaks during **summer months (June–August)**.  
- Higher delays on **weekends** due to flight density.  
- Certain ANSPs consistently contribute more to total delay minutes.

Visualizations included:
- Daily and monthly delay trend plots.  
- Distribution of total delay minutes (revealing heavy tails).  
- ANSP-level comparison of delay contribution.

---

## Model Development

### Phase 1 — SARIMA (Statistical Baseline)
- Captured seasonality and long-term trends.
- Struggled with irregular spikes and high variance.

### Phase 2 — XGBoost (Machine Learning)
- Engineered **lag features**, **rolling means**, and **volatility indicators**.  
- Tuned with cross-validation for optimal generalization.  
- Delivered a dramatic improvement in predictive accuracy.

---

## Model Comparison

| Metric        | SARIMA     | XGBoost (Tuned) |
| ------------- | ---------- | --------------- |
| **MAE**       | 28,270 min | **5,058 min**   |
| **RMSE**      | 33,594 min | **8,160 min**   |
| **Adjusted MAPE** | 917.67% | **51.42%**      |

✅ **XGBoost outperformed SARIMA by reducing error over 80%**, thanks to lag and rolling feature engineering.

---

## Model Interpretability

Interpretability was key to validating the model for operational decisions.  
Feature importance analysis (via **XGBoost gain values**) highlighted:

| Feature | Influence | Description |
|----------|------------|-------------|
| Lag_1, Lag_7 | High | Recent delay patterns heavily influence forecasts. |
| RollingMean_7 | Medium | Weekly delay trend smooths volatility. |
| Month, DayOfWeek | Medium | Seasonality and weekday patterns affect delays. |
| Entity_Name | Low | Some ANSPs have consistent baseline delay behavior. |

This interpretability ensures stakeholders understand *why* the model predicts high delay periods — improving trust and adoption.

---

## Cost–Benefit Impact

- Predictive delay insights can **reduce fuel and operational costs** by allowing airlines to adjust proactively.  
- A reduction of just **2 minutes per flight** across Europe translates to **hundreds of tons of CO₂ savings daily**.  
- Enables **resource balancing** for ANSPs — minimizing overtime and airspace congestion.  
- Contributes to **sustainability goals** under the Single European Sky initiative.  

The forecast model supports both **economic optimization** and **environmental impact reduction**.

---

## Streamlit Web App

Live Demo: **[Click Here to Open](https://sha-md-eurocontrol-flight-delay-forecasting-app-lpetac.streamlit.app/)**

App features include:
- Delay trend visualizations by time, ANSP, and cause.  
- XGBoost-based daily delay forecasting.  
- Interactive selection of date range and entity.  
- Cached inference for faster prediction.  

Technologies used: **Streamlit, Python, Pandas, XGBoost, Matplotlib, Statsmodels**

---

## Author

**Shabnam Begam Mahammad**  
[LinkedIn](https://www.linkedin.com/in/shabnam-b-mahammad) | [Email](mailto:shabnam71.md@gmail.com) 

“Transforming air traffic data into smarter skies through machine learning.”
