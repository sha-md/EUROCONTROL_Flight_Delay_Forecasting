# ✈️ EUROCONTROL Flight Delay Forecasting — ANSP Dataset (2020–2024)

Forecasting **daily en-route Air Traffic Flow Management (ATFM) delays** across Europe using real **EUROCONTROL ANSP operational data**.  
The project explores two complementary forecasting approaches — first using a **SARIMA** model, then transitioning to a more powerful **XGBoost** model that captures non-linear patterns and volatility.

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)  
2. [Business Objective](#business-objective)  
3. [Dataset](#dataset)  
4. [Exploratory Analysis](#exploratory-analysis)  
5. [Model Comparison](#model-comparison)  
6. [Key Insights](#key-insights)  
7. [Tech Stack](#tech-stack)  
8. [Author](#author)

---

## 🧠 Project Overview
This project focuses on predicting **daily en-route flight delays** across European airspace using **EUROCONTROL ATFM datasets** from 2020–2024.  
Accurate forecasting helps **air navigation service providers (ANSPs)**, **airlines**, and **airports** optimize resources, manage congestion, and improve network efficiency.

First modeled the time series using **SARIMA**, but due to high volatility and irregular spikes in daily delays, I switched to **XGBoost** with lag and rolling features.  
This significantly improved **model accuracy, stability, and interpretability.**

---

## 💼 Business Objective
Air traffic delays have major operational and financial impacts, including:
- Increased **fuel consumption** and **CO₂ emissions**  
- Crew rescheduling and slot management issues  
- Passenger dissatisfaction and cascading disruptions  

Forecasting these delays enables **predictive planning**, **capacity management**, and **cost savings** through data-driven operational decisions.

---

## 📊 Dataset
- **Source:** [EUROCONTROL Performance Review Unit (PRU)](https://ansperformance.eu/data/)  
- **Files Used:** `ert_dly_ansp_2020.csv.bz2` → `ert_dly_ansp_2024.csv.bz2`  
- **Time Range:** January 2020 – September 2025  
- **Data Volume:** ~50,000 records  
- **Main Fields:**
  - `FLT_DATE` — Flight date (UTC)  
  - `ENTITY_NAME` — Air Navigation Service Provider (ANSP)  
  - `FLT_ERT_1_DLY` — Total en-route delay in minutes  
  - Other delay causes: `DLY_ERT_A_1`, `DLY_ERT_C_1`, etc.

---

## 🔍 Exploratory Analysis (EDA)
- Strong **seasonality**: delay peaks during **summer months** (June–August).  
- Higher average delays on **weekends** due to increased flight density.  
- Some ANSPs contribute disproportionately to delay totals.  

**Visuals included:**
- 📈 Daily and monthly trend plots  
- 🏢 Top 10 entities contributing to delays  
- 📊 Distribution of total delay minutes (revealing heavy outliers)

---

## ⚖️ Model Comparison

| Metric        | SARIMA     | XGBoost (Tuned) |
| ------------- | ---------- | --------------- |
| **MAE**       | 28,270 min | **5,058 min**   |
| **RMSE**      | 33,594 min | **8,160 min**   |
| **Adjusted MAPE** | 917.67% | **51.42%**      |

✅ **XGBoost outperformed SARIMA by reducing error over 80%**, thanks to lag and rolling feature engineering.

---

## 💡 Key Insights
- Delays peak during **summer** and **weekends**.  
- XGBoost’s **lag-based and rolling features** were the strongest predictors.  
- Data-driven forecasting can support **strategic staffing**, **slot optimization**, and **fuel efficiency**.  
- Transitioning from **SARIMA → XGBoost** demonstrates how **ML-based time series modeling** can outperform traditional methods under volatility.

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python (3.10) |
| **Libraries** | Pandas • NumPy • Matplotlib • Seaborn • Statsmodels • Scikit-learn • XGBoost |
| **Environment** | Jupyter Notebook |
| **Version Control** | Git & GitHub |
| **Data Source** | EUROCONTROL PRU (Public Data) |

---

##  Author

**SHABNAM BEGAM MAHAMMAD**  
 [www.linkedin.com/in/shabnam-b-mahammad-377520272](#)  
 shabnam71.md@gmail.com  


