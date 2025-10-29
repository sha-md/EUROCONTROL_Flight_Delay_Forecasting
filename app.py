import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ----------------------------
# 🌍 PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="EUROCONTROL Flight Delay Forecasting", layout="wide")

st.title("✈️ EUROCONTROL Flight Delay Forecasting Dashboard (2020–2024)")
st.markdown("""
Forecasting **daily en-route Air Traffic Flow Management (ATFM) delays** across Europe using real **EUROCONTROL ANSP data**.
The project compares traditional **SARIMA** and **XGBoost** models for improved forecasting of operational delays.
""")

# ----------------------------
# 📥 LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    files = [f for f in os.listdir() if f.endswith(".bz2")]
    if not files:
        st.error("⚠️ No .bz2 data files found! Please upload at least one file (e.g., `ert_dly_ansp_2024.csv.bz2`) to your GitHub repository.")
        return pd.DataFrame(columns=["DATE", "TOTAL_DELAY"])
    
    dfs = []
    for f in files:
        df_year = pd.read_csv(f, compression="bz2")
        dfs.append(df_year)
    
    df = pd.concat(dfs, ignore_index=True)
    df["FLT_DATE"] = pd.to_datetime(df["FLT_DATE"], errors="coerce")
    df = df.dropna(subset=["FLT_DATE"]).sort_values("FLT_DATE")
    df["TOTAL_DELAY"] = df["FLT_ERT_1_DLY"].fillna(0)
    daily = df.groupby("FLT_DATE", as_index=False)["TOTAL_DELAY"].sum()
    daily.rename(columns={"FLT_DATE": "DATE"}, inplace=True)
    daily = daily.set_index("DATE").asfreq("D").fillna(0).reset_index()
    return daily

    

daily = load_data()

# ----------------------------
# 🧭 SIDEBAR
# ----------------------------
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Choose a section:", ["SARIMA Overview", "XGBoost Forecasting"])

st.sidebar.markdown("---")
st.sidebar.info("Developed by **[Your Name]** 🇪🇺\n\n📧 your@email.com\n\n🌐 [GitHub Repo](#)")

# ----------------------------
# 📈 SARIMA OVERVIEW TAB
# ----------------------------
if page == "SARIMA Overview":
    st.subheader("📊 SARIMA Model — Statistical Forecasting Overview")

    st.markdown("""
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) is a traditional time series model that captures:
    - **Trend** — gradual increase/decrease in delays  
    - **Seasonality** — weekly or monthly repeating patterns  
    - **Noise** — short-term random fluctuations  

    However, due to high volatility and sudden spikes in ATFM delays, SARIMA struggled to generalize effectively.
    """)

    st.write("### Example Daily Delay Trends")
    st.line_chart(daily.set_index("DATE")["TOTAL_DELAY"])

    st.markdown("""
    #### 📉 SARIMA Model Results (2020–2024)
    | Metric | Score |
    |:-------|------:|
    | **MAE** | 28,270 minutes |
    | **RMSE** | 33,594 minutes |
    | **Adjusted MAPE** | 917.67% |

    These metrics indicate poor performance during irregular delay spikes — motivating a shift to **XGBoost**.
    """)

# ----------------------------
# ⚙️ XGBOOST FORECASTING TAB
# ----------------------------
elif page == "XGBoost Forecasting":
    st.subheader("🧠 XGBoost Model — Machine Learning Forecasting")

    # Feature engineering
    df = daily.copy()
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["month"] = df["DATE"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["lag_1"] = df["TOTAL_DELAY"].shift(1)
    df["lag_7"] = df["TOTAL_DELAY"].shift(7)
    df["rolling_mean_7"] = df["TOTAL_DELAY"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["TOTAL_DELAY"].shift(1).rolling(7).std()
    df = df.dropna().reset_index(drop=True)

    FEATURES = ["day_of_week", "month", "is_weekend", "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]
    X = df[FEATURES]
    y = df["TOTAL_DELAY"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mask = y_test != 0
    mape = (np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]).mean() * 100

    st.markdown("### ⚡ Model Performance")
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:,.0f} min")
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:,.0f} min")
    st.metric(label="Adjusted MAPE", value=f"{mape:.2f}%")

    st.markdown("### 📈 Actual vs Predicted Delays")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["DATE"].iloc[split_idx:], y_test.values, label="Actual", linewidth=1.5)
    ax.plot(df["DATE"].iloc[split_idx:], y_pred, label="Predicted", linewidth=1.5)
    ax.legend()
    ax.set_title("Actual vs Predicted Daily Delay — XGBoost")
    ax.set_xlabel("Date")
    ax.set_ylabel("Delay (min)")
    st.pyplot(fig)

    # Feature Importance
    st.markdown("### 🔍 Feature Importance")
    importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(FEATURES, importance)
    ax.set_title("Top Features Impacting Delays")
    st.pyplot(fig)

    st.markdown("""
    **Observations:**
    - Lag and rolling features are the strongest predictors.
    - The model learns patterns between weekday/month and delay spikes.
    - Provides a robust alternative to traditional SARIMA under high variance conditions.
    """)

st.markdown("---")
st.caption("Developed with ❤️ by [Your Name] — Forecasting Flight Delays using SARIMA & XGBoost")
