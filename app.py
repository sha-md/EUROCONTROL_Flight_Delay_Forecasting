import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

st.set_page_config(page_title="EUROCONTROL Flight Delay Forecaster", layout="wide")

st.title("✈️ EUROCONTROL Flight Delay Forecasting Dashboard (2020–2024)")
st.markdown("""
Upload EUROCONTROL **ANSP delay data** to forecast **future daily delays** using a tuned **XGBoost model**.

This interactive tool:
- Cleans and aggregates your dataset
- Analyzes daily patterns
- Forecasts **next 7 days** of en-route ATFM delays  
""")

# ---------------------------
# 📂 FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload your EUROCONTROL .bz2 or .csv data file", type=["bz2", "csv"])

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")

    # Load uploaded data
    if uploaded_file.name.endswith(".bz2"):
        df = pd.read_csv(uploaded_file, compression="bz2")
    else:
        df = pd.read_csv(uploaded_file)

    # ---------------------------
    # 🧹 DATA CLEANING
    # ---------------------------
    st.subheader("🔍 Data Overview")

    df["FLT_DATE"] = pd.to_datetime(df["FLT_DATE"], errors="coerce")
    df = df.dropna(subset=["FLT_DATE"]).sort_values("FLT_DATE")

    # If total delay not directly available, sum all delay columns
    if "FLT_ERT_1_DLY" in df.columns:
        df["TOTAL_DELAY"] = df["FLT_ERT_1_DLY"].fillna(0)
    else:
        dly_cols = [c for c in df.columns if c.startswith("DLY_ERT_")]
        df["TOTAL_DELAY"] = df[dly_cols].sum(axis=1, skipna=True)

    # Aggregate to daily totals
    daily = df.groupby("FLT_DATE", as_index=False)["TOTAL_DELAY"].sum()
    daily.rename(columns={"FLT_DATE": "DATE"}, inplace=True)
    daily = daily.set_index("DATE").asfreq("D").fillna(0).reset_index()

    st.write("#### Sample of processed daily data")
    st.dataframe(daily.head(10))

    # ---------------------------
    # 📊 VISUALIZE HISTORICAL DELAYS
    # ---------------------------
    st.subheader("📈 Historical Daily Delays")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["DATE"], daily["TOTAL_DELAY"], color="steelblue", linewidth=1.5)
    ax.set_title("Daily Total En-route ATFM Delay (Minutes)")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Delay (min)")
    ax.grid(True)
    st.pyplot(fig)

    # ---------------------------
    # ⚙️ FEATURE ENGINEERING
    # ---------------------------
    st.subheader("🧠 Model Training (XGBoost)")
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

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ---------------------------
    # 📏 MODEL METRICS
    # ---------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mask = y_test != 0
    mape = (np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:,.0f} min")
    col2.metric("RMSE", f"{rmse:,.0f} min")
    col3.metric("Adjusted MAPE", f"{mape:.2f}%")

    # ---------------------------
    # 🔮 FORECAST NEXT 7 DAYS
    # ---------------------------
    st.subheader("🔮 7-Day Delay Forecast")

    future = df.copy()
    for i in range(7):
        last_date = future["DATE"].iloc[-1] + timedelta(days=1)
        new_data = {
            "day_of_week": last_date.dayofweek,
            "month": last_date.month,
            "is_weekend": 1 if last_date.dayofweek in [5, 6] else 0,
            "lag_1": future["TOTAL_DELAY"].iloc[-1],
            "lag_7": future["TOTAL_DELAY"].iloc[-7] if len(future) >= 7 else future["TOTAL_DELAY"].iloc[-1],
            "rolling_mean_7": future["TOTAL_DELAY"].tail(7).mean(),
            "rolling_std_7": future["TOTAL_DELAY"].tail(7).std()
        }
        y_future = model.predict(pd.DataFrame([new_data]))[0]
        new_data["TOTAL_DELAY"] = y_future
        new_data["DATE"] = last_date
        future = pd.concat([future, pd.DataFrame([new_data])], ignore_index=True)

    future_tail = future.set_index("DATE").tail(20)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(future_tail.index, future_tail["TOTAL_DELAY"], marker="o", label="Forecast + Actual")
    ax.axvline(future["DATE"].iloc[-8], color='red', linestyle='--', label="Forecast Start")
    ax.set_title("7-Day Flight Delay Forecast — XGBoost")
    ax.set_ylabel("Total Delay (minutes)")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

    st.success("✅ Forecast generated successfully!")

else:
    st.info("👆 Upload a EUROCONTROL `.bz2` or `.csv` dataset to start analysis and forecasting.")
