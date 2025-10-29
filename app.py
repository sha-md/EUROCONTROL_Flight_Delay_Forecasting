import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import numpy as np
import os

# ---------------------------------------
# 🌤️ App Header
# ---------------------------------------
st.set_page_config(page_title="EUROCONTROL Flight Delay Forecasting", layout="wide")
st.title("✈️ EUROCONTROL Flight Delay Forecasting Dashboard (2020–2024)")
st.markdown("""
Forecasting **daily en-route Air Traffic Flow Management (ATFM) delays** using EUROCONTROL ANSP data.
This app demonstrates two approaches: **SARIMA** and **XGBoost** models.
""")

# ---------------------------------------
# 📥 Load Data
# ---------------------------------------
@st.cache_data
def load_data():
    files = [f for f in os.listdir() if f.endswith(".bz2")]
    dfs = []
    for f in files:
        df_year = pd.read_csv(f, compression="bz2")
        dfs.append(df_year)
    df = pd.concat(dfs, ignore_index=True)
    df["FLT_DATE"] = pd.to_datetime(df["FLT_DATE"], errors="coerce")
    df = df.dropna(subset=["FLT_DATE"])
    df = df.sort_values("FLT_DATE")
    df["TOTAL_DELAY"] = df["FLT_ERT_1_DLY"].fillna(0)
    daily = df.groupby("FLT_DATE", as_index=False)["TOTAL_DELAY"].sum()
    daily.rename(columns={"FLT_DATE": "DATE"}, inplace=True)
    daily = daily.set_index("DATE").asfreq("D").fillna(0).reset_index()
    return daily

daily = load_data()

st.subheader("📊 Daily Total Delays (2020–2024)")
st.line_chart(daily.set_index("DATE")["TOTAL_DELAY"])

# ---------------------------------------
# ⚙️ Feature Engineering
# ---------------------------------------
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

# ---------------------------------------
# 🧠 XGBoost Model
# ---------------------------------------
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
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mask = y_test != 0
mape = (np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]).mean() * 100

st.subheader("🧮 Model Performance")
st.write(f"**MAE:** {mae:,.2f} min  |  **RMSE:** {rmse:,.2f} min  |  **MAPE:** {mape:.2f}%")

# ---------------------------------------
# 📈 Visualization
# ---------------------------------------
st.subheader("📈 Actual vs Predicted Daily Delays")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["DATE"].iloc[split_idx:], y_test.values, label="Actual", linewidth=1.5)
ax.plot(df["DATE"].iloc[split_idx:], y_pred, label="Predicted", linewidth=1.5)
ax.legend()
ax.set_title("Actual vs Predicted Daily Delay — XGBoost")
ax.set_xlabel("Date")
ax.set_ylabel("Total Delay (min)")
st.pyplot(fig)

st.markdown("---")
st.caption("Developed by [Your Name] — Forecasting Flight Delays using SARIMA and XGBoost.")
