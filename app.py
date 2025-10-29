import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# 🎨 PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="EUROCONTROL Delay Forecasting",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------
# 🏷️ HEADER
# ---------------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 2.2em;
    color: #0d47a1;
    font-weight: 800;
    background: linear-gradient(90deg, #0052D4, #65C7F7, #9CECFB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subheader {
    text-align: center;
    font-size: 1.1em;
    color: #424242;
}
.upload {
    background-color: #E3F2FD;
    padding: 1em;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">✈️ EUROCONTROL Flight Delay Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Forecast daily en-route ATFM delays across Europe (2020–2024)<br>using a tuned XGBoost time-series model.</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------
# 📤 FILE UPLOAD
# ---------------------------
st.markdown('<div class="upload">Upload your EUROCONTROL .bz2 or .csv data file below:</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Upload delay data", type=["bz2", "csv"])

# ---------------------------
# ⚡ DEMO DATA OPTION
# ---------------------------
if st.button("🧪 Load Demo Data (2024 Sample)"):
    demo_url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_flights.csv"  # Just a placeholder
    st.info("Using demo sample for preview — replace with your EUROCONTROL dataset for full forecasting.")
    demo_df = pd.DataFrame({
        "FLT_DATE": pd.date_range("2024-01-01", periods=365, freq="D"),
        "FLT_ERT_1_DLY": np.random.randint(500, 10000, 365)
    })
    uploaded_file = "demo"
    df = demo_df.copy()
else:
    df = None

# ---------------------------
# 🧹 LOAD & CLEAN DATA
# ---------------------------
if uploaded_file is not None:
    if uploaded_file != "demo":
        if uploaded_file.name.endswith(".bz2"):
            df = pd.read_csv(uploaded_file, compression="bz2")
        else:
            df = pd.read_csv(uploaded_file)

    st.success("✅ Data loaded successfully!")

    # Parse & clean
    df["FLT_DATE"] = pd.to_datetime(df["FLT_DATE"], errors="coerce")
    df = df.dropna(subset=["FLT_DATE"]).sort_values("FLT_DATE")

    if "FLT_ERT_1_DLY" in df.columns:
        df["TOTAL_DELAY"] = df["FLT_ERT_1_DLY"].fillna(0)
    else:
        delay_cols = [c for c in df.columns if c.startswith("DLY_ERT_")]
        df["TOTAL_DELAY"] = df[delay_cols].sum(axis=1, skipna=True)

    daily = df.groupby("FLT_DATE", as_index=False)["TOTAL_DELAY"].sum()
    daily.rename(columns={"FLT_DATE": "DATE"}, inplace=True)
    daily = daily.set_index("DATE").asfreq("D").fillna(0).reset_index()

    # ---------------------------
    # 📊 VISUALIZATION
    # ---------------------------
    st.subheader("📈 Historical Delay Trends")
    fig = px.line(
        daily,
        x="DATE",
        y="TOTAL_DELAY",
        title="Daily Total En-route ATFM Delay (Minutes)",
        template="plotly_white",
        line_shape="spline"
    )
    fig.update_traces(line_color="#0d47a1", line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # ⚙️ FEATURE ENGINEERING
    # ---------------------------
    df = daily.copy()
    df["day_of_week"] = df["DATE"].dt.dayofweek
    df["month"] = df["DATE"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["lag_1"] = df["TOTAL_DELAY"].shift(1)
    df["lag_7"] = df["TOTAL_DELAY"].shift(7)
    df["rolling_mean_7"] = df["TOTAL_DELAY"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["TOTAL_DELAY"].shift(1).rolling(7).std()
    df = df.dropna().reset_index(drop=True)

    FEATURES = ["day_of_week","month","is_weekend","lag_1","lag_7","rolling_mean_7","rolling_std_7"]
    X, y = df[FEATURES], df["TOTAL_DELAY"]

    split_idx = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ---------------------------
    # 🚀 TRAIN MODEL
    # ---------------------------
    with st.spinner("Training XGBoost model..."):
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
    st.success("✅ Model trained successfully!")

    # ---------------------------
    # 📏 METRICS
    # ---------------------------
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mask = y_test != 0
    mape = (np.abs(y_test[mask] - y_pred[mask]) / y_test[mask]).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("📊 MAE", f"{mae:,.0f} min")
    c2.metric("📉 RMSE", f"{rmse:,.0f} min")
    c3.metric("📈 Adjusted MAPE", f"{mape:.2f}%")

    st.divider()

    # ---------------------------
    # 🔮 FORECAST NEXT 7 DAYS
    # ---------------------------
    st.subheader("🔮 7-Day Forecast Visualization")

    future = df.copy()
    for i in range(7):
        last_date = future["DATE"].iloc[-1] + timedelta(days=1)
        new_data = {
            "day_of_week": last_date.dayofweek,
            "month": last_date.month,
            "is_weekend": 1 if last_date.dayofweek in [5,6] else 0,
            "lag_1": future["TOTAL_DELAY"].iloc[-1],
            "lag_7": future["TOTAL_DELAY"].iloc[-7] if len(future) >= 7 else future["TOTAL_DELAY"].iloc[-1],
            "rolling_mean_7": future["TOTAL_DELAY"].tail(7).mean(),
            "rolling_std_7": future["TOTAL_DELAY"].tail(7).std()
        }
        y_future = model.predict(pd.DataFrame([new_data]))[0]
        new_data["TOTAL_DELAY"] = y_future
        new_data["DATE"] = last_date
        future = pd.concat([future, pd.DataFrame([new_data])], ignore_index=True)

    future_plot = future.set_index("DATE").tail(50).reset_index()
    fig_forecast = px.line(
        future_plot,
        x="DATE",
        y="TOTAL_DELAY",
        title="Next 7-Day Forecast (XGBoost)",
        template="plotly_white",
        markers=True
    )
    fig_forecast.add_vline(x=future_plot["DATE"].iloc[-8], line_dash="dash", line_color="red")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ---------------------------
    # 💾 DOWNLOAD FORECAST
    # ---------------------------
    st.download_button(
        label="📥 Download 7-Day Forecast as CSV",
        data=future_plot.to_csv(index=False).encode('utf-8'),
        file_name="7_day_flight_delay_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload or load demo data to start forecasting.")
