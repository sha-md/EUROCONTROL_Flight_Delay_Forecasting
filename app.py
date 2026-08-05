import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import plotly.express as px

st.set_page_config(
    page_title="EUROCONTROL Flight Delay Analytics",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ EUROCONTROL Flight Delay Analytics & Forecasting")

st.caption(
    "Interactive dashboard for analysing historical EUROCONTROL ANSP delays and forecasting future en-route ATFM delays using XGBoost."
)
# ---------------------------
# 📂 FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload your EUROCONTROL .bz2 or .csv data file", type=["bz2", "csv"])

st.sidebar.title("⚙️ Dashboard")

st.sidebar.markdown("""
**Workflow**

1. Upload dataset
2. Explore historical delays
3. Train forecasting model
4. View forecast
5. Review feature importance
""")

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
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Dashboard",
            "📈 Forecast",
            "🤖 Model"
        ]
    )
    with tab1:
        # ---------------------------
        # Dashboard Filters
        # ---------------------------
        
        filter_col1, filter_col2 = st.columns(2)
        
        available_years = sorted(
            daily["DATE"].dt.year.unique()
        )

        selected_year = filter_col1.selectbox(
            "Year",
            available_years,
            index=len(available_years)-1
        )

        months = [
            "All",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]

        selected_month = filter_col2.selectbox(
            "Month",
            months
        )

        filtered_daily = daily[
            daily["DATE"].dt.year == selected_year
        ]

        if selected_month != "All":
        
            month_number = months.index(selected_month)
        
            filtered_daily = filtered_daily[
                filtered_daily["DATE"].dt.month == month_number
            ]

        if filtered_daily.empty:
            st.warning("No data available for the selected filters.")
            st.stop()
            
        c1,c2,c3,c4 = st.columns(4)

        c1.metric(
            "Observations",
            f"{len(df):,}"
        )

        c2.metric(
            "ANSPs",
            df["ENTITY_NAME"].nunique()
        )

        c3.metric(
            "Date Range",
            f"{df['FLT_DATE'].dt.year.min()}–{df['FLT_DATE'].dt.year.max()}"
        )

        c4.metric(
            "Average Delay",
            f"{filtered_daily['TOTAL_DELAY'].mean():,.0f} min"
        )


        with st.expander("Preview Processed Dataset"):
            st.dataframe(filtered_daily.head(10))

        # ============================
    

   
        # ---------------------------
        # 📊 VISUALIZE HISTORICAL DELAYS
        # ---------------------------
        st.divider()
        st.subheader("Daily Delay Trend")
        title = (
            f"{selected_month} {selected_year} Daily Delay"
            if selected_month != "All"
            else f"Daily Delay Trend ({selected_year})"
        )
        
        fig = px.line(
            filtered_daily,
            x="DATE",
            y="TOTAL_DELAY",
            title=title
        )

        st.plotly_chart(fig, use_container_width=True)
    
        st.metric(
            "Maximum Daily Delay",
            f"{filtered_daily['TOTAL_DELAY'].max():,.0f} min"
        )
        # ============================
        # Monthly Trend
        # ============================
    
        monthly = (
            filtered_daily
            .set_index("DATE")["TOTAL_DELAY"]
            .resample("M")
            .mean()
        )

        st.subheader("Monthly Trend")
    
        fig2 = px.line(
            x=monthly.index,
            y=monthly.values,
            labels={
                "x":"Month",
                "y":"Average Delay (minutes)"
            },
            title="Monthly Average En-route Delay"
        )
    
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------
    # ⚙️ FEATURE ENGINEERING
    # ---------------------------
    with tab3:

        st.subheader("🤖 Model Training & Evaluation")
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

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
    
        st.subheader("Feature Importance")

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )
    
        st.plotly_chart(
            fig,
            use_container_width=True
        )

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
    with tab2:

        st.subheader("📈 7-Day Flight Delay Forecast")
    
        forecast_days = st.slider(
            "Forecast Horizon",
            1,
            30,
            7
        )
    
        future = df.copy()
        for i in range(forecast_days):
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
        forecast_plot = future_tail.reset_index()

        fig = px.line(
            forecast_plot,
            x="DATE",
            y="TOTAL_DELAY",
            markers=True,
            title=f"{forecast_days}-Day Flight Delay Forecast"
        )
        fig.add_vline(
            x=future["DATE"].iloc[-forecast_days],
            line_dash="dash",
            line_color="red"
        )
    
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Forecast generated successfully for the next {forecast_days} days."
        )
        forecast = future.tail(forecast_days)
    
        st.subheader("Forecast Summary")
        st.metric(
            "Average Forecasted Delay",
            f"{forecast['TOTAL_DELAY'].mean():,.0f} min"
        )
    
        avg_delay = forecast["TOTAL_DELAY"].mean()

        if avg_delay > 10000:
            st.error(f"""
         ### Business Insight

        The model forecasts an average daily delay of **{avg_delay:,.0f} minutes** over the next **{forecast_days} days**.
        
        High congestion is expected. Airlines, airports and ANSPs should consider additional operational capacity and proactive traffic management.
        """)       

        elif avg_delay > 5000:
            st.warning(f"""
        ### Business Insight
        
        The model forecasts an average daily delay of **{avg_delay:,.0f} minutes** over the next **{forecast_days} days**.
        
        Moderate congestion is expected. Resource planning and schedule optimisation are recommended.
        """)

        else:
            st.success(f"""
        ### Business Insight
        
        The model forecasts an average daily delay of **{avg_delay:,.0f} minutes** over the next **{forecast_days} days**.
        
        Forecasted delays remain relatively low, indicating relatively stable network conditions.
        """)

        forecast_display = forecast.copy()

        forecast_display["TOTAL_DELAY"] = (
            forecast_display["TOTAL_DELAY"]
            .round(0)
            .astype(int)
        )

        st.dataframe(
            forecast_display,
            use_container_width=True
        )
    
        csv = forecast.to_csv(index=False)
    
        st.download_button(
            "📥 Download Forecast",
            csv,
            "forecast.csv",
            "text/csv"
        )

    

else:
    st.info("👆 Upload a EUROCONTROL `.bz2` or `.csv` dataset to start analysis and forecasting.")
