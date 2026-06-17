import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Kenya Tourism Forecasting",
    layout="wide"
)

st.title("📊 Kenya Tourism Forecasting Dashboard")
st.write("Comparison of ARIMA, SARIMA, Prophet, and Holt-Winters forecasting models.")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("TOURIST_ARRIVALS_DATA.xlsx")
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    df = df.asfreq('MS')
    return df

df = load_data()

# -----------------------------
# LOAD FORECAST CSV FILES
# -----------------------------
sarima_forecast = pd.read_csv("sarima_forecast.csv")
arima_forecast = pd.read_csv("arima_forecast.csv")
hw_forecast = pd.read_csv("hw_forecast.csv")
prophet_forecast = pd.read_csv("prophet_forecast.csv")

# -----------------------------
# FIX FORECAST EXTRACTION (IMPORTANT)
# -----------------------------
def get_forecast_values(df):
    # Your SARIMA file uses "predicted_mean"
    if "predicted_mean" in df.columns:
        return df["predicted_mean"].values

    # Prophet or cleaned CSVs
    if "Forecast" in df.columns:
        return df["Forecast"].values

    # fallback (safe)
    return df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "Model Comparison", "Forecasting", "Conclusion"]
)

# -----------------------------
# PAGE 1: DATASET
# -----------------------------
if page == "Dataset Overview":
    st.header("📌 Dataset Overview")

    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.plot(df.index, df.iloc[:, 0])
    ax.set_title("Monthly Tourist Arrivals (2019–2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Arrivals")
    st.pyplot(fig)

    st.write(df.iloc[:, 0].describe())

# -----------------------------
# PAGE 2: MODEL COMPARISON
# -----------------------------
elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")

    metrics = pd.DataFrame({
        "Model": ["SARIMA", "Holt-Winters", "ARIMA", "Prophet"],
        "MAE": [11762.24, 15385.76, 30218.54, 35028.12],
        "RMSE": [14619.38, 19615.83, 35903.41, 38509.58],
        "MAPE (%)": [5.57, 7.06, 15.34, 16.30]
    })

    st.dataframe(metrics)

    st.success("🏆 Best Model: SARIMA")

    fig, ax = plt.subplots()
    ax.bar(metrics["Model"], metrics["MAPE (%)"])
    ax.set_title("MAPE Comparison")
    ax.set_ylabel("MAPE (%)")
    st.pyplot(fig)

# -----------------------------
# PAGE 3: FORECASTING
# -----------------------------
elif page == "Forecasting":
    st.header("🔮 Forecasting Section")

    model_choice = st.selectbox(
        "Select Model",
        ["SARIMA", "ARIMA", "Prophet", "Holt-Winters"]
    )

    periods = st.slider("Forecast Horizon (Months)", 1, 24, 12)

    if st.button("Generate Forecast"):

        # -------------------------
        # SELECT FORECAST DATA
        # -------------------------
        if model_choice == "SARIMA":
            forecast = get_forecast_values(sarima_forecast)

        elif model_choice == "ARIMA":
            forecast = get_forecast_values(arima_forecast)

        elif model_choice == "Holt-Winters":
            forecast = get_forecast_values(hw_forecast)

        else:
            forecast = get_forecast_values(prophet_forecast)

        # -------------------------
        # CREATE FUTURE DATES
        # -------------------------
        future_dates = pd.date_range(
            start=df.index[-1] + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )

        # -------------------------
        # BUILD RESULT TABLE
        # -------------------------
        result = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast[:periods]
        })

        st.subheader(f"{model_choice} Forecast Results")
        st.dataframe(result)

        # -------------------------
        # PLOT
        # -------------------------
        fig, ax = plt.subplots()
        ax.plot(result["Date"], result["Forecast"], marker="o")
        ax.set_title(f"{model_choice} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Tourist Arrivals")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # -------------------------
        # DOWNLOAD
        # -------------------------
        csv = result.to_csv(index=False)

        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "forecast.csv",
            "text/csv"
        )

# -----------------------------
# PAGE 4: CONCLUSION
# -----------------------------
elif page == "Conclusion":
    st.header("📌 Key Findings")

    st.markdown("""
- SARIMA performed best with lowest error (MAPE = 5.57%)
- Holt-Winters captured seasonality well
- ARIMA underperformed due to weak seasonal handling
- Prophet struggled with small dataset size

### 🏆 Final Decision:
**SARIMA is the recommended model for Kenya tourism forecasting.**
""")g tourism arrivals in Kenya.**
""")
