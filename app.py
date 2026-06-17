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
# LOAD FORECASTS (CSV ONLY - STABLE)
# -----------------------------
sarima_forecast = pd.read_csv("sarima_forecast.csv")
arima_forecast = pd.read_csv("arima_forecast.csv")
hw_forecast = pd.read_csv("hw_forecast.csv")
prophet_forecast = pd.read_csv("prophet_forecast.csv")

# Ensure correct column name fallback
def get_forecast(df):
    if "Forecast" in df.columns:
        return df["Forecast"].values
    elif df.shape[1] >= 1:
        return df.iloc[:, 0].values
    else:
        return []

# -----------------------------
# SIDEBAR MENU
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "Model Comparison", "Forecasting", "Conclusion"]
)

# -----------------------------
# PAGE 1: DATASET OVERVIEW
# -----------------------------
if page == "Dataset Overview":
    st.header("📌 Dataset Overview")

    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Tourist Arrivals Trend")

    fig, ax = plt.subplots()
    ax.plot(df.index, df.iloc[:, 0])
    ax.set_title("Monthly Tourist Arrivals (2019–2023)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Arrivals")
    st.pyplot(fig)

    st.subheader("Summary Statistics")
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

    st.success("🏆 Best Performing Model: SARIMA")

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

        if model_choice == "SARIMA":
            forecast = get_forecast(sarima_forecast)

        elif model_choice == "ARIMA":
            forecast = get_forecast(arima_forecast)

        elif model_choice == "Holt-Winters":
            forecast = get_forecast(hw_forecast)

        else:
            forecast = get_forecast(prophet_forecast)

        future_dates = pd.date_range(
            start=df.index[-1],
            periods=periods + 1,
            freq='MS'
        )[1:]

        result = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast[:periods]
        })

        st.subheader(f"{model_choice} Forecast Results")
        st.dataframe(result)

        fig, ax = plt.subplots()
        ax.plot(result["Date"], result["Forecast"], marker="o")
        ax.set_title(f"{model_choice} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Tourist Arrivals")
        plt.xticks(rotation=45)
        st.pyplot(fig)

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
- Holt-Winters captured smoothing effects well
- ARIMA underperformed due to seasonality
- Prophet was less effective for small yearly dataset

### 🏆 Final Decision:
**SARIMA is the recommended model for forecasting tourism arrivals in Kenya.**
""")
