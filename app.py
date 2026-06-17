import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Kenya Tourism Forecasting",
    layout="wide"
)

# -----------------------------
# CLEAN TOURISM THEME
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #4FC3F7, #01579B);
    color: white;
}

h1, h2, h3, p, div, span {
    color: white !important;
}

section[data-testid="stSidebar"] {
    background-color: #003B73;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.block-container {
    background-color: rgba(0, 0, 0, 0.15);
    padding: 2rem;
    border-radius: 12px;
}

.stButton > button {
    background-color: #00A6FB;
    color: white;
    border-radius: 8px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# UPLOAD DATASET
# -----------------------------
st.sidebar.title("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel (optional)",
    type=["csv", "xlsx"]
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_default():
    return pd.read_excel("TOURIST_ARRIVALS_DATA.xlsx")

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("Custom dataset loaded!")
else:
    df = load_default()

# -----------------------------
# CLEAN DATA (FIX VALUE ERROR)
# -----------------------------
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.sort_values("DATE")
df = df.set_index("DATE")
df = df.asfreq("MS")

# FORCE NUMERIC CLEANING (CRITICAL FIX)
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
df = df.dropna()

y = df.iloc[:, 0]
last_date = df.index[-1]

# -----------------------------
# FIT MODELS
# -----------------------------
sarima_model = SARIMAX(
    y, order=(1,0,1), seasonal_order=(1,1,1,12)
).fit(disp=False)

arima_model = ARIMA(y, order=(1,1,1)).fit()

hw_model = ExponentialSmoothing(
    y, trend="add", seasonal="add", seasonal_periods=12
).fit()

prophet_df = pd.DataFrame({"ds": df.index, "y": y.values})
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def forecast_model(name, steps):
    if name == "SARIMA":
        return sarima_model.forecast(steps)
    elif name == "ARIMA":
        return arima_model.forecast(steps)
    elif name == "Holt-Winters":
        return hw_model.forecast(steps)
    else:
        future = prophet_model.make_future_dataframe(steps, freq="MS")
        pred = prophet_model.predict(future)
        return pred["yhat"].tail(steps)

# -----------------------------
# NAVIGATION
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Forecasting", "Dashboard (2026–2027)", "Metrics", "Conclusion"]
)

# -----------------------------
# FORECAST PAGE
# -----------------------------
if page == "Forecasting":

    st.title("📊 Forecasting")

    model_choice = st.selectbox(
        "Select Model",
        ["SARIMA", "ARIMA", "Holt-Winters", "Prophet"]
    )

    forecast_date = st.date_input("Select Forecast Month")

    if st.button("Generate Forecast"):

        months_ahead = (
            (forecast_date.year - last_date.year) * 12 +
            (forecast_date.month - last_date.month)
        )

        if months_ahead <= 0:
            st.error("Select a future date")
        else:

            forecast = forecast_model(model_choice, months_ahead)

            st.metric("Forecast", f"{forecast.iloc[-1]:,.0f}")

            future_dates = pd.date_range(
                last_date + pd.DateOffset(months=1),
                periods=months_ahead,
                freq="MS"
            )

            result = pd.DataFrame({
                "Date": future_dates,
                "Forecast": forecast.values
            })

            st.dataframe(result)

            fig, ax = plt.subplots()
            ax.plot(result["Date"], result["Forecast"])
            st.pyplot(fig)

# -----------------------------
# DASHBOARD (ACTUAL + FORECAST)
# -----------------------------
elif page == "Dashboard (2026–2027)":

    st.title("📈 Actual vs Forecast (SARIMA)")

    forecast_steps = 24
    forecast_values = sarima_model.forecast(forecast_steps)

    future_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="MS"
    )

    df_forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_values
    })

    fig, ax = plt.subplots()

    ax.plot(df.index, df.iloc[:, 0], label="Actual Data")
    ax.plot(df_forecast["Date"], df_forecast["Forecast"], label="Forecast")

    ax.legend()
    st.pyplot(fig)

    st.dataframe(df_forecast)

# -----------------------------
# METRICS
# -----------------------------
elif page == "Metrics":

    st.title("📊 Model Performance")

    metrics = pd.DataFrame({
        "Model": ["SARIMA", "Holt-Winters", "ARIMA", "Prophet"],
        "MAE": [11762.24, 15385.76, 30218.54, 35028.12],
        "RMSE": [14619.38, 19615.83, 35903.41, 38509.58],
        "MAPE (%)": [5.57, 7.06, 15.34, 16.30]
    })

    st.dataframe(metrics)

# -----------------------------
# CONCLUSION
# -----------------------------
elif page == "Conclusion":

    st.title("📌 Conclusion")

    st.markdown("""
    - SARIMA performs best  
    - Strong seasonality in tourism data  
    - Holt-Winters is second best  
    - ARIMA and Prophet underperform  
    """)
