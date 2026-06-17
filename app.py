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
# CLEAN THEME
# -----------------------------
st.markdown(
    """
    <style>

    .stApp {
        background-color: #F5F7FA;
    }

    section[data-testid="stSidebar"] {
        background-color: #0B3D2E;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    h1, h2, h3 {
        color: #0B3D2E;
    }

    .stButton > button {
        background-color: #1B7F5A;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stButton > button:hover {
        background-color: #145C42;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("TOURIST_ARRIVALS_DATA.xlsx")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")
    df = df.set_index("DATE")
    df = df.asfreq("MS")
    return df

df = load_data()

y = df["TOURIST ARRIVALS"]
last_date = df.index[-1]

# -----------------------------
# MODEL FIT (ONCE)
# -----------------------------
sarima_model = SARIMAX(y, order=(1,0,1), seasonal_order=(1,1,1,12)).fit(disp=False)
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
def generate_forecast(months_ahead, model_name):

    if model_name == "SARIMA":
        return sarima_model.forecast(months_ahead)

    elif model_name == "ARIMA":
        return arima_model.forecast(months_ahead)

    elif model_name == "Holt-Winters":
        return hw_model.forecast(months_ahead)

    else:
        future = prophet_model.make_future_dataframe(months_ahead, freq="MS")
        pred = prophet_model.predict(future)
        return pred["yhat"].tail(months_ahead)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
page = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Main Forecast", "📊 Dashboard (2026–2027)", "📋 Metrics", "📌 Conclusion"]
)

# -----------------------------
# MAIN PAGE (FORECASTING)
# -----------------------------
if page == "🏠 Main Forecast":

    st.title("📊 Kenya Tourism Forecasting System")

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
            st.error("Choose a future date")
        else:

            forecast = generate_forecast(months_ahead, model_choice)

            prediction = forecast.iloc[-1]

            st.metric("Forecast", f"{prediction:,.0f}")

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
            ax.plot(result["Date"], result["Forecast"], marker="o")
            ax.set_title(f"{model_choice} Forecast")
            st.pyplot(fig)

# -----------------------------
# DASHBOARD PAGE (2026–2027)
# -----------------------------
elif page == "📊 Dashboard (2026–2027)":

    st.title("📈 Forecast Overview (2026–2027)")

    forecast = sarima_model.forecast(24)

    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=24,
        freq="MS"
    )

    fig, ax = plt.subplots()
    ax.plot(future_dates, forecast)
    ax.set_title("Tourism Forecast 2026–2027 (SARIMA)")
    st.pyplot(fig)

# -----------------------------
# METRICS PAGE
# -----------------------------
elif page == "📋 Metrics":

    st.title("📊 Model Performance Metrics")

    metrics = pd.DataFrame({
        "Model": ["SARIMA", "Holt-Winters", "ARIMA", "Prophet"],
        "MAE": [11762.24, 15385.76, 30218.54, 35028.12],
        "RMSE": [14619.38, 19615.83, 35903.41, 38509.58],
        "MAPE (%)": [5.57, 7.06, 15.34, 16.30]
    })

    st.dataframe(metrics)

# -----------------------------
# CONCLUSION PAGE
# -----------------------------
elif page == "📌 Conclusion":

    st.title("📌 Conclusion")

    st.markdown("""
    - SARIMA performed best across all metrics  
    - Tourism data shows strong seasonality  
    - Holt-Winters performed moderately well  
    - ARIMA and Prophet underperformed  

    ### 🏆 Final Recommendation:
    **SARIMA is the best model for forecasting tourism arrivals in Kenya**
    """)
