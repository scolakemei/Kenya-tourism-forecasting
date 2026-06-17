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
# CLEAN TOURISM THEME (NO BLUR ISSUE FIXED)
# -----------------------------
st.markdown(
    """
    <style>

    /* MAIN BACKGROUND - CLEAN SOLID (NO OVERLAY BLUR) */
    .stApp {
        background-color: #F5F7FA;
    }

    /* REMOVE ANY TRANSPARENT LAYER ISSUES */
    .block-container {
        background-color: #F5F7FA;
        padding: 2rem;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0B3D2E;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* HEADINGS */
    h1 {
        color: #0B3D2E;
        font-weight: 700;
    }

    h2, h3 {
        color: #1B5E20;
    }

    /* BUTTONS */
    .stButton > button {
        background-color: #1B7F5A;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
    }

    .stButton > button:hover {
        background-color: #145C42;
    }

    /* METRICS */
    div[data-testid="metric-container"] {
        background-color: #E8F5E9;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #C8E6C9;
    }

    /* TABLE FIX */
    table {
        color: black !important;
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
# FIT MODELS ONCE
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
def forecast_model(model, steps, name):
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
# SIDEBAR NAVIGATION
# -----------------------------
page = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Forecasting", "📊 Dashboard (2026–2027)", "📋 Metrics", "📌 Conclusion"]
)

# -----------------------------
# MAIN FORECAST PAGE
# -----------------------------
if page == "🏠 Forecasting":

    st.title("📊 Kenya Tourism Forecasting")

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

            forecast = forecast_model(None, months_ahead, model_choice)

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
# DASHBOARD PAGE (FIXED CHART DISPLAY)
# -----------------------------
elif page == "📊 Dashboard (2026–2027)":

    st.title("📈 SARIMA Forecast (2026–2027)")

    # EXPECTS CSV FROM YOUR COLAB OUTPUT
    df_forecast = pd.read_csv(
        "sarima_forecast/tourist_arrivals_forecast_2026_2027.csv"
    )

    df_forecast.columns = ["Date", "Forecast", "Lower", "Upper"]
    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

    fig, ax = plt.subplots()

    ax.plot(df_forecast["Date"], df_forecast["Forecast"], color="#1B7F5A")

    ax.fill_between(
        df_forecast["Date"],
        df_forecast["Lower"],
        df_forecast["Upper"],
        color="lightgreen",
        alpha=0.3
    )

    ax.set_title("Tourism Forecast (2026–2027) - SARIMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tourist Arrivals")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.dataframe(df_forecast)

# -----------------------------
# METRICS PAGE
# -----------------------------
elif page == "📋 Metrics":

    st.title("📊 Model Performance")

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
    - SARIMA is the best performing model  
    - Strong seasonality in tourism data  
    - Holt-Winters performs moderately well  
    - ARIMA & Prophet underperform  

    ### 🏆 Recommendation:
    SARIMA is the most suitable model for Kenya tourism forecasting
    """)
    
