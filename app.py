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

    /* MAIN BACKGROUND (SKY BLUE GRADIENT) */
    .stApp {
        background: linear-gradient(to bottom right, #4FC3F7, #01579B);
        color: white;
    }

    /* FIX TEXT COLOR (VERY IMPORTANT) */
    h1, h2, h3, p, div, span {
        color: white !important;
    }

    /* SIDEBAR (OCEAN BLUE) */
    section[data-testid="stSidebar"] {
        background-color: #003B73;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* MAIN CONTAINER (GLASS EFFECT - NOT BLUR) */
    .block-container {
        background-color: rgba(0, 0, 0, 0.15);
        padding: 2rem;
        border-radius: 12px;
    }

    /* BUTTONS (OCEAN GREENISH BLUE) */
    .stButton > button {
        background-color: #00A6FB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #0077B6;
    }

    /* METRICS CARDS */
    div[data-testid="metric-container"] {
        background-color: rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.3);
        color: white !important;
    }

    /* TABLES */
    table {
        color: white !important;
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

    # LOAD EXCEL FILE
    df_forecast = pd.read_excel(
        "tourist_arrivals_forecast_2026_2027.xlsx",
        engine="openpyxl"
    )

    # FIX COLUMN ISSUES FROM COLAB EXPORT
    if "Unnamed: 0" in df_forecast.columns:
        df_forecast = df_forecast.drop(columns=["Unnamed: 0"])

    # Rename columns safely
    df_forecast.columns = ["Date", "Forecast", "Lower", "Upper"]

    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

    # PLOT
    fig, ax = plt.subplots()

    ax.plot(df_forecast["Date"], df_forecast["Forecast"], color="#00A6FB")

    ax.fill_between(
        df_forecast["Date"],
        df_forecast["Lower"],
        df_forecast["Upper"],
        color="skyblue",
        alpha=0.3
    )

    ax.set_title("Kenya Tourism Forecast (2026–2027)")
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
    
