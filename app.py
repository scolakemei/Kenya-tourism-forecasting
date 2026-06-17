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
# TOURISM THEME (NO IMAGE)
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
        color: white;
    }

    div[data-testid="metric-container"] {
        background-color: #E8F5E9;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #C8E6C9;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Kenya Tourism Forecasting Dashboard")
st.caption("Forecasting tourist arrivals using ARIMA, SARIMA, Holt-Winters, and Prophet")

# -----------------------------
# LOAD DATA (NO UPLOAD)
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

# -----------------------------
# DATA OVERVIEW
# -----------------------------
st.subheader("📌 Dataset Overview")
st.dataframe(df)

fig, ax = plt.subplots()
ax.plot(df.index, df["TOURIST ARRIVALS"])
ax.set_title("Tourist Arrivals Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Arrivals")
st.pyplot(fig)

# -----------------------------
# MODEL SELECTION
# -----------------------------
model_choice = st.selectbox(
    "Select Forecast Model",
    ["SARIMA", "ARIMA", "Holt-Winters", "Prophet"]
)

forecast_date = st.date_input("Select Forecast Month (e.g. Jan 2026)")

# -----------------------------
# FORECAST BUTTON
# -----------------------------
if st.button("Generate Forecast"):

    y = df["TOURIST ARRIVALS"]
    last_date = df.index[-1]

    months_ahead = (
        (forecast_date.year - last_date.year) * 12 +
        (forecast_date.month - last_date.month)
    )

    if months_ahead <= 0:
        st.error("Please select a future date beyond dataset range")

    else:

        # ---------------- SARIMA ----------------
        if model_choice == "SARIMA":

            model = SARIMAX(
                y,
                order=(1,0,1),
                seasonal_order=(1,1,1,12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fit = model.fit(disp=False)
            forecast = fit.forecast(steps=months_ahead)

        # ---------------- ARIMA ----------------
        elif model_choice == "ARIMA":

            model = ARIMA(y, order=(1,1,1))
            fit = model.fit()
            forecast = fit.forecast(steps=months_ahead)

        # ---------------- HOLT WINTERS ----------------
        elif model_choice == "Holt-Winters":

            model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=12
            )

            fit = model.fit()
            forecast = fit.forecast(months_ahead)

        # ---------------- PROPHET ----------------
        else:

            prophet_df = pd.DataFrame({
                "ds": df.index,
                "y": y.values
            })

            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(
                periods=months_ahead,
                freq="MS"
            )

            forecast_df = model.predict(future)
            forecast = forecast_df["yhat"].tail(months_ahead)

        # ---------------- RESULT ----------------
        prediction = forecast.iloc[-1]

        st.success(f"Forecast for {forecast_date.strftime('%B %Y')}")

        st.metric(
            "Predicted Tourist Arrivals",
            f"{prediction:,.0f}"
        )

        # ---------------- TABLE ----------------
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=months_ahead,
            freq="MS"
        )

        result = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast.values
        })

        st.subheader("Forecast Results")
        st.dataframe(result)

        # ---------------- PLOT ----------------
        fig, ax = plt.subplots()
        ax.plot(result["Date"], result["Forecast"], marker="o")
        ax.set_title(f"{model_choice} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Tourist Arrivals")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ---------------- DOWNLOAD ----------------
        csv = result.to_csv(index=False)

        st.download_button(
            "📥 Download Forecast",
            csv,
            "forecast.csv",
            "text/csv"
        )

else:
    st.info("Please upload a dataset to begin forecasting.")
