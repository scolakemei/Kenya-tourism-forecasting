import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64

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
# BACKGROUND IMAGE
# -----------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(255,255,255,0.88);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background image (must exist in repo)
set_bg("tourism_bg.jpg")

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Kenya Tourism Forecasting System")
st.write("Upload dataset and forecast tourist arrivals using ML models")

# -----------------------------
# UPLOAD DATA
# -----------------------------
uploaded_file = st.file_uploader("Upload Dataset (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file is not None:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # -----------------------------
    # DATA PREPARATION
    # -----------------------------
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")
    df = df.set_index("DATE")
    df = df.asfreq("MS")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # USER INPUT
    # -----------------------------
    model_choice = st.selectbox(
        "Select Forecasting Model",
        ["SARIMA", "ARIMA", "Holt-Winters", "Prophet"]
    )

    forecast_date = st.date_input("Select Forecast Month (e.g. Jan 2026)")

    # -----------------------------
    # FORECAST BUTTON
    # -----------------------------
    if st.button("Generate Forecast"):

        y = df["TOURIST ARRIVALS"]
        last_date = df.index[-1]

        # months ahead calculation
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
                label="Predicted Tourist Arrivals",
                value=f"{prediction:,.0f}"
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

            st.subheader("Forecast Table")
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
**SARIMA is the recommended model for Kenya tourism forecasting.**
""")
