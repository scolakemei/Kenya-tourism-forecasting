import io
import calendar

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
# DATA LOADING HELPERS
# -----------------------------
MIN_MONTHS_FOR_SEASONAL_MODELS = 25  # need > 2 full seasonal cycles for period=12


def _finalize_series(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, set monthly frequency, and fill any gaps so models don't choke on NaNs."""
    df = df.sort_values("DATE").set_index("DATE")
    df = df.asfreq("MS")
    df["TOURIST ARRIVALS"] = df["TOURIST ARRIVALS"].interpolate(limit_direction="both")
    return df


@st.cache_data
def load_default_data():
    df = pd.read_excel("TOURIST_ARRIVALS_DATA.xlsx")
    df["DATE"] = pd.to_datetime(df["DATE"])
    return _finalize_series(df)


@st.cache_data
def load_uploaded_data(file_bytes: bytes, filename: str, date_col: str, value_col: str):
    if filename.lower().endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw = pd.read_excel(io.BytesIO(file_bytes))

    df = raw[[date_col, value_col]].rename(
        columns={date_col: "DATE", value_col: "TOURIST ARRIVALS"}
    )
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["TOURIST ARRIVALS"] = pd.to_numeric(df["TOURIST ARRIVALS"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    return _finalize_series(df)


@st.cache_data
def peek_columns(file_bytes: bytes, filename: str):
    if filename.lower().endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw = pd.read_excel(io.BytesIO(file_bytes))
    return list(raw.columns)


# -----------------------------
# SIDEBAR: DATA SOURCE SELECTION
# -----------------------------
st.sidebar.markdown("## 📂 Data Source")

data_source = st.sidebar.radio(
    "Choose dataset",
    ["Use Default KTB Dataset", "Upload My Own Dataset"]
)

df = None

if data_source == "Upload My Own Dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        try:
            columns = peek_columns(file_bytes, uploaded_file.name)
        except Exception as e:
            columns = None
            st.sidebar.error(f"Could not read file: {e}")

        if columns:
            date_col = st.sidebar.selectbox("Date column", columns, key="date_col_select")
            value_col_options = [c for c in columns if c != date_col] or columns
            value_col = st.sidebar.selectbox(
                "Tourist arrivals column", value_col_options, key="value_col_select"
            )

            if st.sidebar.button("Load Dataset"):
                try:
                    loaded_df = load_uploaded_data(
                        file_bytes, uploaded_file.name, date_col, value_col
                    )
                    if len(loaded_df) < MIN_MONTHS_FOR_SEASONAL_MODELS:
                        st.sidebar.error(
                            f"This dataset has only {len(loaded_df)} monthly records. "
                            f"At least {MIN_MONTHS_FOR_SEASONAL_MODELS} months are needed "
                            "for the seasonal models (SARIMA / Holt-Winters) to fit reliably."
                        )
                    else:
                        st.session_state["active_df"] = loaded_df
                        st.session_state["active_source"] = "uploaded"
                        st.sidebar.success(f"Loaded {len(loaded_df)} monthly records.")
                except Exception as e:
                    st.sidebar.error(f"Error processing file: {e}")

    if st.session_state.get("active_source") == "uploaded" and "active_df" in st.session_state:
        df = st.session_state["active_df"]
    else:
        st.info(
            "⬅️ Upload a dataset in the sidebar (with a date column and a tourist "
            "arrivals column), then click **Load Dataset** to continue."
        )
        st.stop()

else:
    df = load_default_data()
    st.session_state["active_df"] = df
    st.session_state["active_source"] = "default"

y = df["TOURIST ARRIVALS"]
last_date = df.index[-1]

# -----------------------------
# FIT MODELS (cached per dataset, refit only when the data changes)
# -----------------------------
@st.cache_resource
def fit_models(series: pd.Series):
    sarima = SARIMAX(
        series, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12)
    ).fit(disp=False)

    arima = ARIMA(series, order=(1, 1, 1)).fit()

    hw = ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=12
    ).fit()

    prophet_df = pd.DataFrame({"ds": series.index, "y": series.values})
    prophet = Prophet()
    prophet.fit(prophet_df)

    return sarima, arima, hw, prophet


with st.spinner("Training forecasting models on the selected dataset..."):
    sarima_model, arima_model, hw_model, prophet_model = fit_models(y)


# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def forecast_model(steps, name):
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

    # ---- Month / Year selector (replaces the day-level date picker) ----
    month_names = list(calendar.month_name)[1:]  # ["January", ..., "December"]

    default_target = last_date + pd.DateOffset(months=1)

    col1, col2 = st.columns(2)
    with col1:
        selected_month_name = st.selectbox(
            "Forecast Month",
            month_names,
            index=default_target.month - 1
        )
    with col2:
        year_options = list(range(last_date.year, last_date.year + 11))
        default_year_index = (
            year_options.index(default_target.year)
            if default_target.year in year_options
            else 0
        )
        selected_year = st.selectbox(
            "Forecast Year",
            year_options,
            index=default_year_index
        )

    selected_month = month_names.index(selected_month_name) + 1

    if st.button("Generate Forecast"):

        months_ahead = (
            (selected_year - last_date.year) * 12 +
            (selected_month - last_date.month)
        )

        if months_ahead <= 0:
            st.error("Please select a month/year after the last available data point "
                      f"({last_date.strftime('%B %Y')}).")
        else:

            forecast = forecast_model(months_ahead, model_choice)

            prediction = forecast.iloc[-1]

            st.metric(f"Forecast for {selected_month_name} {selected_year}", f"{prediction:,.0f}")

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

    st.title("📈 Actual vs Forecast (SARIMA)")

    # -----------------------------
    # LOAD DATASET (ACTUAL DATA)
    # -----------------------------
    df_actual = df.copy()

    # -----------------------------
    # FORECAST (24 MONTHS)
    # -----------------------------
    forecast_steps = 24
    forecast_values = sarima_model.forecast(forecast_steps)

    future_dates = pd.date_range(
        start=df_actual.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="MS"
    )

    df_forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_values
    })

    # -----------------------------
    # PLOT BOTH TOGETHER
    # -----------------------------
    fig, ax = plt.subplots()

    # ACTUAL DATA
    ax.plot(
        df_actual.index,
        df_actual["TOURIST ARRIVALS"],
        label="Actual Data",
        color="#003B73",
        linewidth=2
    )

    # FORECAST DATA
    ax.plot(
        df_forecast["Date"],
        df_forecast["Forecast"],
        label="SARIMA Forecast",
        color="#00A6FB",
        linewidth=2
    )

    ax.set_title("Kenya Tourism: Actual vs Forecast (SARIMA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tourist Arrivals")

    ax.legend()

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # -----------------------------
    # TABLE OUTPUT
    # -----------------------------
    st.subheader("Forecast Table")
    st.dataframe(df_forecast)

# -----------------------------
# METRICS PAGE
# -----------------------------
elif page == "📋 Metrics":

    st.title("📊 Model Performance")

    if st.session_state.get("active_source") == "uploaded":
        st.caption(
            "ℹ️ These performance metrics were computed on the original KTB dataset "
            "and are shown for reference only. They are not recalculated for "
            "uploaded datasets."
        )

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
