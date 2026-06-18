import io
import calendar

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
st.markdown(
    """
    <style>

    /* MAIN BACKGROUND (SKY BLUE GRADIENT) */
    .stApp {
        background: linear-gradient(to bottom right, #4FC3F7, #01579B);
        color: white;
    }

    /* FIX TEXT COLOR */
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

    /* MAIN CONTAINER */
    .block-container {
        background-color: rgba(0, 0, 0, 0.15);
        padding: 2rem;
        border-radius: 12px;
    }

    /* BUTTONS */
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

    /* DOWNLOAD BUTTON */
    .stDownloadButton > button {
        background-color: #0077B6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    /* SLIDER */
    div[data-testid="stSlider"] * {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# DATA LOADING HELPERS
# -----------------------------
MIN_MONTHS_FOR_SEASONAL_MODELS = 25


def _finalize_series(df: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_data
def load_pbi_purpose_data():
    """
    Loads the purpose-of-visit breakdown, annual summary, and scenario
    slicer values that originally lived inside the Power BI data model
    (FORECASTING_SARIMA.pbix). These were extracted once from the .pbix
    and saved as PBI_PURPOSE_SCENARIO_DATA.xlsx, which must sit next to
    this script (same folder as TOURIST_ARRIVALS_DATA.xlsx).
    """
    hist = pd.read_excel("PBI_PURPOSE_SCENARIO_DATA.xlsx", sheet_name="HISTORICAL_PURPOSE")
    fcst = pd.read_excel("PBI_PURPOSE_SCENARIO_DATA.xlsx", sheet_name="FORECAST")
    scenario_opts = pd.read_excel("PBI_PURPOSE_SCENARIO_DATA.xlsx", sheet_name="SCENARIO_OPTIONS")
    annual = pd.read_excel("PBI_PURPOSE_SCENARIO_DATA.xlsx", sheet_name="ANNUAL_SUMMARY")

    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fcst["DATE"] = pd.to_datetime(fcst["DATE"])
    annual["YEAR"] = pd.to_datetime(annual["YEAR"]).dt.year

    return hist, fcst, scenario_opts, annual


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
# FIT MODELS
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
    [
        "🏠 Forecasting",
        "📊 Dashboard (2026–2027)",
        "🥧 Power BI Dashboard",
        "📋 Metrics",
        "📌 Conclusion",
    ]
)

# ============================================================
# PAGE 1 — FORECASTING
# ============================================================
if page == "🏠 Forecasting":

    st.title("📊 Kenya Tourism Forecasting")

    model_choice = st.selectbox(
        "Select Model",
        ["SARIMA", "ARIMA", "Holt-Winters", "Prophet"]
    )

    month_names = list(calendar.month_name)[1:]
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
            st.error(
                "Please select a month/year after the last available data point "
                f"({last_date.strftime('%B %Y')})."
            )
        else:
            forecast = forecast_model(months_ahead, model_choice)
            prediction = forecast.iloc[-1]

            st.metric(
                f"Forecast for {selected_month_name} {selected_year}",
                f"{prediction:,.0f}"
            )

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

# ============================================================
# PAGE 2 — ENHANCED DASHBOARD
# ============================================================
elif page == "📊 Dashboard (2026–2027)":

    st.title("📈 Kenya Tourism Dashboard")
    st.caption("SARIMA model · 24-month outlook · Data from Kenya Tourism Board")

    # --- Compute 24-month SARIMA forecast ---
    forecast_steps = 24
    forecast_values = sarima_model.forecast(forecast_steps)
    future_dates = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="MS"
    )
    df_forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_values.values
    })

    # ── KPI CARDS ──────────────────────────────────────────
    peak_row   = df_forecast.loc[df_forecast["Forecast"].idxmax()]
    total_2026 = df_forecast[df_forecast["Date"].dt.year == 2026]["Forecast"].sum()
    total_2027 = df_forecast[df_forecast["Date"].dt.year == 2027]["Forecast"].sum()
    yoy_growth = ((total_2027 - total_2026) / total_2026) * 100
    last_actual = y.iloc[-1]
    latest_forecast = df_forecast["Forecast"].iloc[0]
    mom_change = ((latest_forecast - last_actual) / last_actual) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "🏔️ Peak Forecast Month",
        peak_row["Date"].strftime("%b %Y"),
        f"{peak_row['Forecast']:,.0f} arrivals"
    )
    k2.metric(
        "📅 Total 2026 Forecast",
        f"{total_2026:,.0f}",
        f"Across 12 months"
    )
    k3.metric(
        "📅 Total 2027 Forecast",
        f"{total_2027:,.0f}",
        f"{yoy_growth:+.1f}% vs 2026"
    )
    k4.metric(
        "📈 Next Month vs Last Actual",
        f"{latest_forecast:,.0f}",
        f"{mom_change:+.1f}%"
    )

    st.divider()

    # ── MAIN INTERACTIVE CHART ──────────────────────────────
    st.subheader("Historical Data vs 24-Month SARIMA Forecast")

    fig_main = go.Figure()

    # Actual data line
    fig_main.add_trace(go.Scatter(
        x=df.index,
        y=df["TOURIST ARRIVALS"],
        name="Actual Arrivals",
        line=dict(color="#003B73", width=2.5),
        hovertemplate="%{x|%b %Y}<br>Actual: <b>%{y:,.0f}</b><extra></extra>"
    ))

    # Forecast line
    fig_main.add_trace(go.Scatter(
        x=df_forecast["Date"],
        y=df_forecast["Forecast"],
        name="SARIMA Forecast",
        line=dict(color="#00A6FB", width=2.5, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>Forecast: <b>%{y:,.0f}</b><extra></extra>"
    ))

    # Shaded forecast zone
    fig_main.add_vrect(
        x0=str(df_forecast["Date"].iloc[0]),
        x1=str(df_forecast["Date"].iloc[-1]),
        fillcolor="rgba(0,166,251,0.08)",
        layer="below",
        line_width=0,
        annotation_text="Forecast Period",
        annotation_position="top left",
        annotation_font_color="#00A6FB"
    )

    fig_main.update_layout(
        xaxis_title="Date",
        yaxis_title="Tourist Arrivals",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=420,
        margin=dict(t=40, b=40)
    )

    st.plotly_chart(fig_main, use_container_width=True)

    st.divider()

    # ── TWO CHARTS SIDE BY SIDE ─────────────────────────────
    col_a, col_b = st.columns(2)

    # LEFT: Average monthly seasonality bar chart
    with col_a:
        st.subheader("📅 Average Arrivals by Month")
        st.caption("Based on historical data — shows the seasonal pattern")

        df_monthly = df.copy()
        df_monthly["Month"] = df_monthly.index.month
        monthly_avg = df_monthly.groupby("Month")["TOURIST ARRIVALS"].mean()
        month_labels = [calendar.month_abbr[m] for m in monthly_avg.index]
        peak_val = monthly_avg.max()

        bar_colors = [
            "#00A6FB" if v == peak_val else "#4FC3F7"
            for v in monthly_avg.values
        ]

        fig_seasonal = go.Figure(go.Bar(
            x=month_labels,
            y=monthly_avg.values,
            marker_color=bar_colors,
            hovertemplate="%{x}: <b>%{y:,.0f}</b> avg arrivals<extra></extra>"
        ))
        fig_seasonal.update_layout(
            yaxis_title="Avg Arrivals",
            template="plotly_white",
            showlegend=False,
            height=360,
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

    # RIGHT: 2026 vs 2027 grouped bar chart
    with col_b:
        st.subheader("📊 2026 vs 2027 Monthly Comparison")
        st.caption("Forecast arrivals — month by month side-by-side")

        df_26 = df_forecast[df_forecast["Date"].dt.year == 2026].copy()
        df_27 = df_forecast[df_forecast["Date"].dt.year == 2027].copy()

        months_26 = [calendar.month_abbr[d.month] for d in df_26["Date"]]
        months_27 = [calendar.month_abbr[d.month] for d in df_27["Date"]]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="2026",
            x=months_26,
            y=df_26["Forecast"].values,
            marker_color="#003B73",
            hovertemplate="%{x} 2026: <b>%{y:,.0f}</b><extra></extra>"
        ))
        fig_compare.add_trace(go.Bar(
            name="2027",
            x=months_27,
            y=df_27["Forecast"].values,
            marker_color="#00A6FB",
            hovertemplate="%{x} 2027: <b>%{y:,.0f}</b><extra></extra>"
        ))
        fig_compare.update_layout(
            barmode="group",
            template="plotly_white",
            yaxis_title="Forecast Arrivals",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=360,
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    st.divider()

    # ── CUMULATIVE GROWTH AREA CHART ────────────────────────
    st.subheader("📈 Cumulative Forecast Arrivals (2026–2027)")
    st.caption("Running total of projected tourist arrivals over the forecast window")

    df_forecast_cum = df_forecast.copy()
    df_forecast_cum["Cumulative"] = df_forecast_cum["Forecast"].cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=df_forecast_cum["Date"],
        y=df_forecast_cum["Cumulative"],
        fill="tozeroy",
        fillcolor="rgba(0,166,251,0.18)",
        line=dict(color="#00A6FB", width=2.5),
        hovertemplate="%{x|%b %Y}<br>Cumulative: <b>%{y:,.0f}</b><extra></extra>",
        name="Cumulative Arrivals"
    ))

    # Annotate year boundaries
    for yr in [2026, 2027]:
        yr_end = df_forecast_cum[df_forecast_cum["Date"].dt.year == yr]
        if not yr_end.empty:
            last_row = yr_end.iloc[-1]
            fig_cum.add_annotation(
                x=last_row["Date"],
                y=last_row["Cumulative"],
                text=f"End of {yr}: {last_row['Cumulative']:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#003B73",
                font=dict(color="#003B73", size=11),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#003B73",
                borderwidth=1,
                borderpad=4
            )

    fig_cum.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Arrivals",
        template="plotly_white",
        showlegend=False,
        height=360,
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    st.divider()

    # ── FORECAST TABLE + DOWNLOAD ───────────────────────────
    st.subheader("📋 Full Forecast Table")

    df_display = df_forecast.copy()
    df_display["Date"] = df_display["Date"].dt.strftime("%B %Y")
    df_display["Forecast"] = df_display["Forecast"].round(0).astype(int)
    df_display.columns = ["Month", "Forecast Arrivals"]

    st.dataframe(
        df_display.style.format({"Forecast Arrivals": "{:,}"}),
        use_container_width=True,
        height=300
    )

    csv = df_forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast as CSV",
        data=csv,
        file_name="kenya_tourism_forecast_2026_2027.csv",
        mime="text/csv"
    )

# ============================================================
# PAGE 3 — PURPOSE & SCENARIO (Power BI replica, fully live)
# ============================================================
elif page == "🥧 Power BI Dashboard":

    st.title("🥧 Power BI Dashboard")

    try:
        hist_p, fcst_p, scenario_opts, annual_p = load_pbi_purpose_data()
    except FileNotFoundError:
        st.error(
            "PBI_PURPOSE_SCENARIO_DATA.xlsx was not found next to this script. "
            "Place it in the same folder as TOURIST_ARRIVALS_DATA.xlsx and reload."
        )
        st.stop()

    # ── KPI CARDS (mirrors the four PBI cardVisuals) ────────
    actual_2025 = hist_p.loc[hist_p["Year"] == 2025, "HISTORICAL VALUES"].sum()
    forecast_2026 = fcst_p.loc[fcst_p["Year"] == 2026, "FORECAST VALUES"].sum()
    forecast_growth_pct = (forecast_2026 - actual_2025) / actual_2025 * 100

    peak_year_row = annual_p.loc[annual_p["TOURIST ARRIVALS"].idxmax()]
    peak_year = int(peak_year_row["YEAR"])

    purpose_totals = {
        "Holiday": hist_p["HOLIDAY"].sum(),
        "Business": hist_p["BUSINESS"].sum(),
        "VFR": hist_p["VFR"].sum(),
        "Others": hist_p["OTHERS"].sum(),
    }
    best_purpose = max(purpose_totals, key=purpose_totals.get)
    total_arrivals_all_time = hist_p["HISTORICAL VALUES"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🧮 Total Arrivals", f"{total_arrivals_all_time:,.0f}", "2019–2025")
    k2.metric("📈 Forecast Growth %", f"{forecast_growth_pct:+.1f}%", "2026 forecast vs 2025 actual")
    k3.metric("🏆 Peak Year", str(peak_year), f"{peak_year_row['TOURIST ARRIVALS']:,.0f} arrivals")
    k4.metric("🎯 Dominant Purpose", best_purpose, f"{purpose_totals[best_purpose]:,.0f} total")

    st.divider()

    col_a, col_b = st.columns(2)

    # ── LEFT: 100% stacked annual purpose chart ─────────────
    with col_a:
        st.subheader("📊 Purpose of Visit — Annual Stacked (2019–2025)")

        annual_purpose = hist_p.groupby("Year")[["HOLIDAY", "BUSINESS", "VFR", "OTHERS"]].sum()
        annual_purpose_pct = annual_purpose.div(annual_purpose.sum(axis=1), axis=0) * 100
        year_labels = annual_purpose_pct.index.astype(int).astype(str)

        colors = {"HOLIDAY": "#00A6FB", "BUSINESS": "#003B73", "VFR": "#4FC3F7", "OTHERS": "#FFD166"}

        fig_stack = go.Figure()
        for col in ["HOLIDAY", "BUSINESS", "VFR", "OTHERS"]:
            fig_stack.add_trace(go.Bar(
                name=col.title(),
                x=year_labels,
                y=annual_purpose_pct[col],
                marker_color=colors[col],
                hovertemplate=f"%{{x}} {col.title()}: <b>%{{y:.1f}}%</b><extra></extra>"
            ))
        fig_stack.update_layout(
            barmode="stack",
            template="plotly_white",
            yaxis_title="Share of Arrivals (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380,
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # ── RIGHT: donut share of purpose, full period ──────────
    with col_b:
        st.subheader("🥧 Purpose Share — Full Period (2019–2025)")

        fig_donut = go.Figure(go.Pie(
            labels=list(purpose_totals.keys()),
            values=list(purpose_totals.values()),
            hole=0.55,
            marker=dict(colors=["#00A6FB", "#003B73", "#4FC3F7", "#FFD166"]),
            hovertemplate="%{label}: <b>%{percent}</b><extra></extra>"
        ))
        fig_donut.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(t=20, b=40),
            showlegend=True
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # ── SCENARIO SIMULATOR (replaces the static PBI slicer) ─
    st.subheader("🎛️ Scenario Simulator — 2026 Growth Impact")

    growth_options = sorted(scenario_opts["Growth Impact"].tolist())

    growth_pct = st.slider(
        "Growth Impact (%)",
        min_value=int(min(growth_options)),
        max_value=int(max(growth_options)),
        value=0,
        step=10
    )

    base_2026 = actual_2025 * 1.05
    scenario_2026 = actual_2025 * (1 + growth_pct / 100)

    s1, s2, s3 = st.columns(3)
    s1.metric("2025 Actual", f"{actual_2025:,.0f}")
    s2.metric("2026 Base (+5%)", f"{base_2026:,.0f}")
    s3.metric(
        f"2026 Scenario ({growth_pct:+d}%)",
        f"{scenario_2026:,.0f}",
        f"{scenario_2026 - base_2026:+,.0f} vs base"
    )

    fig_scenario = go.Figure(go.Bar(
        x=["2025 Actual", "2026 Base", "2026 Scenario"],
        y=[actual_2025, base_2026, scenario_2026],
        marker_color=["#4FC3F7", "#003B73", "#00A6FB"],
        text=[f"{v:,.0f}" for v in [actual_2025, base_2026, scenario_2026]],
        textposition="outside",
        hovertemplate="%{x}: <b>%{y:,.0f}</b><extra></extra>"
    ))
    fig_scenario.update_layout(
        template="plotly_white",
        yaxis_title="Tourist Arrivals",
        height=380,
        showlegend=False,
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_scenario, use_container_width=True)

# ============================================================
# PAGE 4 — METRICS
# ============================================================
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

    # KPI cards for best model
    st.subheader("🏆 Best Model at a Glance (SARIMA)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", "11,762", "Lowest among all models")
    m2.metric("RMSE", "14,619", "Lowest among all models")
    m3.metric("MAPE", "5.57%", "Closest to actual values")

    st.divider()
    st.subheader("Full Comparison Table")
    st.dataframe(metrics, use_container_width=True)

    # Bar chart comparison
    fig_metrics = go.Figure()
    for metric, color in zip(["MAE", "RMSE"], ["#003B73", "#00A6FB"]):
        fig_metrics.add_trace(go.Bar(
            name=metric,
            x=metrics["Model"],
            y=metrics[metric],
            marker_color=color,
            hovertemplate=f"{metric}: <b>%{{y:,.0f}}</b><extra></extra>"
        ))

    fig_metrics.update_layout(
        barmode="group",
        title="MAE & RMSE by Model (lower is better)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # MAPE chart
    fig_mape = go.Figure(go.Bar(
        x=metrics["Model"],
        y=metrics["MAPE (%)"],
        marker_color=["#00A6FB" if v == metrics["MAPE (%)"].min() else "#4FC3F7"
                      for v in metrics["MAPE (%)"]],
        hovertemplate="MAPE: <b>%{y:.2f}%</b><extra></extra>"
    ))
    fig_mape.update_layout(
        title="MAPE by Model — % error (lower is better)",
        yaxis_title="MAPE (%)",
        template="plotly_white",
        showlegend=False,
        height=320
    )
    st.plotly_chart(fig_mape, use_container_width=True)

# ============================================================
# PAGE 5 — CONCLUSION
# ============================================================
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
