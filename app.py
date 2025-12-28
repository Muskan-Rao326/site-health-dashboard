import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Site Health Dashboard", layout="wide")

st.title("📊 Site Health Dashboard")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload site health CSV (single_site_health_scored.csv)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    # =========================
    # KPIs
    # =========================
    latest = df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "💰 Revenue",
        f"{latest['Revenue']:,.0f}",
        delta=f"{latest['Revenue'] - df.iloc[-2]['Revenue']:,.0f}"
    )

    col2.metric(
        "👥 Sessions",
        f"{latest['Sessions']:,.0f}",
        delta=f"{latest['Sessions'] - df.iloc[-2]['Sessions']:,.0f}"
    )

    col3.metric(
        "❤️ Health Score",
        f"{latest['HealthScore']:.1f}"
    )

    st.divider()

    # =========================
    # Charts
    # =========================
    colA, colB = st.columns(2)

    with colA:
        fig_rev = px.line(
            df, x="Date", y="Revenue",
            title="Revenue Trend",
            markers=True
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with colB:
        fig_sess = px.line(
            df, x="Date", y="Sessions",
            title="Sessions Trend",
            markers=True
        )
        st.plotly_chart(fig_sess, use_container_width=True)

    # =========================
    # Health Score Gauge
    # =========================
    score = latest["HealthScore"]

    color = "green" if score >= 75 else "orange" if score >= 50 else "red"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Overall Site Health"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 75], "color": "#fff2cc"},
                {"range": [75, 100], "color": "#ccffcc"},
            ]
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    # =========================
    # Raw Data (Optional)
    # =========================
    with st.expander("📄 View Raw Data"):
        st.dataframe(df)

else:
    st.info("⬆️ Upload your `single_site_health_scored.csv` file to view the dashboard.")
