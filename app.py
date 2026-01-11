import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from scipy.stats import norm

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Revenue Intelligence (PowerBI Style)", layout="wide")

# =============================
# STYLE (PowerBI-ish)
# =============================
st.markdown("""
<style>
    .report-title { font-size: 26px; font-weight: 800; margin-bottom: 4px; }
    .report-sub { font-size: 13px; opacity: 0.8; margin-bottom: 14px; }
    .card { border-radius: 14px; padding: 14px; color: white; }
    .kpi-label { font-size: 13px; opacity: 0.95; }
    .kpi-val { font-size: 26px; font-weight: 800; margin-top: 2px; }
    .kpi-meta { font-size: 12px; opacity: 0.95; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# =============================
# HELPERS
# =============================
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

def pct_change(today, prev):
    if prev == 0:
        return 0.0 if today == 0 else 999.0
    return (today - prev) / prev * 100

def robust_z(val, median, mad):
    denom = 1.4826 * mad
    if denom == 0 or np.isnan(denom):
        return 0.0
    return (val - median) / denom

def confidence_from_z(z):
    p = 2 * (1 - norm.cdf(abs(z)))
    return (1 - p) * 100

def color_from_z(z):
    if z >= -1:
        return "#2ECC71"
    elif z >= -2:
        return "#F1C40F"
    return "#E74C3C"

def kpi_card(title, value, delta_pct, z, conf, suffix=""):
    bg = color_from_z(z)
    return f"""
    <div class="card" style="background:{bg}">
        <div class="kpi-label">{title}</div>
        <div class="kpi-val">{value:,.2f}{suffix}</div>
        <div class="kpi-meta">Δ vs Yesterday: {delta_pct:+.2f}%</div>
        <div class="kpi-meta">Robust Z: {z:.2f} • Confidence: {conf:.1f}%</div>
    </div>
    """

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

# =============================
# LOAD
# =============================
st.markdown('<div class="report-title">📊 Revenue Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="report-sub">PowerBI-style daily reporting (GA4 + GAM) — single-site mode</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_data(uploaded)

# Single site
if "site_name" in df_raw.columns:
    site = sorted(df_raw["site_name"].dropna().unique().tolist())[0]
    df_raw = df_raw[df_raw["site_name"] == site].copy()
    st.caption(f"Using site: **{site}**")

# Required additive
base_cols = ["revenue","ad_requests","impressions","clicks","sessions","users","pageviews"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived (truth)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)
daily["requests_per_pageview"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["impressions_per_session"] = daily.apply(lambda r: safe_div(r["impressions"], r["sessions"], 1), axis=1)

# =============================
# SIDEBAR SLICERS (PowerBI feel)
# =============================
st.sidebar.header("Slicers")

selected_date = st.sidebar.date_input(
    "Date",
    value=daily["date"].max().date(),
    min_value=daily["date"].min().date(),
    max_value=daily["date"].max().date()
)

baseline_days = st.sidebar.slider("Baseline Window (days)", 7, 30, 7)
compare_mode = st.sidebar.selectbox("Compare To", ["Yesterday", "7-day median", "Baseline median"], index=0)

today = pd.to_datetime(selected_date).normalize()
yesterday = today - timedelta(days=1)
baseline_start = today - timedelta(days=baseline_days)

today_row = daily[daily["date"] == today]
if today_row.empty:
    st.error("No data for selected date.")
    st.stop()

t = today_row.iloc[0].to_dict()
y_row = daily[daily["date"] == yesterday]
y = y_row.iloc[0].to_dict() if not y_row.empty else None

baseline_df = daily[(daily["date"] < today) & (daily["date"] >= baseline_start)]
display_df = daily[(daily["date"] >= baseline_start - timedelta(days=1)) & (daily["date"] <= today)]

# compare reference
if compare_mode == "Yesterday" and y:
    ref = y
elif compare_mode == "7-day median":
    ref_df = daily[(daily["date"] < today)].tail(7)
    ref = ref_df.median(numeric_only=True).to_dict() if not ref_df.empty else {}
else:
    ref = baseline_df.median(numeric_only=True).to_dict() if not baseline_df.empty else {}

# =============================
# KPI ENGINE (Median + MAD baseline)
# =============================
kpi_keys = [
    ("revenue","Revenue",""),
    ("impressions","Impressions",""),
    ("ad_requests","Ad Requests",""),
    ("fill_rate","Fill Rate","%"),
    ("ecpm","eCPM",""),
    ("rpm","RPM",""),
    ("requests_per_pageview","Req/Pageview",""),
]

kpi = {}
for key, label, suffix in kpi_keys:
    s = baseline_df[key].dropna().astype(float)
    med = float(s.median()) if len(s) else 0.0
    mad = float(np.median(np.abs(s - med))) if len(s) else 0.0

    val_today = float(t.get(key, 0.0))
    val_ref = float(ref.get(key, 0.0)) if ref else 0.0

    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)
    delta = pct_change(val_today, val_ref) if val_ref is not None else 0.0

    kpi[key] = dict(label=label, suffix=suffix, today=val_today, ref=val_ref, med=med, mad=mad, z=z, conf=conf, delta=delta)

# =============================
# FORECAST / EXPECTED REVENUE (PowerBI "Target" style)
# =============================
expected_revenue = float(baseline_df["revenue"].median()) if not baseline_df.empty else 0.0
actual_revenue = float(t["revenue"])
lost = expected_revenue - actual_revenue
lost_pct = safe_div(lost, expected_revenue, 100) if expected_revenue else 0.0

# =============================
# KPI ROW (PowerBI cards)
# =============================
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi_card("Revenue", kpi["revenue"]["today"], kpi["revenue"]["delta"], kpi["revenue"]["z"], kpi["revenue"]["conf"]), unsafe_allow_html=True)
c2.markdown(kpi_card("Impressions", kpi["impressions"]["today"], kpi["impressions"]["delta"], kpi["impressions"]["z"], kpi["impressions"]["conf"]), unsafe_allow_html=True)
c3.markdown(kpi_card("Ad Requests", kpi["ad_requests"]["today"], kpi["ad_requests"]["delta"], kpi["ad_requests"]["z"], kpi["ad_requests"]["conf"]), unsafe_allow_html=True)
c4.markdown(kpi_card("Fill Rate", kpi["fill_rate"]["today"], kpi["fill_rate"]["delta"], kpi["fill_rate"]["z"], kpi["fill_rate"]["conf"], "%"), unsafe_allow_html=True)
c5.markdown(kpi_card("eCPM", kpi["ecpm"]["today"], kpi["ecpm"]["delta"], kpi["ecpm"]["z"], kpi["ecpm"]["conf"]), unsafe_allow_html=True)

# =============================
# MAIN ROW: Decomposition + Expected Revenue
# =============================
left, right = st.columns([2, 1])

with left:
    st.subheader("📌 Revenue Change Breakdown (PowerBI Waterfall style)")

    if y:
        # Revenue decomposition approximation:
        # revenue = impressions * ecpm / 1000
        imp_y = float(y["impressions"])
        ecpm_y = float(safe_div(y["revenue"], y["impressions"], 1000))
        imp_t = float(t["impressions"])
        ecpm_t = float(safe_div(t["revenue"], t["impressions"], 1000))

        rev_y = float(y["revenue"])
        rev_t = float(t["revenue"])

        # Contribution
        contrib_imp = (imp_t - imp_y) * (ecpm_y / 1000)
        contrib_ecpm = imp_t * ((ecpm_t - ecpm_y) / 1000)
        residual = (rev_t - rev_y) - (contrib_imp + contrib_ecpm)

        wf = pd.DataFrame({
            "Step": ["Yesterday Revenue", "Impressions Effect", "eCPM Effect", "Other/Residual", "Today Revenue"],
            "Value": [rev_y, contrib_imp, contrib_ecpm, residual, rev_t],
            "Type": ["base", "change", "change", "change", "total"]
        })

        # Build a waterfall-like bar using cumulative logic
        wf["Cumulative"] = wf["Value"].cumsum()
        wf["Start"] = wf["Cumulative"] - wf["Value"]

        chart = alt.Chart(wf).mark_bar().encode(
            x=alt.X("Step:N", sort=None),
            y=alt.Y("Start:Q", title="Revenue"),
            y2="Cumulative:Q",
            tooltip=["Step", alt.Tooltip("Value:Q", format=",.2f")]
        ).properties(height=280)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need yesterday data to show decomposition waterfall.")

with right:
    st.subheader("🎯 Expected Revenue (Target)")
    st.metric("Expected (Baseline Median)", f"{expected_revenue:,.2f}")
    st.metric("Actual (Today)", f"{actual_revenue:,.2f}", delta=f"{pct_change(actual_revenue, expected_revenue):+.2f}%" if expected_revenue else None)
    st.metric("Lost vs Expected", f"{max(lost,0):,.2f}", delta=f"{lost_pct:+.2f}%" if expected_revenue else None)
    st.caption(f"Baseline window: last **{baseline_days}** days (median).")

# =============================
# TOP ISSUES TABLE (PowerBI matrix feel)
# =============================
st.subheader("🧯 Top Issues (Anomaly + Drop ranked)")

all_metrics = [
    ("revenue","Revenue"),
    ("sessions","Sessions"),
    ("pageviews","Pageviews"),
    ("ad_requests","Ad Requests"),
    ("requests_per_pageview","Req/Pageview"),
    ("fill_rate","Fill Rate"),
    ("impressions","Impressions"),
    ("ecpm","eCPM"),
    ("rpm","RPM"),
    ("ctr","CTR"),
    ("impressions_per_session","Impressions/Session"),
]

rows = []
for key, label in all_metrics:
    s = baseline_df[key].dropna().astype(float)
    med = float(s.median()) if len(s) else 0.0
    mad = float(np.median(np.abs(s - med))) if len(s) else 0.0
    val_today = float(t.get(key, 0.0))
    val_ref = float(ref.get(key, 0.0)) if ref else 0.0
    z = robust_z(val_today, med, mad)
    conf = confidence_from_z(z)
    delta = pct_change(val_today, val_ref) if val_ref else 0.0

    # rank score
    score = (abs(z) * 2 if z < 0 else 0) + (abs(delta) / 10 if delta < 0 else 0)

    rows.append({
        "Metric": label,
        "Today": val_today,
        "Compare": val_ref,
        "Δ %": delta,
        "Baseline Median": med,
        "Robust Z": z,
        "Confidence %": conf,
        "Score": score
    })

issues = pd.DataFrame(rows).sort_values("Score", ascending=False).head(8).drop(columns=["Score"])
st.dataframe(issues, use_container_width=True)

# =============================
# TABS (PowerBI pages)
# =============================
tab1, tab2, tab3, tab4 = st.tabs(["Overview Trends", "Traffic", "Delivery", "Value"])

def line_chart(cols, title):
    d = display_df[["date"] + cols].copy()
    melted = d.melt(id_vars="date", value_vars=cols, var_name="metric", value_name="value")
    return alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("date:T"),
        y=alt.Y("value:Q"),
        color="metric:N",
        tooltip=["date:T","metric:N","value:Q"]
    ).properties(title=title, height=280)

with tab1:
    st.altair_chart(line_chart(["revenue","ecpm"], "Revenue vs eCPM"), use_container_width=True)
    st.altair_chart(line_chart(["rpm","fill_rate"], "RPM vs Fill Rate"), use_container_width=True)

with tab2:
    st.altair_chart(line_chart(["sessions","pageviews"], "Sessions vs Pageviews"), use_container_width=True)

with tab3:
    st.altair_chart(line_chart(["ad_requests","requests_per_pageview"], "Requests & Requests/Pageview (Ad Loading)"), use_container_width=True)
    st.altair_chart(line_chart(["fill_rate","impressions"], "Fill Rate → Impressions (Demand/Delivery)"), use_container_width=True)

with tab4:
    st.altair_chart(line_chart(["ecpm","ctr"], "eCPM vs CTR (Value Signals)"), use_container_width=True)

st.caption("PowerBI-style layout: slicers → KPI cards → decomposition → issues matrix → drill tabs.")
