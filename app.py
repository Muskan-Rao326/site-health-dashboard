# app.py
# Publisher Performance Dashboard (GA4 + GAM)
# Clean UI + Full pipeline detection (Users → Sessions → Pageviews → Requests → Impressions → Revenue)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Publisher Performance Dashboard", layout="wide")

# =========================
# STYLE
# =========================
st.markdown(
    """
<style>
.stApp { background: #f4f7fb; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Header */
.header{
  background: linear-gradient(90deg, #183a63, #244f7c);
  border-radius: 14px;
  padding: 18px 22px;
  color: #fff;
  box-shadow: 0 10px 22px rgba(15,23,42,0.20);
}
.header-title{ font-size: 28px; font-weight: 900; line-height: 1.15; }
.header-sub{ margin-top: 4px; font-size: 13px; opacity: 0.85; font-weight: 650; }

/* Panels */
.panel{
  background: #ffffff;
  border: 1px solid #e7eef7;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
}
.panel-title{
  font-size: 16px;
  font-weight: 900;
  color: #0f172a;
  margin-bottom: 10px;
}

/* KPI cards */
.kpi-card{
  background:#ffffff;
  border: 1px solid #e7eef7;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
  height: 116px;
  overflow:hidden;
}
.kpi-label{ font-size: 14px; font-weight: 850; color:#0f172a; opacity: 0.85; }
.kpi-value{ margin-top: 6px; font-size: 38px; font-weight: 950; color:#0b1220; line-height: 1.05; }
.kpi-delta{
  margin-top: 10px;
  display:inline-block;
  padding: 6px 12px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 900;
  color: #fff;
}
.badge-red{ background:#d9534f; }
.badge-green{ background:#16a34a; }
.badge-gray{ background:#64748b; }

/* Mini KPI */
.mini-grid{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.mini-card{
  background:#ffffff;
  border: 1px solid #e7eef7;
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
  height: 96px;
}
.mini-label{ font-size: 13px; font-weight: 850; color:#0f172a; opacity: 0.85;}
.mini-row{ margin-top: 6px; display:flex; align-items: baseline; justify-content:space-between; gap: 8px; }
.mini-value{ font-size: 26px; font-weight: 950; color:#0b1220; line-height: 1.0; }
.mini-delta{ font-size: 13px; font-weight: 950; }
.mini-bar{ height:6px; border-radius: 10px; margin-top: 10px; background:#e2e8f0; overflow:hidden; }
.mini-bar > div{ height:100%; border-radius:10px; }

/* Diagnosis */
.diag-strip{
  background: #fde9dd;
  border-left: 6px solid #ef4444;
  padding: 10px 12px;
  border-radius: 12px;
  font-weight: 950;
  color:#0b1220;
}
.chips{ margin-top: 12px; display:flex; gap:10px; flex-wrap:wrap; }
.chip{
  padding: 8px 12px;
  border-radius: 12px;
  font-weight: 900;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background:#f1f5f9;
  color:#0b1220;
}
.chip-green{ background:#dcfce7; border-color:#22c55e; }
.chip-red{ background:#fee2e2; border-color:#ef4444; }
.chip-amber{ background:#ffedd5; border-color:#f59e0b; }
.small-note{ font-size: 12px; color:#475569; font-weight: 650; margin-top: 8px; }
hr.soft { border:none; height:1px; background:#e7eef7; margin: 12px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

def pct_change(new, old):
    if old == 0:
        return 0.0 if new == 0 else -100.0
    return (new - old) / old * 100.0

def fmt_compact(x):
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{sign}{x/1_000:.0f}K"
    return f"{sign}{x:.0f}"

def badge_class(delta_pct):
    if np.isnan(delta_pct):
        return "badge-gray"
    return "badge-red" if delta_pct < 0 else "badge-green"

def kpi_card(label, value_str, delta_pct):
    b = badge_class(delta_pct)
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value_str}</div>
      <span class="kpi-delta {b}">{delta_pct:+.0f}%</span>
    </div>
    """

def mini_card(label, value_str, delta_pct):
    mag = min(abs(delta_pct) / 40.0, 1.0)
    color = "#ef4444" if delta_pct < 0 else "#16a34a"
    sign = "▼" if delta_pct < 0 else "▲"
    if abs(delta_pct) < 1:
        txt = "= 0%"
        color_txt = "#0b1220"
        bar_color = "#94a3b8"
        mag = 0.35
    else:
        txt = f"{sign}{abs(delta_pct):.0f}%"
        color_txt = color
        bar_color = color

    return f"""
    <div class="mini-card">
      <div class="mini-label">{label}</div>
      <div class="mini-row">
        <div class="mini-value">{value_str}</div>
        <div class="mini-delta" style="color:{color_txt};">{txt}</div>
      </div>
      <div class="mini-bar"><div style="width:{mag*100:.0f}%; background:{bar_color};"></div></div>
    </div>
    """

def panel(title, inner_html):
    return f"""
    <div class="panel">
      <div class="panel-title">{title}</div>
      {inner_html}
    </div>
    """

# =========================
# LOAD CSV
# =========================
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

uploaded = st.file_uploader("Upload merged GA4 + GAM CSV", type=["csv"])
if not uploaded:
    st.stop()

df_raw = load_csv(uploaded)

# Site selector (optional)
site = None
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    site = st.sidebar.selectbox("Site", sites, index=0)
    df_raw = df_raw[df_raw["site_name"] == site].copy()

# Required totals
base_cols = ["users", "sessions", "pageviews", "ad_requests", "impressions", "clicks", "revenue"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

# Daily totals
daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived ratios (derived from totals ONLY)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["requests_per_page"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["pageviews_per_session"] = daily.apply(lambda r: safe_div(r["pageviews"], r["sessions"], 1), axis=1)
daily["sessions_per_user"] = daily.apply(lambda r: safe_div(r["sessions"], r["users"], 1), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["pageviews"], 1000), axis=1)

# =========================
# CONTROLS
# =========================
st.sidebar.header("Controls")
selected_date = st.sidebar.date_input(
    "Date",
    value=daily["date"].max().date(),
    min_value=daily["date"].min().date(),
    max_value=daily["date"].max().date(),
)
baseline_days = st.sidebar.slider("Baseline window (days)", 7, 30, 7)

today = pd.to_datetime(selected_date).normalize()
baseline_start = today - timedelta(days=baseline_days)

today_df = daily[daily["date"] == today]
if today_df.empty:
    st.error("No data for selected date.")
    st.stop()

t = today_df.iloc[0].to_dict()
baseline_df = daily[(daily["date"] < today) & (daily["date"] >= baseline_start)]

# Expected baseline = median of totals
if baseline_df.empty:
    exp = {c: 0.0 for c in base_cols}
else:
    exp = baseline_df[base_cols].median(numeric_only=True).to_dict()

# expected ratios derived from expected totals
exp["ecpm"] = safe_div(exp["revenue"], exp["impressions"], 1000)
exp["fill_rate"] = safe_div(exp["impressions"], exp["ad_requests"], 100)
exp["requests_per_page"] = safe_div(exp["ad_requests"], exp["pageviews"], 1)
exp["pageviews_per_session"] = safe_div(exp["pageviews"], exp["sessions"], 1)
exp["sessions_per_user"] = safe_div(exp["sessions"], exp["users"], 1)
exp["ctr"] = safe_div(exp["clicks"], exp["impressions"], 100)
exp["rpm"] = safe_div(exp["revenue"], exp["pageviews"], 1000)

# deltas vs expected
metrics_for_deltas = [
    "revenue","ecpm","fill_rate","requests_per_page",
    "users","sessions","pageviews","ad_requests","impressions",
    "pageviews_per_session","sessions_per_user","rpm","ctr"
]
d = {k: pct_change(t.get(k,0.0), exp.get(k,0.0)) for k in metrics_for_deltas}

# =========================
# HEADER
# =========================
domain = site if site else "abc.com"
st.markdown(
    f"""
    <div class="header">
      <div class="header-title">Publisher Performance Dashboard <span style="font-weight:650;">for {domain}</span></div>
      <div class="header-sub">Expected = baseline median of last {baseline_days} days (excluding selected date) • GA4 + GAM combined</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# =========================
# TOP KPIs (clean & consistent)
# =========================
c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="large")
with c1:
    st.markdown(kpi_card("Revenue", f"${t['revenue']:,.0f}", d["revenue"]), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("eCPM", f"${t['ecpm']:.2f}", d["ecpm"]), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("Fill Rate", f"{t['fill_rate']:.0f}%", d["fill_rate"]), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("Requests / Page", f"{t['requests_per_page']:.2f}", d["requests_per_page"]), unsafe_allow_html=True)

st.write("")

# =========================
# PIPELINE (GA → GAM) + BOTTLENECK DETECTION
# =========================
def pipeline_table(t, exp):
    """
    Full funnel with ratios:
      Users -> Sessions -> Pageviews -> Ad Requests -> Impressions -> Revenue
    And the key "rate steps" in between:
      Sessions/User, Pageviews/Session, Requests/Pageview, Fill (Imp/Req), eCPM (Rev/Imp*1000)
    Bottleneck = the step with largest negative % change vs expected.
    """
    rows = []

    # absolute stages
    rows += [
        ("Users", t["users"], exp["users"], "GA"),
        ("Sessions", t["sessions"], exp["sessions"], "GA"),
        ("Pageviews", t["pageviews"], exp["pageviews"], "GA"),
        ("Ad Requests", t["ad_requests"], exp["ad_requests"], "GAM"),
        ("Impressions", t["impressions"], exp["impressions"], "GAM"),
        ("Revenue", t["revenue"], exp["revenue"], "GAM"),
    ]

    # rate steps (these are the “where it broke” levers)
    steps = [
        ("Sessions / User", safe_div(t["sessions"], t["users"], 1), safe_div(exp["sessions"], exp["users"], 1), "Engagement"),
        ("Pageviews / Session", safe_div(t["pageviews"], t["sessions"], 1), safe_div(exp["pageviews"], exp["sessions"], 1), "Engagement"),
        ("Requests / Pageview", safe_div(t["ad_requests"], t["pageviews"], 1), safe_div(exp["ad_requests"], exp["pageviews"], 1), "Setup"),
        ("Fill Rate (Imp/Req)", safe_div(t["impressions"], t["ad_requests"], 100), safe_div(exp["impressions"], exp["ad_requests"], 100), "Demand"),
        ("eCPM (Rev/Imp)", safe_div(t["revenue"], t["impressions"], 1000), safe_div(exp["revenue"], exp["impressions"], 1000), "Pricing"),
    ]

    df_abs = pd.DataFrame(rows, columns=["Stage", "Today", "Expected", "Source"])
    df_abs["Delta %"] = df_abs.apply(lambda r: pct_change(r["Today"], r["Expected"]), axis=1)

    df_steps = pd.DataFrame(steps, columns=["Stage", "Today", "Expected", "Source"])
    df_steps["Delta %"] = df_steps.apply(lambda r: pct_change(r["Today"], r["Expected"]), axis=1)

    # Bottleneck detection on step rates (this is usually the most actionable)
    bottleneck_row = df_steps.sort_values("Delta %").iloc[0] if len(df_steps) else None
    return df_abs, df_steps, bottleneck_row

df_abs, df_steps, bottleneck = pipeline_table(t, exp)

# Explain revenue gap using multiplicative factors:
# Revenue ratio approx = (Users ratio) * (Sessions/User ratio) * (PV/Session ratio) * (Req/PV ratio) * (Fill ratio) * (eCPM ratio)
def revenue_factor_breakdown(t, exp):
    def ratio(a, b): return (a / b) if b and b != 0 else 0.0
    factors = [
        ("Users", ratio(t["users"], exp["users"])),
        ("Sessions/User", ratio(safe_div(t["sessions"], t["users"], 1), safe_div(exp["sessions"], exp["users"], 1))),
        ("PV/Session", ratio(safe_div(t["pageviews"], t["sessions"], 1), safe_div(exp["pageviews"], exp["sessions"], 1))),
        ("Req/PV", ratio(safe_div(t["ad_requests"], t["pageviews"], 1), safe_div(exp["ad_requests"], exp["pageviews"], 1))),
        ("Fill", ratio(safe_div(t["impressions"], t["ad_requests"], 1), safe_div(exp["impressions"], exp["ad_requests"], 1))),
        ("eCPM", ratio(safe_div(t["revenue"], t["impressions"], 1), safe_div(exp["revenue"], exp["impressions"], 1))),
    ]
    # compute contribution score (log space for fair additive contributions)
    # negative contributions = drivers of decline
    contrib = []
    for name, r in factors:
        r = max(r, 1e-9)
        contrib.append((name, np.log(r)))
    dfc = pd.DataFrame(contrib, columns=["Driver", "LogContribution"])
    dfc["Impact Share %"] = (dfc["LogContribution"].abs() / dfc["LogContribution"].abs().sum() * 100.0).replace([np.inf, -np.inf], 0).fillna(0)
    dfc["Direction"] = np.where(dfc["LogContribution"] < 0, "Down", "Up")
    return dfc.sort_values("LogContribution")

df_contrib = revenue_factor_breakdown(t, exp)

# Diagnosis label based on worst step
def diagnosis_from_bottleneck(b):
    if b is None:
        return "Mixed Movement", [("Check Data", "amber")]
    step = str(b["Stage"])
    delta = float(b["Delta %"])
    chips = []
    if "Req" in step:
        return "Setup Problem", [("Requests/Pageview Down", "red"), ("Tag/Slot/CMP risk", "amber")]
    if "Fill" in step:
        return "Demand / Fill Problem", [("Fill Down", "red"), ("Floors/Blocks/Buyer loss", "amber")]
    if "eCPM" in step:
        return "Pricing / eCPM Problem", [("eCPM Down", "red"), ("Geo/Device mix", "amber")]
    if "PV/Session" in step or "Sessions / User" in step:
        return "Engagement Problem", [("PV/Session Down", "red"), ("Content/UX shift", "amber")]
    # fallback
    if delta < 0:
        return "Pipeline Degradation", [("Multiple Steps Down", "amber")]
    return "Normal", [("Pipeline Healthy", "green")]

issue_title, chips = diagnosis_from_bottleneck(bottleneck)

# =========================
# MIDDLE ROW: Diagnosis + Key Metrics Overview
# =========================
left, right = st.columns([1.2, 1.6], gap="large")

with left:
    chips_html = ""
    for label, typ in chips:
        cls = "chip"
        if typ == "red":
            cls = "chip chip-red"
        elif typ == "green":
            cls = "chip chip-green"
        elif typ == "amber":
            cls = "chip chip-amber"
        chips_html += f'<div class="{cls}">{label}</div>'

    bn_txt = ""
    if bottleneck is not None:
        bn_txt = f"<div class='small-note'><b>Biggest leak:</b> {bottleneck['Stage']} ({float(bottleneck['Delta %']):+.1f}%)</div>"

    inner = f"""
      <div class="diag-strip">Issue Detected: <span style="font-weight:950;">{issue_title}</span></div>
      <div class="chips">{chips_html}</div>
      {bn_txt}
      <hr class="soft"/>
      <div class="small-note"><b>How to read this:</b> Users→Sessions→Pageviews is GA. Requests→Impressions→Revenue is GAM. The “step rates” show exactly where the leak is.</div>
    """
    st.markdown(panel("Diagnosis (Pipeline-based)", inner), unsafe_allow_html=True)

with right:
    inner = f"""
    <div class="mini-grid">
      {mini_card("Users", fmt_compact(t["users"]), d["users"])}
      {mini_card("Pageviews", fmt_compact(t["pageviews"]), d["pageviews"])}
      {mini_card("Ad Requests", fmt_compact(t["ad_requests"]), d["ad_requests"])}
      {mini_card("Impressions", fmt_compact(t["impressions"]), d["impressions"])}
    </div>
    <div class="small-note">Deltas above are vs <b>Expected baseline median</b>.</div>
    """
    st.markdown(panel("Key Metrics Overview", inner), unsafe_allow_html=True)

st.write("")

# =========================
# PIPELINE TABLE PANEL
# =========================
p1, p2 = st.columns([1.2, 1], gap="large")

with p1:
    # format display tables
    df_abs_disp = df_abs.copy()
    df_abs_disp["Today"] = df_abs_disp["Today"].map(lambda x: f"{x:,.0f}")
    df_abs_disp["Expected"] = df_abs_disp["Expected"].map(lambda x: f"{x:,.0f}")
    df_abs_disp["Delta %"] = df_abs["Delta %"].map(lambda x: f"{x:+.1f}%")

    df_steps_disp = df_steps.copy()
    df_steps_disp["Today"] = df_steps_disp["Today"].map(lambda x: f"{x:,.2f}" if "Rate" not in df_steps_disp["Stage"].iloc[0] else f"{x:,.2f}")
    # better per-step formatting
    def fmt_step(stage, val):
        if "Fill" in stage:
            return f"{val:,.1f}%"
        if "eCPM" in stage:
            return f"${val:,.2f}"
        return f"{val:,.2f}"

    df_steps_disp["Today"] = [fmt_step(s, v) for s, v in zip(df_steps["Stage"], df_steps["Today"])]
    df_steps_disp["Expected"] = [fmt_step(s, v) for s, v in zip(df_steps["Stage"], df_steps["Expected"])]
    df_steps_disp["Delta %"] = df_steps["Delta %"].map(lambda x: f"{x:+.1f}%")

    inner = """
    <div class="small-note" style="margin-bottom:10px;">
      <b>Absolute stages:</b> show volume changes.<br/>
      <b>Step rates:</b> show where the pipeline leaked (these are the actionable levers).
    </div>
    """
    st.markdown(panel("Pipeline (GA → GAM)", inner), unsafe_allow_html=True)
    st.dataframe(df_abs_disp, use_container_width=True, hide_index=True)
    st.dataframe(df_steps_disp, use_container_width=True, hide_index=True)

with p2:
    # Contribution panel (what explains revenue change)
    # show top negative contributions only
    dfc = df_contrib.copy()
    dfc["LogContribution"] = dfc["LogContribution"].round(3)
    dfc["Impact Share %"] = dfc["Impact Share %"].round(1)

    inner = """
    <div class="small-note">
      This splits the revenue change into pipeline drivers using a multiplicative model.<br/>
      <b>Down</b> drivers = likely causes of the drop.
    </div>
    """
    st.markdown(panel("Revenue Change Drivers", inner), unsafe_allow_html=True)
    st.dataframe(dfc[["Driver","Direction","Impact Share %"]], use_container_width=True, hide_index=True)

st.write("")

# =========================
# CLEAN TREND CHART (INDEXED) — no distorted dual scales
# =========================
st.markdown(panel("Trend (Indexed to Expected = 100)", ""), unsafe_allow_html=True)

# last 14 days view
trend = daily[daily["date"] <= today].tail(14).copy()
# build index vs expected (baseline median)
def idx(val, base): return (val / base * 100.0) if base and base != 0 else 0.0

trend_long = []
for _, r in trend.iterrows():
    trend_long += [
        {"date": r["date"], "Metric": "Revenue", "Index": idx(r["revenue"], exp["revenue"])},
        {"date": r["date"], "Metric": "Requests/Page", "Index": idx(r["requests_per_page"], exp["requests_per_page"])},
        {"date": r["date"], "Metric": "Fill Rate", "Index": idx(r["fill_rate"], exp["fill_rate"])},
        {"date": r["date"], "Metric": "eCPM", "Index": idx(r["ecpm"], exp["ecpm"])},
        {"date": r["date"], "Metric": "Pageviews", "Index": idx(r["pageviews"], exp["pageviews"])},
    ]

trend_long = pd.DataFrame(trend_long)

chart = (
    alt.Chart(trend_long)
    .mark_line(point=True, strokeWidth=3)
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("Index:Q", title=None),
        color=alt.Color("Metric:N", legend=alt.Legend(orient="top")),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Index:Q", title="Index (Expected=100)", format=",.0f"),
        ],
    )
    .properties(height=320)
    .configure_view(stroke=None)
    .configure_axis(labelColor="#334155", gridColor="#e5e7eb", tickColor="#cbd5e1")
)

st.altair_chart(chart, use_container_width=True)
st.caption("This avoids mixed Y-scales. Index 100 = expected baseline median. Below 100 = underperforming vs expected.")

st.write("")

# =========================
# SANITY CHECKS
# =========================
checks = []
if t["sessions"] > 0 and t["pageviews"] == 0:
    checks.append("Sessions > 0 but Pageviews = 0 → GA4 pull may be incomplete.")
if t["pageviews"] > 0 and t["ad_requests"] == 0:
    checks.append("Pageviews > 0 but Ad Requests = 0 → tag/slot/CMP issue OR GAM data missing.")
if t["ad_requests"] > 0 and t["impressions"] == 0:
    checks.append("Ad Requests > 0 but Impressions = 0 → fill collapsed OR GAM reporting lag.")
if t["impressions"] > 0 and t["revenue"] == 0:
    checks.append("Impressions > 0 but Revenue = 0 → revenue reporting lag OR merge mismatch.")
if (not baseline_df.empty) and float(exp["revenue"]) == 0:
    checks.append("Expected revenue baseline is 0 but baseline rows exist → too many zeros in baseline window.")

if checks:
    st.markdown(panel("Sanity Checks", "<br/>".join([f"• {c}" for c in checks])), unsafe_allow_html=True)
else:
    st.markdown(panel("Sanity Checks", "<span class='small-note'>No obvious integrity red flags for selected day.</span>"), unsafe_allow_html=True)
