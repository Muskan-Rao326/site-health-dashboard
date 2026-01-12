# app.py
# Publisher Performance Dashboard (GA4 + GAM) — matches the layout in your image
# ✅ Correct math:
#    - Only additive totals are SUMmed (users, sessions, pageviews, ad_requests, impressions, clicks, revenue)
#    - Ratios are ALWAYS derived from totals (never summed): eCPM, Fill Rate, Requests/Page, CTR
# ✅ One story reference everywhere: Expected = baseline median (last N days, excluding selected day)
# ✅ Diagnosis logic uses the full leakage pipeline:
#    Users/Sessions/Pageviews (GA4) → Requests/Page (bridge) → Fill (GAM) → Impressions → eCPM → Revenue

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Publisher Performance Dashboard", layout="wide")

# ----------------------------
# STYLE (minimal + stable)
# ----------------------------
st.markdown(
    """
<style>
/* page bg */
.stApp { background: #f3f6fb; }

/* top header bar */
.header-bar{
  background: linear-gradient(90deg, #1e3a5f, #2b4f77);
  padding: 18px 22px;
  border-radius: 10px;
  color: white;
  font-weight: 800;
  font-size: 28px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

/* card shell */
.card{
  background:#ffffff;
  border: 1px solid #e7eef7;
  border-radius: 10px;
  padding: 16px 18px;
  box-shadow: 0 4px 10px rgba(15,23,42,0.06);
}

/* KPI title/value */
.kpi-title{ font-size: 16px; font-weight: 700; color:#22324a; }
.kpi-value{ font-size: 40px; font-weight: 900; color:#111827; line-height: 1.05; margin-top: 6px; }

/* delta badge (red like your image) */
.delta-badge{
  display:inline-block;
  background:#d9534f;
  color:white;
  font-weight:800;
  border-radius: 4px;
  padding: 6px 12px;
  margin-top: 10px;
  font-size: 14px;
}

/* mini KPI */
.mini-title{ font-size: 14px; font-weight:700; color:#22324a; }
.mini-value{ font-size: 28px; font-weight:900; color:#111827; line-height:1.0; margin-top: 6px; }
.mini-delta{ font-size: 14px; font-weight:800; margin-left: 6px; }
.bar-green{ height:6px; background:#22c55e; border-radius: 10px; margin-top:10px; }
.bar-red{ height:6px; background:#ef4444; border-radius: 10px; margin-top:10px; }
.bar-gray{ height:6px; background:#cbd5e1; border-radius: 10px; margin-top:10px; }

/* diagnosis */
.diag-title{ font-size: 18px; font-weight: 800; color:#22324a; margin-bottom: 10px; }
.diag-strip{
  background: #f6d9c9;
  border-left: 6px solid #ef4444;
  padding: 10px 12px;
  border-radius: 6px;
  font-weight: 800;
  color:#111827;
}
.chips { margin-top: 12px; display:flex; gap:10px; flex-wrap:wrap; }
.chip{
  background:#e5e7eb;
  color:#111827;
  padding: 8px 12px;
  border-radius: 6px;
  font-weight: 800;
  font-size: 13px;
}
.chip-green{
  background:#d1fae5;
  border:1px solid #22c55e;
}
.chip-red{
  background:#fee2e2;
  border:1px solid #ef4444;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# HELPERS (math)
# ----------------------------
def safe_div(n, d, mult=1.0):
    return (n / d) * mult if d and d != 0 else 0.0

def pct_change(new, old):
    if old == 0:
        return 0.0 if new == 0 else -100.0
    return (new - old) / old * 100.0

def fmt_compact(x):
    # 120000 -> 120K, 1200000 -> 1.2M
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{sign}{x/1_000:.0f}K"
    return f"{sign}{x:.0f}"

def kpi_card_html(title, value_str, delta_pct):
    return f"""
    <div class="card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value_str}</div>
      <div class="delta-badge">{delta_pct:+.0f}%</div>
    </div>
    """

def mini_kpi_html(title, value_str, delta_pct):
    # mimic your “small red percent + underline bar”
    if abs(delta_pct) < 1:
        bar = "bar-green"
        delta_txt = "= 0%"
        delta_color = "#111827"
    else:
        bar = "bar-red" if delta_pct < 0 else "bar-green"
        arrow = "▼" if delta_pct < 0 else "▲"
        delta_txt = f"{arrow}{abs(delta_pct):.0f}%"
        delta_color = "#ef4444" if delta_pct < 0 else "#16a34a"

    return f"""
    <div class="card" style="padding:12px 14px;">
      <div class="mini-title">{title}</div>
      <div style="display:flex; align-items:baseline; gap:8px; margin-top:6px;">
        <div class="mini-value">{value_str}</div>
        <div class="mini-delta" style="color:{delta_color};">{delta_txt}</div>
      </div>
      <div class="{bar}"></div>
    </div>
    """

# ----------------------------
# LOAD DATA
# ----------------------------
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

# If you have multi-site in the CSV
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

# Daily totals (ONLY totals are summed)
daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived metrics from totals (NEVER SUM ratios)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["requests_per_page"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
daily["ctr"] = daily.apply(lambda r: safe_div(r["clicks"], r["impressions"], 100), axis=1)

# ----------------------------
# CONTROLS
# ----------------------------
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
if baseline_df.empty:
    # no baseline available
    exp = {c: 0.0 for c in base_cols}
    exp.update({"ecpm": 0.0, "fill_rate": 0.0, "requests_per_page": 0.0, "ctr": 0.0})
else:
    # Expected totals = median of totals (baseline)
    exp_tot = baseline_df[base_cols].median(numeric_only=True).to_dict()
    exp = dict(exp_tot)
    exp["ecpm"] = safe_div(exp["revenue"], exp["impressions"], 1000)
    exp["fill_rate"] = safe_div(exp["impressions"], exp["ad_requests"], 100)
    exp["requests_per_page"] = safe_div(exp["ad_requests"], exp["pageviews"], 1)
    exp["ctr"] = safe_div(exp["clicks"], exp["impressions"], 100)

# Deltas vs expected (this is what your screenshot shows)
d_rev = pct_change(t["revenue"], exp["revenue"])
d_ecpm = pct_change(t["ecpm"], exp["ecpm"])
d_fill = pct_change(t["fill_rate"], exp["fill_rate"])
d_rpp  = pct_change(t["requests_per_page"], exp["requests_per_page"])

d_users = pct_change(t["users"], exp["users"])
d_pv    = pct_change(t["pageviews"], exp["pageviews"])
d_req   = pct_change(t["ad_requests"], exp["ad_requests"])
d_imp   = pct_change(t["impressions"], exp["impressions"])

# ----------------------------
# DIAGNOSIS (leakage pipeline)
# ----------------------------
def diagnose(t, exp):
    """
    Output:
      issue_title: short label like "Setup Problem"
      chips: list of (label, color) where color in {"red","green","neutral"}
    Logic follows your stated pipeline:
      Pageviews stable + Requests/Page down → Tag issue / Ad density drop (setup)
      Requests stable + Fill down → Demand/fill problem
      Impressions stable + eCPM down → Pricing/eCPM problem
      Pageviews down big + users/sessions down → Traffic issue
    """
    chips = []

    # Quick helpers
    pv_stable = exp["pageviews"] > 0 and (t["pageviews"] >= exp["pageviews"] * 0.95)
    rpp_drop  = exp["requests_per_page"] > 0 and (t["requests_per_page"] <= exp["requests_per_page"] * 0.90)
    req_drop  = exp["ad_requests"] > 0 and (t["ad_requests"] <= exp["ad_requests"] * 0.90)
    fill_drop = exp["fill_rate"] > 0 and (t["fill_rate"] <= exp["fill_rate"] * 0.90)
    ecpm_drop = exp["ecpm"] > 0 and (t["ecpm"] <= exp["ecpm"] * 0.90)

    traffic_drop = exp["users"] > 0 and (t["users"] <= exp["users"] * 0.90)

    # 1) Setup / tag / density leak: PV ok but requests/page down
    if pv_stable and rpp_drop:
        issue = "Setup Problem"
        chips.append(("Tag Issue", "green"))          # green label like your image chip
        chips.append(("Ad Density Drop", "neutral"))
        return issue, chips

    # 2) Traffic issue: PV/users down
    if traffic_drop and (exp["pageviews"] > 0 and t["pageviews"] <= exp["pageviews"] * 0.90):
        issue = "Traffic Problem"
        chips.append(("Traffic Drop", "red"))
        chips.append(("Content/SEO/Source Mix", "neutral"))
        return issue, chips

    # 3) Fill problem: requests exist but fill down
    if (not req_drop) and fill_drop:
        issue = "Demand / Fill Problem"
        chips.append(("Fill Down", "red"))
        chips.append(("Floors/Blocks/Policy", "neutral"))
        return issue, chips

    # 4) Pricing problem: impressions ok but ecpm down
    imp_stable = exp["impressions"] > 0 and (t["impressions"] >= exp["impressions"] * 0.95)
    if imp_stable and ecpm_drop:
        issue = "Pricing / eCPM Problem"
        chips.append(("eCPM Down", "red"))
        chips.append(("Geo/Device Mix", "neutral"))
        return issue, chips

    # 5) Default: mixed
    issue = "Mixed Movement"
    chips.append(("Check Segments", "neutral"))
    return issue, chips

issue_title, chips = diagnose(t, exp)

# ----------------------------
# HEADER
# ----------------------------
domain_title = site if site else "abc.com"
st.markdown(
    f'<div class="header-bar">Publisher Performance Dashboard <span style="font-weight:600;">for {domain_title}</span></div>',
    unsafe_allow_html=True,
)
st.write("")  # spacing

# ----------------------------
# TOP KPI ROW (4 tiles like image)
# ----------------------------
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4], gap="large")

with c1:
    st.markdown(kpi_card_html("Revenue", f"${t['revenue']:,.0f}", d_rev), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card_html("eCPM", f"${t['ecpm']:.2f}", d_ecpm), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card_html("Fill Rate", f"{t['fill_rate']:.0f}%", d_fill), unsafe_allow_html=True)
with c4:
    # Requests Per Page has a split area in your image (value + %)
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">Requests Per Page</div>
          <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:6px;">
            <div class="kpi-value" style="font-size:38px;">{t['requests_per_page']:.1f}</div>
            <div style="width:1px; height:56px; background:#e5e7eb;"></div>
            <div style="text-align:center;">
              <div class="kpi-value" style="font-size:34px;">{d_rpp:+.0f}%</div>
              <div style="height:6px; background:#ef4444; border-radius:10px; margin-top:8px;"></div>
            </div>
          </div>
          <div class="delta-badge">{d_rpp:+.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------
# MIDDLE ROW (Diagnosis + Key Metrics Overview)
# ----------------------------
left, right = st.columns([1.2, 1.4], gap="large")

with left:
    chips_html = ""
    for label, color in chips:
        klass = "chip"
        if color == "red":
            klass = "chip chip-red"
        elif color == "green":
            klass = "chip chip-green"
        chips_html += f'<div class="{klass}">{label}</div>'

    st.markdown(
        f"""
        <div class="card">
          <div class="diag-title">Diagnosis</div>
          <div class="diag-strip">Issue Detected: &nbsp; <span style="font-weight:900;">{issue_title}</span></div>
          <div class="chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="card"><div class="diag-title">Key Metrics Overview</div></div>', unsafe_allow_html=True)
    # Embed mini cards inside columns to match screenshot
    r1, r2, r3, r4 = st.columns(4, gap="large")
    with r1:
        st.markdown(mini_kpi_html("Users", fmt_compact(t["users"]), d_users), unsafe_allow_html=True)
    with r2:
        st.markdown(mini_kpi_html("Pageviews", fmt_compact(t["pageviews"]), d_pv), unsafe_allow_html=True)
    with r3:
        st.markdown(mini_kpi_html("Ad Requests", fmt_compact(t["ad_requests"]), d_req), unsafe_allow_html=True)
    with r4:
        st.markdown(mini_kpi_html("Matched Impressions", fmt_compact(t["impressions"]), d_imp), unsafe_allow_html=True)

st.write("")

# ----------------------------
# REVENUE BREAKDOWN (last 7 days, like image)
# Bars: Pageviews
# Lines: Requests/Page, Fill Rate, eCPM
# ----------------------------
st.markdown(
    """
    <div class="card">
      <div class="diag-title">Revenue Breakdown</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# last 7 days ending today (if less, it will show whatever exists)
last7 = daily[daily["date"] <= today].tail(7).copy()
if last7.empty:
    st.info("Not enough data to plot.")
    st.stop()

# Label days like Mon/Tue...
last7["dow"] = last7["date"].dt.day_name().str.slice(0, 3)

# Altair layered chart:
# - bars for pageviews
# - lines for requests_per_page, fill_rate, ecpm
# We use independent scales (closest to your screenshot feel).
base = alt.Chart(last7).encode(
    x=alt.X("dow:N", sort=list(last7["dow"]), title=None)
)

bars = base.mark_bar(opacity=0.95).encode(
    y=alt.Y("pageviews:Q", title=None),
    color=alt.value("#1f77ff"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("pageviews:Q", title="Pageviews", format=",")
    ]
)

line_rpp = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("requests_per_page:Q", title=None),
    color=alt.value("#f59e0b"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("requests_per_page:Q", title="Requests/Page", format=",.2f")
    ]
)

line_fill = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("fill_rate:Q", title=None),
    color=alt.value("#22c55e"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("fill_rate:Q", title="Fill Rate %", format=",.1f")
    ]
)

line_ecpm = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("ecpm:Q", title=None),
    color=alt.value("#7c3aed"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("ecpm:Q", title="eCPM", format="$.2f")
    ]
)

chart = alt.layer(bars, line_rpp, line_fill, line_ecpm).resolve_scale(
    y="independent"
).properties(
    height=320
).configure_view(
    stroke=None
).configure_axis(
    labelColor="#334155",
    title=None,
    gridColor="#e5e7eb"
).configure_legend(
    orient="top",
    title=None,
    labelFontSize=13
)

# Custom legend labels (match your screenshot)
legend_df = pd.DataFrame({
    "Metric": ["Pageviews", "Requests/Page", "Fill Rate", "eCPM"],
    "Color": ["#1f77ff", "#f59e0b", "#22c55e", "#7c3aed"]
})
legend = alt.Chart(legend_df).mark_point(filled=True, size=140).encode(
    y=alt.value(0),
    x=alt.X("Metric:N", axis=alt.Axis(labelAngle=0, title=None)),
    color=alt.Color("Color:N", scale=None, legend=None),
).properties(height=40)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Notes: Totals are summed per day. Ratios (eCPM, Fill Rate, Requests/Page, CTR) are derived from totals each day."
)
