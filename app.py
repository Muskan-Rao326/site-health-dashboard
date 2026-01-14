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
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

/* Header */
.header{
  background: linear-gradient(90deg, #183a63, #244f7c);
  border-radius: 12px;
  padding: 18px 22px;
  color: #fff;
  box-shadow: 0 10px 22px rgba(15,23,42,0.20);
}
.header-title{ font-size: 28px; font-weight: 900; line-height: 1.15; }
.header-sub{ margin-top: 4px; font-size: 13px; opacity: 0.85; font-weight: 600; }

/* Panels */
.panel{
  background: #ffffff;
  border: 1px solid #e7eef7;
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
  overflow: hidden;
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
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
  height: 120px;
}
.kpi-label{ font-size: 14px; font-weight: 800; color:#0f172a; opacity: 0.85; }
.kpi-value{ margin-top: 6px; font-size: 40px; font-weight: 950; color:#0b1220; line-height: 1.05; }

/* ✅ NEW: 2 small badges row inside KPI */
.kpi-badges{
  margin-top: 10px;
  display:flex;
  gap:8px;
  flex-wrap:wrap;
}
.kpi-badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 900;
  color:#fff;
}

.badge-red{ background:#d9534f; }
.badge-green{ background:#16a34a; }
.badge-gray{ background:#64748b; }

/* Mini KPI */
.mini-card{
  background:#ffffff;
  border: 1px solid #e7eef7;
  border-radius: 12px;
  padding: 12px 14px;
  box-shadow: 0 6px 16px rgba(15,23,42,0.06);
  height: 104px;
  min-width: 0;
}
.mini-label{
  font-size: 12px;
  font-weight: 900;
  color:#334155;
  line-height: 1.15;
  white-space: normal;
  word-break: break-word;
  overflow-wrap:anywhere;
}
.mini-row{
  margin-top: 6px;
  display:flex;
  align-items: baseline;
  gap: 8px;
  min-width:0;
  flex-wrap: nowrap;
}
.mini-value{
  font-size: 26px;
  font-weight: 950;
  color:#0b1220;
  line-height: 1.0;
  min-width:0;
}
.mini-delta{
  font-size: 12px;
  font-weight: 950;
  white-space: nowrap;
}
.mini-bar{
  height:6px;
  border-radius: 10px;
  margin-top: 10px;
  background:#e2e8f0;
  overflow:hidden;
}
.mini-bar > div{ height:100%; border-radius:10px; }

/* Diagnosis strip + chips */
.diag-strip{
  background: #f6d9c9;
  border-left: 6px solid #ef4444;
  padding: 10px 12px;
  border-radius: 10px;
  font-weight: 950;
  color:#0b1220;
}
.chips{
  margin-top: 12px;
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  align-items:flex-start;
  max-width:100%;
  min-width:0;
}
.chip{
  padding: 8px 12px;
  border-radius: 10px;
  font-weight: 900;
  font-size: 12px;
  border: 1px solid #e5e7eb;
  background:#f1f5f9;
  color:#0b1220;
  flex: 0 1 auto;
  min-width: 0;
  max-width: 100%;
  white-space: normal;
  word-break: break-word;
  overflow-wrap:anywhere;
  line-height: 1.15;
}
.chip-green{ background:#dcfce7; border-color:#22c55e; }
.chip-red{ background:#fee2e2; border-color:#ef4444; }
.chip-amber{ background:#ffedd5; border-color:#f59e0b; }

/* Miss chips */
.miss-chip{
  padding:10px 12px;
  border-radius:10px;
  font-weight:950;
  font-size:12px;
  border:1px solid #e5e7eb;
  background:#f1f5f9;
  color:#0b1220;
  width:100%;
  text-align:center;
}
.miss-chip-imp{ background:#dbeafe; border-color:#3b82f6; }
.miss-chip-ecpm{ background:#fee2e2; border-color:#ef4444; }
.miss-chip-res{ background:#ede9fe; border-color:#7c3aed; }

.small-note{ font-size: 12px; color:#475569; font-weight: 750; margin-top: 8px; }
.body-text{ font-size: 13px; color:#0b1220; font-weight: 850; line-height: 1.55; }

hr.soft { border:none; height:1px; background:#e7eef7; margin: 12px 0; }

/* Altair background fix */
.vega-embed, .vega-embed details, .vega-embed summary { background: transparent !important; }
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
    new = float(new)
    old = float(old)
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
    if delta_pct is None:
        return "badge-gray"
    if isinstance(delta_pct, float) and np.isnan(delta_pct):
        return "badge-gray"
    return "badge-red" if float(delta_pct) < 0 else "badge-green"

# ✅ UPDATED: KPI card shows 2 badges (Baseline + Yesterday)
def kpi_card(label, value_str, delta_baseline_pct, delta_yday_pct):
    b1 = badge_class(delta_baseline_pct)

    if delta_yday_pct is None or (isinstance(delta_yday_pct, float) and np.isnan(delta_yday_pct)):
        b2 = "badge-gray"
        y_txt = "vs Yday N/A"
    else:
        b2 = badge_class(delta_yday_pct)
        y_txt = f"vs Yday {float(delta_yday_pct):+.0f}%"

    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value_str}</div>
      <div class="kpi-badges">
        <span class="kpi-badge {b1}">vs Base {float(delta_baseline_pct):+.0f}%</span>
        <span class="kpi-badge {b2}">{y_txt}</span>
      </div>
    </div>
    """

def mini_card(label, value_str, delta_pct):
    mag = min(abs(delta_pct) / 40.0, 1.0)
    color = "#ef4444" if delta_pct < 0 else "#16a34a"
    sign = "▼" if delta_pct < 0 else "▲"
    if abs(delta_pct) < 1:
        txt = "=0%"
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

def zscore_label(value: float, baseline_series: pd.Series) -> str:
    s = pd.to_numeric(baseline_series, errors="coerce").dropna()
    if len(s) < 6:
        return "Noise-like movement"
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd <= 1e-9:
        if mu == 0:
            return "Noise-like movement"
        p = safe_div(value - mu, mu, 100)
        return "Statistically abnormal" if abs(p) >= 20 else "Noise-like movement"
    z = (value - mu) / sd
    return "Statistically abnormal" if abs(z) >= 2 else "Noise-like movement"

def altair_theme_clean():
    return (
        alt.theme.ThemeConfig(
            config={
                "background": "transparent",
                "view": {"stroke": None, "fill": "white"},
                "axis": {
                    "labelColor": "#334155",
                    "titleColor": "#0f172a",
                    "gridColor": "#e5e7eb",
                    "tickColor": "#cbd5e1",
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                },
                "legend": {"labelColor": "#334155", "titleColor": "#0f172a"},
            }
        )
    )

alt.themes.register("clean", altair_theme_clean)
alt.themes.enable("clean")

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

# Derived ratios
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["sessions"], 1000), axis=1)
daily["rpu"] = daily.apply(lambda r: safe_div(r["revenue"], r["users"], 1), axis=1)
daily["pv_per_session"] = daily.apply(lambda r: safe_div(r["pageviews"], r["sessions"], 1), axis=1)

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

# =========================
# EXPECTED (baseline median totals)
# =========================
if baseline_df.empty:
    exp = {c: 0.0 for c in base_cols}
    exp.update({"ecpm": 0.0, "fill_rate": 0.0, "rpm": 0.0, "rpu": 0.0, "pv_per_session": 0.0})
else:
    exp_tot = baseline_df[base_cols].median(numeric_only=True).to_dict()
    exp = dict(exp_tot)
    exp["ecpm"] = safe_div(exp["revenue"], exp["impressions"], 1000)
    exp["fill_rate"] = safe_div(exp["impressions"], exp["ad_requests"], 100)
    exp["rpm"] = safe_div(exp["revenue"], exp["sessions"], 1000)
    exp["rpu"] = safe_div(exp["revenue"], exp["users"], 1)
    exp["pv_per_session"] = safe_div(exp["pageviews"], exp["sessions"], 1)

# Deltas vs Expected (baseline)
d = {}
for k in ["revenue", "ecpm", "fill_rate", "ad_requests", "users", "sessions", "pageviews", "impressions", "rpm", "rpu"]:
    d[k] = pct_change(t.get(k, 0.0), exp.get(k, 0.0))

# ✅ NEW: Deltas vs Yesterday
yday = today - timedelta(days=1)
yday_df = daily[daily["date"] == yday]
if yday_df.empty:
    y = None
else:
    y = yday_df.iloc[0].to_dict()

d_y = {}
for k in ["revenue", "ecpm", "fill_rate", "ad_requests", "users", "sessions", "pageviews", "impressions", "rpm", "rpu"]:
    if y is None:
        d_y[k] = np.nan
    else:
        d_y[k] = pct_change(t.get(k, 0.0), y.get(k, 0.0))

# Z-score label (revenue)
noise_label = "Noise-like movement"
if not baseline_df.empty:
    noise_label = zscore_label(float(t["revenue"]), baseline_df["revenue"])

# =========================
# HEADER
# =========================
domain = site if site else "abc.com"
st.markdown(
    f"""
    <div class="header">
      <div class="header-title">Publisher Performance Dashboard <span style="font-weight:650;">for {domain}</span></div>
      <div class="header-sub">Expected = baseline median of last {baseline_days} days (excluding selected date) • Top cards show vs Baseline + vs Yesterday</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# =========================
# TOP KPI ROW
# =========================
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2], gap="large")
with c1:
    st.markdown(
        kpi_card("Revenue", f"${float(t['revenue']):,.2f}", d["revenue"], d_y["revenue"]),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        kpi_card("eCPM", f"${float(t['ecpm']):,.2f}", d["ecpm"], d_y["ecpm"]),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        kpi_card("Fill Rate", f"{float(t['fill_rate']):.0f}%", d["fill_rate"], d_y["fill_rate"]),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        kpi_card("Ad Requests", fmt_compact(t["ad_requests"]), d["ad_requests"], d_y["ad_requests"]),
        unsafe_allow_html=True,
    )

st.write("")

# =========================
# DIAGNOSIS
# =========================
def pipeline_diagnosis(t, exp):
    chips = []
    story = []

    gap = float(exp["revenue"]) - float(t["revenue"])
    if gap > 0:
        story.append(f"Today missed expected revenue by ${gap:,.2f} (Expected baseline median).")
    else:
        story.append("Revenue is at/above Expected baseline for this day.")

    users_down = exp["users"] > 0 and t["users"] <= exp["users"] * 0.90
    sessions_down = exp["sessions"] > 0 and t["sessions"] <= exp["sessions"] * 0.90
    pvps_down = exp["pv_per_session"] > 0 and t["pv_per_session"] <= exp["pv_per_session"] * 0.90

    if users_down or sessions_down:
        issue = "Traffic Problem"
        if users_down: chips.append(("Users Down", "red"))
        if sessions_down: chips.append(("Sessions Down", "red"))
        story.append("Traffic dropped → fewer sessions/pageviews → fewer ad opportunities.")
        return issue, chips, story

    if pvps_down:
        issue = "Engagement Problem"
        chips.append(("PV/Session Down", "red"))
        story.append("People are viewing fewer pages per session → fewer ads shown per visit.")
        return issue, chips, story

    issue = "Delivery / Price Check"
    chips.append(("Traffic OK", "green"))
    story.append("Next step: validate fill rate + eCPM movement for delivery/price impact.")
    return issue, chips, story

issue_title, chips, story_lines = pipeline_diagnosis(t, exp)

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

    story_html = "<br/>".join([f"• {s}" for s in story_lines]) if story_lines else "• No story notes."

    inner = f"""
      <div class="diag-strip">Issue Detected: <span style="font-weight:950;">{issue_title}</span></div>
      <div class="chips">{chips_html}</div>
      <hr class="soft"/>
      <div class="body-text">{story_html}</div>
      <div class="small-note"><b>Signal check:</b> {noise_label} (revenue vs baseline variability)</div>
    """
    st.markdown(panel("Diagnosis", inner), unsafe_allow_html=True)

with right:
    inner = f"""
    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px;">
      {mini_card("Users", fmt_compact(t["users"]), d["users"])}
      {mini_card("Sessions", fmt_compact(t["sessions"]), d["sessions"])}
      {mini_card("Pageviews", fmt_compact(t["pageviews"]), d["pageviews"])}
      {mini_card("Impressions", fmt_compact(t["impressions"]), d["impressions"])}
      {mini_card("RPM (Rev/1K Sessions)", f"${float(t['rpm']):.2f}", d["rpm"])}
      {mini_card("RPU (Rev/User)", f"${float(t['rpu']):.4f}", d["rpu"])}
    </div>
    <div class="small-note">All deltas are vs <b>Expected baseline median</b> (totals median; ratios derived from totals).</div>
    """
    st.markdown(panel("Key Metrics Overview", inner), unsafe_allow_html=True)

st.write("")

# =========================
# LOSS + DRIVER CARDS
# =========================
s1, s2, s3 = st.columns([1.2, 1, 1], gap="large")

gap = float(exp["revenue"]) - float(t["revenue"])
gap_pct = safe_div(gap, float(exp["revenue"]), 100) if float(exp["revenue"]) > 0 else 0.0
status = "Normal" if gap_pct <= 5 else "Watch" if gap_pct <= 15 else "Critical"
status_color = "#16a34a" if status == "Normal" else "#f59e0b" if status == "Watch" else "#ef4444"

with s1:
    inner = f"""
    <div style="display:flex; justify-content:space-between; gap:10px; align-items:baseline;">
      <div>
        <div style="font-size:13px; font-weight:900; color:#334155;">Today vs Expected</div>
        <div style="font-size:36px; font-weight:950; color:#0b1220; margin-top:4px;">
          {"-" if gap > 0 else "+"}{abs(gap):,.2f}
        </div>
        <div class="small-note">Expected: ${float(exp['revenue']):,.2f} • Actual: ${float(t['revenue']):,.2f}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:13px; font-weight:900; color:#334155;">Status</div>
        <div style="font-size:22px; font-weight:950; color:{status_color}; margin-top:4px;">{status}</div>
        <div class="small-note">Gap: {gap_pct:+.1f}%</div>
        <div class="small-note">{noise_label}</div>
      </div>
    </div>
    """
    st.markdown(panel("Loss Meter", inner), unsafe_allow_html=True)

with s2:
    st.markdown('<div class="panel"><div class="panel-title">Why Revenue Missed</div>', unsafe_allow_html=True)

    if gap > 0 and float(exp["revenue"]) > 0:
        imp_b = float(exp["impressions"])
        imp_t = float(t["impressions"])
        ecpm_b = float(exp["ecpm"])
        ecpm_t = float(t["ecpm"])

        delta_rev = float(t["revenue"]) - float(exp["revenue"])
        imp_effect = (imp_t - imp_b) * (ecpm_b / 1000.0)
        ecpm_effect = imp_t * ((ecpm_t - ecpm_b) / 1000.0)
        residual = delta_rev - (imp_effect + ecpm_effect)

        loss_imp = max(-imp_effect, 0.0)
        loss_ecpm = max(-ecpm_effect, 0.0)
        loss_res = max(-residual, 0.0)
        denom = (loss_imp + loss_ecpm + loss_res) if (loss_imp + loss_ecpm + loss_res) > 0 else 1.0

        p_imp = (loss_imp / denom) * 100
        p_ecpm = (loss_ecpm / denom) * 100
        p_res = (loss_res / denom) * 100

        st.markdown("<div class='body-text'>Miss Split (loss-only)</div>", unsafe_allow_html=True)

        a, b, c = st.columns(3)
        with a:
            st.markdown(f"<div class='miss-chip miss-chip-imp'>Impressions<br><b>{p_imp:.0f}%</b></div>", unsafe_allow_html=True)
        with b:
            st.markdown(f"<div class='miss-chip miss-chip-ecpm'>eCPM<br><b>{p_ecpm:.0f}%</b></div>", unsafe_allow_html=True)
        with c:
            st.markdown(f"<div class='miss-chip miss-chip-res'>Residual<br><b>{p_res:.0f}%</b></div>", unsafe_allow_html=True)

        st.markdown("<div class='small-note'>Only the components that explain the revenue miss.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-note'>Revenue is at/above Expected baseline. No miss to explain.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with s3:
    inner = f"""
    <div class="body-text">Second-Level Driver</div>
    <div class="small-note">Revenue moves because of <b>Delivery</b> (Impressions) and <b>Price</b> (eCPM).</div>
    <div style="margin-top:10px; font-size:13px; font-weight:850; color:#0b1220; line-height:1.7;">
      <div><b>Impressions:</b> {fmt_compact(t['impressions'])} (Δ {d['impressions']:+.0f}%)</div>
      <div><b>Fill Rate:</b> {float(t['fill_rate']):.0f}% (Δ {d['fill_rate']:+.0f}%)</div>
      <div><b>eCPM:</b> ${float(t['ecpm']):.2f} (Δ {d['ecpm']:+.0f}%)</div>
    </div>
    """
    st.markdown(panel("Second-Level Driver", inner), unsafe_allow_html=True)

st.write("")

# =========================
# CHART AREA (Readable)
# =========================
last7 = daily[daily["date"] <= today].tail(7).copy()
last7["label"] = last7["date"].dt.strftime("%d %b")

expected_rev = float(exp.get("revenue", 0.0))
expected_fill = float(exp.get("fill_rate", 0.0))
expected_ecpm = float(exp.get("ecpm", 0.0))

driver = "delivery (impressions)" if abs(d["impressions"]) > abs(d["ecpm"]) else "price (eCPM)"
insight_line = f"Last 7 days view: Revenue is {'down' if d['revenue'] < 0 else 'up'} vs baseline. Main signal: {driver}. (Δ {d['revenue']:+.1f}%)"

st.markdown(
    panel(
        "Last 7 Days View",
        f"""
        <div class="small-note">{insight_line}</div>
        <div class="small-note">Revenue chart is separate from Fill Rate and eCPM so it stays readable.</div>
        """,
    ),
    unsafe_allow_html=True,
)

# Revenue chart
rev = alt.Chart(last7).encode(
    x=alt.X("label:N", sort=list(last7["label"]), title=None)
)

rev_bars = rev.mark_bar(opacity=0.9).encode(
    y=alt.Y("revenue:Q", title="Revenue ($)"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("revenue:Q", title="Revenue", format="$.2f"),
        alt.Tooltip("ad_requests:Q", title="Ad Requests", format=","),
        alt.Tooltip("impressions:Q", title="Impressions", format=","),
    ],
)

rev_expected = alt.Chart(last7).mark_rule(strokeWidth=3).encode(
    y=alt.datum(expected_rev)
)

rev_expected_text = alt.Chart(last7.tail(1)).mark_text(
    align="left", dx=6, dy=-8, fontWeight="bold"
).encode(
    y=alt.datum(expected_rev),
    text=alt.value(f"Expected: ${expected_rev:,.2f}")
)

rev_chart = (rev_bars + rev_expected + rev_expected_text).properties(height=260)
st.altair_chart(rev_chart, use_container_width=True)

# Fill rate chart
fill_line = alt.Chart(last7).mark_line(point=True, strokeWidth=3).encode(
    x=alt.X("label:N", sort=list(last7["label"]), title=None),
    y=alt.Y("fill_rate:Q", title="Fill Rate (%)"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("fill_rate:Q", title="Fill Rate", format=".1f"),
    ],
)

fill_expected = alt.Chart(last7).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
    y=alt.datum(expected_fill)
)

st.altair_chart((fill_line + fill_expected).properties(height=190), use_container_width=True)

# eCPM chart
ecpm_line = alt.Chart(last7).mark_line(point=True, strokeWidth=3).encode(
    x=alt.X("label:N", sort=list(last7["label"]), title=None),
    y=alt.Y("ecpm:Q", title="eCPM ($)"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("ecpm:Q", title="eCPM", format="$.2f"),
    ],
)

ecpm_expected = alt.Chart(last7).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
    y=alt.datum(expected_ecpm)
)

st.altair_chart((ecpm_line + ecpm_expected).properties(height=190), use_container_width=True)

st.caption("Expected lines are baseline medians from your selected window.")
st.write("")

# =========================
# SANITY CHECKS
# =========================
checks = []
if t["sessions"] > 0 and t["pageviews"] == 0:
    checks.append("Sessions > 0 but Pageviews = 0 → GA4 pull may be incomplete for this day.")
if t["pageviews"] > 0 and t["ad_requests"] == 0:
    checks.append("Pageviews > 0 but Ad Requests = 0 → tag/ad-call issue OR GAM data missing.")
if t["ad_requests"] > 0 and t["impressions"] == 0:
    checks.append("Ad Requests > 0 but Impressions = 0 → fill collapsed OR GAM reporting lag.")
if t["impressions"] > 0 and t["revenue"] == 0:
    checks.append("Impressions > 0 but Revenue = 0 → revenue reporting lag OR merge mismatch.")
if float(exp["revenue"]) == 0 and not baseline_df.empty:
    checks.append("Baseline expected revenue is 0 but baseline rows exist → baseline median issue (too many zeros).")

if checks:
    st.markdown(panel("Sanity Checks", "<br/>".join([f"• {c}" for c in checks])), unsafe_allow_html=True)
else:
    st.markdown(panel("Sanity Checks", "<span class='small-note'>No obvious integrity red flags for selected day.</span>"), unsafe_allow_html=True)
