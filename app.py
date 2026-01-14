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
.kpi-delta{
  margin-top: 10px;
  display:inline-block;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 900;
  color: #fff;
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

/* Chips */
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

/* Why-revenue-missed chips */
.miss-chips{ margin-top:10px; display:flex; gap:10px; flex-wrap:wrap; max-width:100%; }
.miss-chip{
  padding:8px 12px; border-radius:10px; font-weight:900; font-size:12px;
  border:1px solid #e5e7eb; background:#f1f5f9; color:#0b1220;
}
.miss-chip-imp{ background:#dbeafe; border-color:#3b82f6; }
.miss-chip-ecpm{ background:#fee2e2; border-color:#ef4444; }
.miss-chip-res{ background:#ede9fe; border-color:#7c3aed; }

.small-note{ font-size: 12px; color:#475569; font-weight: 750; margin-top: 8px; }
.body-text{ font-size: 13px; color:#0b1220; font-weight: 850; line-height: 1.55; }

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
    mag = min(abs(delta_pct) / 40.0, 1.0)  # 40% = full bar
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

# Derived ratios (from totals)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)

# RPM + RPU (your definitions)
daily["rpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["sessions"], 1000), axis=1)  # Rev per 1K sessions
daily["rpu"] = daily.apply(lambda r: safe_div(r["revenue"], r["users"], 1), axis=1)        # Rev per user

# Engagement helper (only for diagnosis story)
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

    # expected ratios derived from expected totals
    exp["ecpm"] = safe_div(exp["revenue"], exp["impressions"], 1000)
    exp["fill_rate"] = safe_div(exp["impressions"], exp["ad_requests"], 100)
    exp["rpm"] = safe_div(exp["revenue"], exp["sessions"], 1000)
    exp["rpu"] = safe_div(exp["revenue"], exp["users"], 1)
    exp["pv_per_session"] = safe_div(exp["pageviews"], exp["sessions"], 1)

# Deltas vs Expected
d = {}
for k in ["revenue", "ecpm", "fill_rate", "ad_requests", "users", "sessions", "pageviews", "impressions", "rpm", "rpu"]:
    d[k] = pct_change(float(t.get(k, 0.0)), float(exp.get(k, 0.0)))

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
      <div class="header-sub">Expected = baseline median of last {baseline_days} days (excluding selected date) • Delta = Today vs Expected</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# =========================
# TOP KPI ROW (Revenue, eCPM, Fill Rate, Ad Requests)
# =========================
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.35], gap="large")
with c1:
    st.markdown(kpi_card("Revenue", f"${float(t['revenue']):,.2f}", d["revenue"]), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("eCPM", f"${float(t['ecpm']):,.2f}", d["ecpm"]), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("Fill Rate", f"{float(t['fill_rate']):.0f}%", d["fill_rate"]), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("Ad Requests", fmt_compact(t["ad_requests"]), d["ad_requests"]), unsafe_allow_html=True)

st.write("")

# =========================
# PIPELINE DIAGNOSIS (simple + clean)
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

# =========================
# MIDDLE ROW (Diagnosis + Key Metrics Overview)
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
    # ✅ ONLY 6 chips (2 rows): Users, Sessions, Pageviews, Impressions, RPM, RPU
    inner = f"""
    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px;">
      {mini_card("Users", fmt_compact(t["users"]), d["users"])}
      {mini_card("Sessions", fmt_compact(t["sessions"]), d["sessions"])}
      {mini_card("Pageviews", fmt_compact(t["pageviews"]), d["pageviews"])}
      {mini_card("Impressions", fmt_compact(t["impressions"]), d["impressions"])}
      {mini_card("RPM (Rev/1K Sessions)", f"${float(t['rpm']):.2f}", d["rpm"])}
      {mini_card("RPU (Rev/User)", f"${float(t['rpu']):.4f}", d["rpu"])}
    </div>
    <div class="small-note">
      Clean funnel view: Traffic → Engagement → Ad Opportunities → Delivery → Price → Money.
      <br/>All deltas are vs <b>Expected baseline median</b> (totals median; ratios derived from totals).
    </div>
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
    if gap > 0:
        imp_b = float(exp["impressions"])
        imp_t = float(t["impressions"])
        ecpm_b = float(exp["ecpm"])
        ecpm_t = float(t["ecpm"])

        imp_effect = (imp_t - imp_b) * (ecpm_b / 1000.0)
        ecpm_effect = imp_t * ((ecpm_t - ecpm_b) / 1000.0)
        residual = (float(t["revenue"]) - float(exp["revenue"])) - (imp_effect + ecpm_effect)

        loss_imp = max(-imp_effect, 0.0)
        loss_ecpm = max(-ecpm_effect, 0.0)
        loss_res = max(-residual, 0.0)

        denom = (loss_imp + loss_ecpm + loss_res)
        if denom <= 0:
            p_imp = p_ecpm = p_res = 0.0
        else:
            p_imp = (loss_imp / denom) * 100
            p_ecpm = (loss_ecpm / denom) * 100
            p_res = (loss_res / denom) * 100

        # ✅ Streamlit-native rendering (no HTML render bugs)
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">Why Revenue Missed</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='body-text'>Miss Split (loss-only)</div>", unsafe_allow_html=True)

        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown(
                f"<div class='miss-chip miss-chip-imp'>Impressions: {p_imp:.0f}%</div>",
                unsafe_allow_html=True,
            )
        with cB:
            st.markdown(
                f"<div class='miss-chip miss-chip-ecpm'>eCPM: {p_ecpm:.0f}%</div>",
                unsafe_allow_html=True,
            )
        with cC:
            st.markdown(
                f"<div class='miss-chip miss-chip-res'>Residual: {p_res:.0f}%</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='small-note'>Only the components that explain the revenue miss.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-title">Why Revenue Missed</div>
              <div style="font-size:13px; font-weight:900; color:#16a34a;">Revenue is at/above Expected baseline.</div>
              <div class="small-note">No miss to explain for this date.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with s3:
    inner = f"""
    <div class="body-text">Second-Level Driver</div>
    <div class="small-note">
      Quick context: if Revenue dropped, it’s always some mix of
      <b>Delivery</b> (Impressions) and <b>Price</b> (eCPM).
    </div>
    <div style="margin-top:10px; font-size:13px; font-weight:850; color:#0b1220; line-height:1.7;">
      <div><b>Impressions:</b> {fmt_compact(t['impressions'])} (Δ {d['impressions']:+.0f}%)</div>
      <div><b>Fill Rate:</b> {float(t['fill_rate']):.0f}% (Δ {d['fill_rate']:+.0f}%)</div>
      <div><b>eCPM:</b> ${float(t['ecpm']):.2f} (Δ {d['ecpm']:+.0f}%)</div>
    </div>
    """
    st.markdown(panel("Second-Level Driver", inner), unsafe_allow_html=True)

st.write("")

# =========================
# REVENUE BREAKDOWN CHART (Last 7 days)
# =========================
st.markdown(panel("Revenue Breakdown (Last 7 days)", ""), unsafe_allow_html=True)

last7 = daily[daily["date"] <= today].tail(7).copy()
last7["dow"] = last7["date"].dt.day_name().str.slice(0, 3)

base = alt.Chart(last7).encode(
    x=alt.X("dow:N", sort=list(last7["dow"]), title=None)
)

bars = base.mark_bar(opacity=0.9).encode(
    y=alt.Y("pageviews:Q", title=None),
    color=alt.value("#1f77ff"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("pageviews:Q", title="Pageviews", format=","),
        alt.Tooltip("ad_requests:Q", title="Ad Requests", format=","),
        alt.Tooltip("fill_rate:Q", title="Fill Rate %", format=",.1f"),
        alt.Tooltip("ecpm:Q", title="eCPM", format="$.2f"),
        alt.Tooltip("revenue:Q", title="Revenue", format="$.2f"),
    ],
)

line_fill = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("fill_rate:Q", title=None),
    color=alt.value("#22c55e"),
)

line_ecpm = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("ecpm:Q", title=None),
    color=alt.value("#7c3aed"),
)

chart = alt.layer(bars, line_fill, line_ecpm).resolve_scale(
    y="independent"
).properties(height=320).configure_view(
    stroke=None
).configure_axis(
    labelColor="#334155",
    gridColor="#e5e7eb",
    tickColor="#cbd5e1",
).configure_legend(
    orient="top",
    title=None
)

st.altair_chart(chart, use_container_width=True)
st.caption("Expected = baseline median (totals). Delta = Today vs Expected. Ratios derived from totals (not summed).")

st.write("")

# =========================
# DATA SANITY CHECKS
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
