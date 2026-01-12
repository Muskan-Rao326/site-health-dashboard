# app.py
# Publisher Performance Dashboard (GA4 + GAM) — Professional + clean (PowerBI-like)
# Layout matches your reference image (excluding the bottom landing pages + ad unit panels)
#
# ✅ 100% correct metric math:
#   - ONLY additive totals are SUMmed per day (users, sessions, pageviews, ad_requests, impressions, clicks, revenue)
#   - ALL ratios are derived from totals (never summed): eCPM, Fill Rate, Requests/Page, CTR, RPM, etc.
#
# ✅ One story reference everywhere:
#   - Expected = baseline median (last N days, excluding selected date)
#   - All “delta %” in KPI tiles are vs Expected (baseline median) to match the image
#
# ✅ Better insights + logic:
#   - “Leakage pipeline” diagnosis: Pageviews → Requests/Page → Fill → Impressions → eCPM → Revenue
#   - Root-cause decomposition (Revenue loss split into Impressions vs eCPM, and Impressions split into Requests vs Fill)
#   - Today vs Expected loss meter + confidence score (residual ratio)
#   - Data sanity checks (merge lag, missing rows, zeros)
#
# ✅ Clean, professional UI:
#   - Header bar, consistent tiles, minimal colors, crisp typography
#   - Charts aligned and labeled like a reporting dashboard

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
  height: 98px;
}
.mini-label{ font-size: 13px; font-weight: 800; color:#0f172a; opacity: 0.85;}
.mini-row{ margin-top: 6px; display:flex; align-items: baseline; gap: 8px; }
.mini-value{ font-size: 28px; font-weight: 950; color:#0b1220; line-height: 1.0; }
.mini-delta{ font-size: 13px; font-weight: 950; }
.mini-bar{ height:6px; border-radius: 10px; margin-top: 10px; background:#e2e8f0; overflow:hidden; }
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
.chips{ margin-top: 12px; display:flex; gap:10px; flex-wrap:wrap; }
.chip{
  padding: 8px 12px;
  border-radius: 10px;
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
# HELPERS (correct math)
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
    # for top KPIs, your screenshot shows red even for mild drops, green for up
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
    # bar fill uses magnitude (capped)
    mag = min(abs(delta_pct) / 40.0, 1.0)  # 40% drop = full bar
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

# Site selector (if present)
site = None
if "site_name" in df_raw.columns:
    sites = sorted(df_raw["site_name"].dropna().unique().tolist())
    site = st.sidebar.selectbox("Site", sites, index=0)
    df_raw = df_raw[df_raw["site_name"] == site].copy()

# Required totals (GA4 + GAM)
base_cols = ["users", "sessions", "pageviews", "ad_requests", "impressions", "clicks", "revenue"]
missing = [c for c in base_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

for c in base_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

# Daily totals (ONLY totals are summed)
daily = df_raw.groupby("date", as_index=False)[base_cols].sum().sort_values("date")

# Derived ratios (NEVER SUM these)
daily["ecpm"] = daily.apply(lambda r: safe_div(r["revenue"], r["impressions"], 1000), axis=1)
daily["fill_rate"] = daily.apply(lambda r: safe_div(r["impressions"], r["ad_requests"], 100), axis=1)
daily["requests_per_page"] = daily.apply(lambda r: safe_div(r["ad_requests"], r["pageviews"], 1), axis=1)
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
display_df = daily[(daily["date"] >= baseline_start - timedelta(days=6)) & (daily["date"] <= today)].copy()

# Expected = baseline median totals
if baseline_df.empty:
    exp = {c: 0.0 for c in base_cols}
    exp.update({"ecpm": 0.0, "fill_rate": 0.0, "requests_per_page": 0.0, "ctr": 0.0, "rpm": 0.0})
else:
    exp_tot = baseline_df[base_cols].median(numeric_only=True).to_dict()
    exp = dict(exp_tot)
    exp["ecpm"] = safe_div(exp["revenue"], exp["impressions"], 1000)
    exp["fill_rate"] = safe_div(exp["impressions"], exp["ad_requests"], 100)
    exp["requests_per_page"] = safe_div(exp["ad_requests"], exp["pageviews"], 1)
    exp["ctr"] = safe_div(exp["clicks"], exp["impressions"], 100)
    exp["rpm"] = safe_div(exp["revenue"], exp["pageviews"], 1000)

# Deltas vs expected (this is what your image is showing)
d = {}
for k in ["revenue", "ecpm", "fill_rate", "requests_per_page", "users", "sessions", "pageviews", "ad_requests", "impressions", "rpm", "ctr"]:
    d[k] = pct_change(t.get(k, 0.0), exp.get(k, 0.0))

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
# TOP KPI ROW (match image)
# =========================
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.35], gap="large")

with c1:
    st.markdown(kpi_card("Revenue", f"${t['revenue']:,.0f}", d["revenue"]), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("eCPM", f"${t['ecpm']:.2f}", d["ecpm"]), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("Fill Rate", f"{t['fill_rate']:.0f}%", d["fill_rate"]), unsafe_allow_html=True)
with c4:
    # Requests/Page with split feel
    st.markdown(
        f"""
        <div class="kpi-card" style="height:120px;">
          <div class="kpi-label">Requests Per Page</div>
          <div style="display:flex; align-items:center; justify-content:space-between; gap:14px; margin-top:6px;">
            <div class="kpi-value" style="font-size:38px;">{t['requests_per_page']:.1f}</div>
            <div style="width:1px; height:54px; background:#e5e7eb;"></div>
            <div style="text-align:center;">
              <div class="kpi-value" style="font-size:34px;">{d['requests_per_page']:+.0f}%</div>
              <div style="height:6px; background:#ef4444; border-radius:10px; margin-top:8px;"></div>
            </div>
          </div>
          <span class="kpi-delta {badge_class(d['requests_per_page'])}">{d['requests_per_page']:+.0f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# =========================
# INSIGHT ENGINE (clean + accurate)
# =========================
def root_cause_story(t, exp):
    """
    Outputs:
      issue_title
      chips (label, type)
      story_lines (short bullets)
      loss_split dict
      confidence (High/Med/Low) based on residual ratio
    Uses ONLY Expected baseline median as story reference.
    """
    # Shortcuts
    rev_t, rev_b = float(t["revenue"]), float(exp["revenue"])
    imp_t, imp_b = float(t["impressions"]), float(exp["impressions"])
    req_t, req_b = float(t["ad_requests"]), float(exp["ad_requests"])

    ecpm_t = safe_div(rev_t, imp_t, 1000)
    ecpm_b = safe_div(rev_b, imp_b, 1000)

    fill_t = safe_div(imp_t, req_t, 100) if req_t else 0.0
    fill_b = safe_div(imp_b, req_b, 100) if req_b else 0.0

    rpp_t = safe_div(req_t, float(t["pageviews"]), 1) if float(t["pageviews"]) else 0.0
    rpp_b = safe_div(req_b, float(exp["pageviews"]), 1) if float(exp["pageviews"]) else 0.0

    # Revenue decomposition: Rev = Imp * eCPM / 1000
    d_rev = rev_t - rev_b
    imp_effect = (imp_t - imp_b) * (ecpm_b / 1000)
    ecpm_effect = imp_t * ((ecpm_t - ecpm_b) / 1000)
    residual = d_rev - (imp_effect + ecpm_effect)

    # Impressions decomposition: Imp = Req * Fill/100
    d_imp = imp_t - imp_b
    req_effect = (req_t - req_b) * (fill_b / 100)
    fill_effect = req_t * ((fill_t - fill_b) / 100)
    imp_residual = d_imp - (req_effect + fill_effect)

    # Confidence from residual ratio
    denom = abs(d_rev) if abs(d_rev) > 1e-9 else max(abs(rev_b), 1.0)
    residual_ratio = abs(residual) / denom
    if residual_ratio <= 0.10:
        conf = "High"
    elif residual_ratio <= 0.25:
        conf = "Medium"
    else:
        conf = "Low"

    # Loss split (only when revenue missed expected)
    gap = rev_b - rev_t  # positive means loss
    loss_imp = max(-imp_effect, 0.0)
    loss_ecpm = max(-ecpm_effect, 0.0)
    loss_res = max(-residual, 0.0)
    denom_loss = (loss_imp + loss_ecpm + loss_res) if (loss_imp + loss_ecpm + loss_res) > 0 else 1.0

    split = {
        "gap": gap,
        "loss_imp_pct": loss_imp / denom_loss * 100,
        "loss_ecpm_pct": loss_ecpm / denom_loss * 100,
        "loss_res_pct": loss_res / denom_loss * 100,
        "imp_effect": imp_effect,
        "ecpm_effect": ecpm_effect,
        "residual": residual,
        "req_effect": req_effect,
        "fill_effect": fill_effect,
        "imp_residual": imp_residual,
        "residual_ratio": residual_ratio,
        "conf": conf,
        "fill_t": fill_t,
        "fill_b": fill_b,
        "rpp_t": rpp_t,
        "rpp_b": rpp_b,
        "ecpm_t": ecpm_t,
        "ecpm_b": ecpm_b,
    }

    # Diagnosis (leakage pipeline)
    pv_t, pv_b = float(t["pageviews"]), float(exp["pageviews"])
    pv_stable = pv_b > 0 and pv_t >= pv_b * 0.95
    rpp_down = rpp_b > 0 and rpp_t <= rpp_b * 0.90

    req_down = req_b > 0 and req_t <= req_b * 0.90
    fill_down = fill_b > 0 and fill_t <= fill_b * 0.90
    ecpm_down = ecpm_b > 0 and ecpm_t <= ecpm_b * 0.90

    users_down = float(exp["users"]) > 0 and float(t["users"]) <= float(exp["users"]) * 0.90

    chips = []
    story = []

    if gap > 0:
        story.append(f"Today missed expected revenue by ${gap:,.0f} (Expected baseline median).")
        story.append(f"Loss split: {split['loss_imp_pct']:.0f}% Impressions, {split['loss_ecpm_pct']:.0f}% eCPM, {split['loss_res_pct']:.0f}% residual.")

    # Priority diagnosis: setup leak first (PV ok but Requests/Page down)
    if pv_stable and rpp_down:
        issue = "Setup Problem"
        chips = [("Tag Issue", "green"), ("Ad Density Drop", "neutral"), ("Requests/Page Down", "red")]
        story.append("Pageviews are stable but Requests/Page dropped → ad calls are missing per pageview (tag/CMP/adblock/template).")
        return issue, chips, story, split

    # Traffic issue
    if users_down and (pv_b > 0 and pv_t <= pv_b * 0.90):
        issue = "Traffic Problem"
        chips = [("Users Down", "red"), ("Pageviews Down", "red")]
        story.append("Traffic is down → fewer pageviews → fewer ad opportunities.")
        return issue, chips, story, split

    # Fill issue (requests exist but fill down)
    if (not req_down) and fill_down:
        issue = "Demand / Fill Problem"
        chips = [("Fill Rate Down", "red"), ("Requests OK", "green")]
        story.append("Requests are present but fill dropped → demand/floors/blocks/policy/buyer loss.")
        return issue, chips, story, split

    # eCPM issue (impressions stable but ecpm down)
    imp_stable = imp_b > 0 and imp_t >= imp_b * 0.95
    if imp_stable and ecpm_down:
        issue = "Pricing / eCPM Problem"
        chips = [("eCPM Down", "red"), ("Impressions OK", "green")]
        story.append("Impressions are stable but eCPM dropped → buyers paid less per 1,000 impressions (mix or demand quality).")
        return issue, chips, story, split

    issue = "Mixed Movement"
    chips = [("Check Segments", "neutral")]
    if gap > 0 and split["conf"] == "Low":
        story.append("Residual is high → this day may have reporting lag / merge mismatch. Treat as directional.")
    return issue, chips, story, split

issue_title, chips, story_lines, split = root_cause_story(t, exp)

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

    conf = split["conf"]
    conf_color = "#16a34a" if conf == "High" else "#f59e0b" if conf == "Medium" else "#ef4444"

    inner = f"""
      <div class="diag-strip">Issue Detected: <span style="font-weight:950;">{issue_title}</span></div>
      <div class="chips">{chips_html}</div>
      <hr class="soft"/>
      <div class="small-note"><b>Story confidence:</b> <span style="color:{conf_color}; font-weight:950;">{conf}</span>
      &nbsp; (Residual ratio {split['residual_ratio']:.2f})</div>
    """
    st.markdown(panel("Diagnosis", inner), unsafe_allow_html=True)

with right:
    inner = f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
      {mini_card("Users", fmt_compact(t["users"]), d["users"])}
      {mini_card("Pageviews", fmt_compact(t["pageviews"]), d["pageviews"])}
      {mini_card("Ad Requests", fmt_compact(t["ad_requests"]), d["ad_requests"])}
      {mini_card("Matched Impressions", fmt_compact(t["impressions"]), d["impressions"])}
    </div>
    <div class="small-note">Deltas above are vs <b>Expected baseline median</b>.</div>
    """
    st.markdown(panel("Key Metrics Overview", inner), unsafe_allow_html=True)

st.write("")

# =========================
# STORY CARDS (clean)
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
          {"-" if gap > 0 else "+"}{abs(gap):,.0f}
        </div>
        <div class="small-note">Expected: ${float(exp['revenue']):,.0f} • Actual: ${float(t['revenue']):,.0f}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:13px; font-weight:900; color:#334155;">Status</div>
        <div style="font-size:22px; font-weight:950; color:{status_color}; margin-top:4px;">{status}</div>
        <div class="small-note">Gap: {gap_pct:+.1f}%</div>
      </div>
    </div>
    """
    st.markdown(panel("Loss Meter", inner), unsafe_allow_html=True)

with s2:
    inner = f"""
    <div style="font-size:13px; font-weight:900; color:#334155;">Loss Split (money story)</div>
    <div style="margin-top:10px; display:flex; flex-direction:column; gap:8px;">
      <div><b>Impressions:</b> {split['loss_imp_pct']:.0f}%</div>
      <div><b>eCPM:</b> {split['loss_ecpm_pct']:.0f}%</div>
      <div><b>Residual:</b> {split['loss_res_pct']:.0f}%</div>
    </div>
    <div class="small-note">Residual is “unexplained” change (often lag or merge mismatch).</div>
    """
    st.markdown(panel("Why Revenue Missed", inner), unsafe_allow_html=True)

with s3:
    # Show the inner split only if impressions are a major driver
    imp_major = split["loss_imp_pct"] >= split["loss_ecpm_pct"]
    if imp_major:
        req_share = 0.0
        fill_share = 0.0
        # Convert impression drivers into revenue-equivalent using expected ecpm
        ecpm_b = split["ecpm_b"]
        loss_req = max(-(split["req_effect"] * (ecpm_b / 1000)), 0.0)
        loss_fill = max(-(split["fill_effect"] * (ecpm_b / 1000)), 0.0)
        denom_rf = (loss_req + loss_fill) if (loss_req + loss_fill) > 0 else 1.0
        req_share = loss_req / denom_rf * 100
        fill_share = loss_fill / denom_rf * 100

        inner = f"""
        <div style="font-size:13px; font-weight:900; color:#334155;">Inside Impressions</div>
        <div style="margin-top:10px; display:flex; flex-direction:column; gap:8px;">
          <div><b>Requests:</b> {req_share:.0f}%</div>
          <div><b>Fill:</b> {fill_share:.0f}%</div>
        </div>
        <div class="small-note">Requests/Page expected {split['rpp_b']:.2f} → today {split['rpp_t']:.2f}</div>
        """
    else:
        inner = f"""
        <div style="font-size:13px; font-weight:900; color:#334155;">eCPM Context</div>
        <div style="margin-top:10px; display:flex; flex-direction:column; gap:8px;">
          <div><b>Expected eCPM:</b> ${split['ecpm_b']:.2f}</div>
          <div><b>Today eCPM:</b> ${split['ecpm_t']:.2f}</div>
          <div><b>CTR:</b> {t['ctr']:.2f}% (diagnostic only)</div>
        </div>
        <div class="small-note">To pinpoint eCPM drop, you’d segment by geo/device/ad unit.</div>
        """
    st.markdown(panel("Second-Level Driver", inner), unsafe_allow_html=True)

st.write("")

# =========================
# REVENUE BREAKDOWN CHART (like image)
# Bars: Pageviews
# Lines: Requests/Page, Fill Rate, eCPM
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
        alt.Tooltip("requests_per_page:Q", title="Requests/Page", format=",.2f"),
        alt.Tooltip("fill_rate:Q", title="Fill Rate %", format=",.1f"),
        alt.Tooltip("ecpm:Q", title="eCPM", format="$.2f"),
    ],
)

line_rpp = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("requests_per_page:Q", title=None),
    color=alt.value("#f59e0b"),
)

line_fill = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("fill_rate:Q", title=None),
    color=alt.value("#22c55e"),
)

line_ecpm = base.mark_line(point=True, strokeWidth=3).encode(
    y=alt.Y("ecpm:Q", title=None),
    color=alt.value("#7c3aed"),
)

chart = alt.layer(bars, line_rpp, line_fill, line_ecpm).resolve_scale(
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

st.caption("All ratio metrics are derived from daily totals. Expected reference is baseline median totals (not averages of ratios).")

st.write("")

# =========================
# OPTIONAL: Decomposition tables (very helpful for ops)
# =========================
with st.expander("Show decomposition numbers (for debugging & audits)"):
    # Revenue decomposition vs expected
    rev_t, rev_b = float(t["revenue"]), float(exp["revenue"])
    imp_t, imp_b = float(t["impressions"]), float(exp["impressions"])
    ecpm_t, ecpm_b = split["ecpm_t"], split["ecpm_b"]

    d_rev = rev_t - rev_b
    imp_effect = split["imp_effect"]
    ecpm_effect = split["ecpm_effect"]
    residual = split["residual"]

    # Impressions decomposition vs expected
    req_t, req_b = float(t["ad_requests"]), float(exp["ad_requests"])
    fill_t, fill_b = split["fill_t"], split["fill_b"]

    d_imp = imp_t - imp_b
    req_effect = split["req_effect"]
    fill_effect = split["fill_effect"]
    imp_residual = split["imp_residual"]

    st.subheader("Revenue decomposition (Expected → Today)")
    st.write(pd.DataFrame([
        {"Component": "Expected Revenue", "Value": rev_b},
        {"Component": "Impressions effect", "Value": imp_effect},
        {"Component": "eCPM effect", "Value": ecpm_effect},
        {"Component": "Residual", "Value": residual},
        {"Component": "Today Revenue", "Value": rev_t},
        {"Component": "Check (Today - Expected)", "Value": d_rev},
    ]))

    st.subheader("Impressions decomposition (Expected → Today)")
    st.write(pd.DataFrame([
        {"Component": "Expected Impressions", "Value": imp_b},
        {"Component": "Requests effect", "Value": req_effect},
        {"Component": "Fill effect", "Value": fill_effect},
        {"Component": "Residual", "Value": imp_residual},
        {"Component": "Today Impressions", "Value": imp_t},
        {"Component": "Check (Today - Expected)", "Value": d_imp},
    ]))

# =========================
# DATA SANITY CHECKS (merge correctness)
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
