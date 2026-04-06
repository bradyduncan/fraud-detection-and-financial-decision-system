# dashboard/pages/5_credit.py - Risk Score Tracker Page

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from dashboard.utils import (
    load_predictions, get_user_card_data,
    render_sidebar, metric_card, USER_COLORS
)

# Sidebar
user, card = render_sidebar()

# Load data
df      = load_predictions()
card_df = get_user_card_data(df, user, card).copy()
color   = USER_COLORS.get(user, "#4f9cf9")

# Risk score engineering: probability, fraud, flags, volatility, trend, amount.
card_df = card_df.sort_values("transaction_date").reset_index(drop=True)

window = 30
base_risk = (
    card_df["fraud_probability"]
    .rolling(window=window, min_periods=1)
    .mean() * 100
)
recent_fraud = (
    card_df["isFraud"]
    .rolling(window=window, min_periods=1)
    .mean() * 100
)
recent_flags = (
    card_df["model_decision"]
    .rolling(window=window, min_periods=1)
    .mean() * 100
)
volatility = (
    card_df["fraud_probability"]
    .rolling(window=window, min_periods=1)
    .std()
    .fillna(0) * 200
).clip(0, 100)
amt_mean = card_df["TransactionAmt"].mean()
amt_std = card_df["TransactionAmt"].std()
amt_z = (card_df["TransactionAmt"] - amt_mean) / (amt_std + 1e-9)
amount_risk = (
    amt_z
    .rolling(window=window, min_periods=1)
    .mean()
    .clip(-2, 2)
    .add(2)
    .mul(25)
)
trend_points = (base_risk - base_risk.shift(window)).clip(lower=0)
trend_scaled = (trend_points * 2).clip(0, 100)

card_df["rolling_risk"] = (
    0.40 * base_risk
    + 0.20 * recent_fraud
    + 0.10 * recent_flags
    + 0.15 * volatility
    + 0.10 * trend_scaled
    + 0.05 * amount_risk
).clip(0, 100)

# Overall risk score (last 30 transactions)
current_risk  = card_df["rolling_risk"].iloc[-1]
peak_risk     = card_df["rolling_risk"].max()
avg_risk      = card_df["rolling_risk"].mean()
risk_trend    = card_df["rolling_risk"].iloc[-1] - card_df["rolling_risk"].iloc[-30] \
                if len(card_df) >= 30 else 0

# Risk band
def get_risk_band(score):
    if score < 20:
        return "Low Risk", "#22c55e"
    elif score < 50:
        return "Medium Risk", "#f97316"
    else:
        return "High Risk", "#ef4444"

band_label, band_color = get_risk_band(current_risk)

# Monthly risk summary
card_df["month"] = card_df["transaction_date"].dt.to_period("M").astype(str)
monthly_risk = card_df.groupby("month").agg(
    avg_risk=("rolling_risk", "mean"),
    max_risk=("rolling_risk", "max"),
    fraud_count=("isFraud", "sum"),
    tx_count=("TransactionAmt", "count")
).reset_index()

# Page header
st.markdown(f"""
<div style='margin-bottom: 32px;'>
    <p style='font-size: 12px; color: #6b7280; text-transform: uppercase;
              letter-spacing: 2px; margin: 0;'>Risk Analysis</p>
    <h1 style='font-family: Syne, sans-serif; font-size: 36px;
               font-weight: 800; margin: 4px 0; color: #e8eaf0;'>
        💳 Risk Score Tracker
    </h1>
    <p style='color: #6b7280; margin: 0;'>
        Card risk profile and fraud exposure over time —
        <strong style='color:{color};'>{user} · {card}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Top metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Current Risk Score",
                f"{current_risk:.1f}",
                delta=f"Band: {band_label}",
                icon="🎯", color=band_color)
with c2:
    trend_icon = "↑" if risk_trend > 0 else "↓"
    trend_color = "#ef4444" if risk_trend > 0 else "#22c55e"
    metric_card("Risk Trend (30 tx)",
                f"{trend_icon} {abs(risk_trend):.2f}",
                delta="vs 30 transactions ago",
                icon="📈", color=trend_color)
with c3:
    metric_card("Peak Risk Score",
                f"{peak_risk:.1f}",
                delta="Highest recorded",
                icon="⚠️", color="#f97316")
with c4:
    metric_card("Avg Risk Score",
                f"{avg_risk:.1f}",
                delta="Across all transactions",
                icon="📊", color=color)

st.markdown("<br>", unsafe_allow_html=True)

# Risk score timeline
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    📉 Risk Score Over Time
</p>
""", unsafe_allow_html=True)

# Sample every 10th row for cleaner visualization
plot_df = card_df.iloc[::10].copy()

fig_timeline = go.Figure()

# Risk zones background
fig_timeline.add_hrect(y0=0,  y1=20, fillcolor="rgba(34,197,94,0.05)",
                        line_width=0, annotation_text="Low Risk",
                        annotation_position="left",
                        annotation_font=dict(color="#22c55e", size=10))
fig_timeline.add_hrect(y0=20, y1=50, fillcolor="rgba(249,115,22,0.05)",
                        line_width=0, annotation_text="Medium Risk",
                        annotation_position="left",
                        annotation_font=dict(color="#f97316", size=10))
fig_timeline.add_hrect(y0=50, y1=100, fillcolor="rgba(239,68,68,0.05)",
                        line_width=0, annotation_text="High Risk",
                        annotation_position="left",
                        annotation_font=dict(color="#ef4444", size=10))

# Risk score line
fig_timeline.add_trace(go.Scatter(
    x=plot_df["transaction_date"],
    y=plot_df["rolling_risk"],
    mode="lines",
    name="Risk Score",
    line=dict(color=color, width=2),
    fill="tozeroy",
    fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}",
    hovertemplate="Date: %{x}<br>Risk Score: %{y:.2f}<extra></extra>"
))

# Mark fraud transactions
fraud_plot = card_df[card_df["isFraud"] == 1].iloc[::3]
if len(fraud_plot) > 0:
    fig_timeline.add_trace(go.Scatter(
        x=fraud_plot["transaction_date"],
        y=fraud_plot["rolling_risk"],
        mode="markers",
        name="Actual Fraud",
        marker=dict(color="#ef4444", size=6, symbol="x"),
        hovertemplate="🚨 Fraud Transaction<br>Date: %{x}<br>Risk: %{y:.2f}<extra></extra>"
    ))

fig_timeline.update_layout(
    paper_bgcolor="#13161e",
    plot_bgcolor="#13161e",
    font=dict(color="#9ca3af", family="DM Sans"),
    legend=dict(bgcolor="#13161e", bordercolor="#1e2330",
                orientation="h", y=1.08),
    xaxis=dict(color="#6b7280", gridcolor="#1e2330", title="Date"),
    yaxis=dict(color="#6b7280", gridcolor="#1e2330",
               title="Risk Score (0-100)", range=[0, 100]),
    margin=dict(l=60, r=0, t=40, b=0),
    height=350,
    hovermode="x unified",
)
st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Two column: monthly risk + distribution
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        📅 Monthly Risk Summary
    </p>
    """, unsafe_allow_html=True)

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_risk["month"],
        y=monthly_risk["avg_risk"],
        name="Avg Risk Score",
        marker_color=color,
        opacity=0.8,
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly_risk["month"],
        y=monthly_risk["max_risk"],
        name="Peak Risk Score",
        mode="lines+markers",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=6),
    ))
    fig_monthly.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        legend=dict(bgcolor="#13161e", bordercolor="#1e2330",
                    orientation="h", y=1.12),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330"),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="Risk Score", range=[0, 100]),
        margin=dict(l=0, r=0, t=40, b=0),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

with right:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        🎯 Risk Score Distribution
    </p>
    """, unsafe_allow_html=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=card_df["rolling_risk"],
        nbinsx=40,
        marker_color=color,
        opacity=0.8,
        name="Risk Score",
        hovertemplate="Risk: %{x:.1f}<br>Count: %{y}<extra></extra>"
    ))
    # Add vertical lines for risk bands
    for threshold, band_c, label in [
        (20, "#22c55e", "Low/Medium boundary"),
        (50, "#f97316", "Medium/High boundary")
    ]:
        fig_dist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=band_c,
            annotation_text=label,
            annotation_font=dict(color=band_c, size=10),
            annotation_position="top right"
        )
    fig_dist.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="Risk Score"),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330", title="Count"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Risk recommendations
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    💡 Risk Recommendations
</p>
""", unsafe_allow_html=True)

fraud_rate    = card_df["isFraud"].mean() * 100
flagged_rate  = card_df["model_decision"].mean() * 100
high_risk_pct = (card_df["rolling_risk"] > 50).mean() * 100

# Dynamic recommendations based on card's risk profile
recommendations = []

if current_risk > 50:
    recommendations.append({
        "icon": "🚨",
        "title": "High Risk Detected",
        "text": f"Your current risk score of {current_risk:.1f} is in the High Risk zone. "
                f"Review recent flagged transactions immediately.",
        "color": "#ef4444"
    })
elif current_risk > 20:
    recommendations.append({
        "icon": "⚠️",
        "title": "Moderate Risk",
        "text": f"Your risk score of {current_risk:.1f} is moderate. "
                f"Monitor transactions closely over the next few weeks.",
        "color": "#f97316"
    })
else:
    recommendations.append({
        "icon": "✅",
        "title": "Low Risk Profile",
        "text": f"Your risk score of {current_risk:.1f} is in the safe zone. "
                f"Your card activity appears normal.",
        "color": "#22c55e"
    })

if fraud_rate > 3:
    recommendations.append({
        "icon": "🔒",
        "title": "Above Average Fraud Rate",
        "text": f"This card has a {fraud_rate:.1f}% fraud rate, above the typical 1-2%. "
                f"Consider enabling transaction alerts with your bank.",
        "color": "#f97316"
    })

if risk_trend > 2:
    recommendations.append({
        "icon": "📈",
        "title": "Rising Risk Trend",
        "text": f"Your risk score has increased by {risk_trend:.2f} points "
                f"over the last 30 transactions. Stay vigilant.",
        "color": "#ef4444"
    })
else:
    recommendations.append({
        "icon": "📉",
        "title": "Stable Risk Trend",
        "text": f"Your risk trend is stable or improving. "
                f"No unusual patterns detected in recent activity.",
        "color": "#22c55e"
    })

# Render recommendations
cols = st.columns(len(recommendations))
for i, rec in enumerate(recommendations):
    with cols[i]:
        st.markdown(f"""
        <div style='background:#13161e; border:1px solid #1e2330;
                    border-top:3px solid {rec["color"]};
                    border-radius:10px; padding:18px; height:160px;'>
            <p style='margin:0; font-size:20px;'>{rec["icon"]}</p>
            <p style='margin:8px 0 6px 0; font-size:14px; font-weight:700;
                      color:{rec["color"]};'>{rec["title"]}</p>
            <p style='margin:0; font-size:12px; color:#9ca3af; 
                      line-height:1.5;'>{rec["text"]}</p>
        </div>
        """, unsafe_allow_html=True)
