# pages/1_home.py — Overview Page
# Place at: pages/1_home.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dashboard.utils import (
    load_predictions, get_user_card_data,
    render_sidebar, metric_card, USER_COLORS
)

# Sidebar + User Selection
user, card = render_sidebar()

# Load Data 
df      = load_predictions()
card_df = get_user_card_data(df, user, card)
color   = USER_COLORS.get(user, "#4f9cf9")

# Page Header
st.markdown(f"""
<div style='margin-bottom: 32px;'>
    <p style='font-size: 12px; color: #6b7280; text-transform: uppercase; 
              letter-spacing: 2px; margin: 0;'>Account Overview</p>
    <h1 style='font-family: Syne, sans-serif; font-size: 36px; 
               font-weight: 800; margin: 4px 0; color: #e8eaf0;'>
        Welcome back, {user} 👋
    </h1>
    <p style='color: #6b7280; margin: 0;'>
        Here's your account summary for <strong style='color:{color};'>{card}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Key Metrics 
total_tx      = len(card_df)
fraud_flagged = int(card_df["model_decision"].sum())
actual_fraud  = int(card_df["isFraud"].sum())
total_spent   = card_df["TransactionAmt"].sum()
avg_tx        = card_df["TransactionAmt"].mean()
fraud_rate    = (fraud_flagged / total_tx * 100) if total_tx > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Total Transactions", f"{total_tx:,}",
                icon="💳", color=color)
with col2:
    metric_card("Fraud Alerts", f"{fraud_flagged:,}",
                delta=f"↑ {fraud_rate:.1f}% of transactions" if fraud_flagged > 0 else None,
                icon="🚨", color="#ef4444")
with col3:
    metric_card("Total Spent", f"${total_spent:,.0f}",
                delta=f"Avg ${avg_tx:.0f} per transaction",
                icon="💰", color="#22c55e")
with col4:
    # Model accuracy for this card
    correct = int(((card_df["model_decision"] == card_df["isFraud"])).sum())
    accuracy = correct / total_tx * 100 if total_tx > 0 else 0
    metric_card("Model Accuracy", f"{accuracy:.1f}%",
                icon="🎯", color="#a855f7")

st.markdown("<br>", unsafe_allow_html=True)

# Two column layout: Activity Chart + Recent Transactions 
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px; 
              font-weight: 700; margin-bottom: 12px;'>
        📈 Transaction Activity
    </p>
    """, unsafe_allow_html=True)

    # Monthly transaction volume
    card_df["month"] = card_df["transaction_date"].dt.to_period("M").astype(str)
    monthly = card_df.groupby("month").agg(
        total=("TransactionAmt", "sum"),
        count=("TransactionAmt", "count"),
        flagged=("model_decision", "sum")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["total"],
        name="Total Spent ($)",
        marker_color=color,
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["flagged"],
        name="Fraud Alerts",
        mode="lines+markers",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=6),
        yaxis="y2"
    ))
    fig.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        legend=dict(
            bgcolor="#13161e", bordercolor="#1e2330",
            orientation="h", y=1.12
        ),
        yaxis=dict(
            title="Amount ($)", gridcolor="#1e2330",
            color="#6b7280", showgrid=True
        ),
        yaxis2=dict(
            title="Fraud Alerts", overlaying="y",
            side="right", color="#ef4444", showgrid=False
        ),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px; 
              font-weight: 700; margin-bottom: 12px;'>
        🕐 Recent Transactions
    </p>
    """, unsafe_allow_html=True)

    recent = card_df.sort_values("transaction_date", ascending=False).head(8)
    for _, row in recent.iterrows():
        is_fraud = row["model_decision"] == 1
        flag     = "🔴" if is_fraud else "🟢"
        bg       = "#2a1515" if is_fraud else "#13161e"
        border   = "#ef4444" if is_fraud else "#1e2330"
        date_str = row["transaction_date"].strftime("%b %d, %H:%M")

        st.markdown(f"""
        <div style='background:{bg}; border:1px solid {border}; 
                    border-radius:8px; padding:10px 14px; 
                    margin-bottom:6px; display:flex; 
                    justify-content:space-between; align-items:center;'>
            <div>
                <span style='font-size:13px; color:#9ca3af;'>{date_str}</span>
                <span style='margin-left:8px; font-size:13px;'>{flag}</span>
            </div>
            <span style='font-size:14px; font-weight:600; 
                         color:{"#ef4444" if is_fraud else "#e8eaf0"};'>
                ${row["TransactionAmt"]:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

# Bottom Row: 3 Metrics + Charts
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px; 
              font-weight: 700; margin-bottom: 12px;'>
        🍩 Transaction Breakdown
    </p>
    """, unsafe_allow_html=True)

    legit   = total_tx - fraud_flagged
    fig_pie = go.Figure(go.Pie(
        labels=["Legitimate", "Fraud Flagged"],
        values=[legit, fraud_flagged],
        hole=0.65,
        marker=dict(colors=[color, "#ef4444"]),
        textfont=dict(color="#e8eaf0"),
        hovertemplate="%{label}: %{value:,}<extra></extra>"
    ))
    fig_pie.add_annotation(
        text=f"<b>{fraud_rate:.1f}%</b><br><span style='font-size:10px'>Fraud Rate</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#e8eaf0")
    )
    fig_pie.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        showlegend=True,
        legend=dict(bgcolor="#13161e", font=dict(color="#9ca3af")),
        margin=dict(l=0, r=0, t=10, b=0),
        height=220,
        font=dict(color="#9ca3af")
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with b2:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px; 
              font-weight: 700; margin-bottom: 12px;'>
        📦 Spending by Product
    </p>
    """, unsafe_allow_html=True)

    PRODUCT_MAP = {0.0: "C (Card)", 1.0: "H (Home)", 2.0: "R (Retail)",
                   3.0: "S (Service)", 4.0: "W (Web)"}
    card_df["product_label"] = card_df["ProductCD"].map(PRODUCT_MAP).fillna("Other")
    prod = card_df.groupby("product_label")["TransactionAmt"].sum().reset_index()
    prod.columns = ["Product", "Amount"]

    fig_bar = px.bar(
        prod.sort_values("Amount", ascending=True),
        x="Amount", y="Product", orientation="h",
        color_discrete_sequence=[color]
    )
    fig_bar.update_layout(
        paper_bgcolor="#13161e", plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        xaxis=dict(gridcolor="#1e2330", color="#6b7280"),
        yaxis=dict(color="#6b7280"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=220,
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with b3:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px; 
              font-weight: 700; margin-bottom: 12px;'>
        ⚡ Account Health
    </p>
    """, unsafe_allow_html=True)

    # Risk score gauge (average fraud probability)
    avg_risk = card_df["fraud_probability"].mean() * 100

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_risk,
        number=dict(suffix="%", font=dict(color="#e8eaf0", size=24)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#6b7280",
                      tickfont=dict(color="#6b7280")),
            bar=dict(color=color),
            bgcolor="#1e2330",
            steps=[
                dict(range=[0, 30],   color="rgba(34, 197, 94, 0.12)"),
                dict(range=[30, 70],  color="rgba(249, 115, 22, 0.12)"),
                dict(range=[70, 100], color="rgba(239, 68, 68, 0.12)"),
            ],
            threshold=dict(
                line=dict(color="#ef4444", width=2),
                thickness=0.75, value=70
            )
        ),
        title=dict(text="Avg Risk Score", font=dict(color="#9ca3af", size=13))
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#13161e",
        font=dict(color="#9ca3af"),
        margin=dict(l=20, r=20, t=30, b=0),
        height=220,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)