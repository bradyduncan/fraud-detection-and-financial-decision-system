# dashboard/pages/4_spending.py — Spending Patterns Page

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from dashboard.utils import (
    load_predictions, get_user_card_data,
    render_sidebar, metric_card, USER_COLORS
)

# ── Sidebar ────────────────────────────────────────────────────────────────
user, card = render_sidebar()

# ── Load data ──────────────────────────────────────────────────────────────
df      = load_predictions()
card_df = get_user_card_data(df, user, card).copy()
color   = USER_COLORS.get(user, "#4f9cf9")

# ── Derived columns ────────────────────────────────────────────────────────
card_df["transaction_date"] = pd.to_datetime(card_df["transaction_date"])
card_df["month"]      = card_df["transaction_date"].dt.to_period("M").astype(str)
card_df["week"]       = card_df["transaction_date"].dt.to_period("W").astype(str)
card_df["hour"]       = card_df["transaction_date"].dt.hour
card_df["dayofweek"]  = card_df["transaction_date"].dt.day_name()

PRODUCT_MAP = {
    0.0: "C — Card Not Present",
    1.0: "H — Home",
    2.0: "R — Retail",
    3.0: "S — Service",
    4.0: "W — Web"
}
card_df["product_label"] = card_df["ProductCD"].map(PRODUCT_MAP).fillna("Other")

# ── Page Header ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 32px;'>
    <p style='font-size: 12px; color: #6b7280; text-transform: uppercase;
              letter-spacing: 2px; margin: 0;'>Financial Insights</p>
    <h1 style='font-family: Syne, sans-serif; font-size: 36px;
               font-weight: 800; margin: 4px 0; color: #e8eaf0;'>
        📊 Spending Patterns
    </h1>
    <p style='color: #6b7280; margin: 0;'>
        Transaction breakdown and spending behaviour —
        <strong style='color:{color};'>{user} · {card}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ────────────────────────────────────────────────────────────
total_spent   = card_df["TransactionAmt"].sum()
avg_tx        = card_df["TransactionAmt"].mean()
max_tx        = card_df["TransactionAmt"].max()
total_tx      = len(card_df)
legit_df      = card_df[card_df["isFraud"] == 0]
avg_legit     = legit_df["TransactionAmt"].mean()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Spent", f"${total_spent:,.0f}",
                icon="💰", color=color)
with c2:
    metric_card("Avg Transaction", f"${avg_tx:.0f}",
                delta=f"Legit avg: ${avg_legit:.0f}",
                icon="📊", color="#22c55e")
with c3:
    metric_card("Largest Transaction", f"${max_tx:,.0f}",
                icon="📈", color="#f97316")
with c4:
    metric_card("Total Transactions", f"{total_tx:,}",
                icon="💳", color="#a855f7")

st.markdown("<br>", unsafe_allow_html=True)

# ── Monthly Spending Trend ─────────────────────────────────────────────────
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    📅 Monthly Spending Trend
</p>
""", unsafe_allow_html=True)

monthly = card_df.groupby("month").agg(
    total_spent=("TransactionAmt", "sum"),
    avg_spent=("TransactionAmt", "mean"),
    tx_count=("TransactionAmt", "count"),
    fraud_amt=("TransactionAmt", lambda x: x[card_df.loc[x.index, "isFraud"] == 1].sum())
).reset_index()

fig_monthly = go.Figure()
fig_monthly.add_trace(go.Bar(
    x=monthly["month"],
    y=monthly["total_spent"],
    name="Total Spent ($)",
    marker_color=color,
    opacity=0.8,
))
fig_monthly.add_trace(go.Scatter(
    x=monthly["month"],
    y=monthly["avg_spent"],
    name="Avg Transaction ($)",
    mode="lines+markers",
    line=dict(color="#f97316", width=2),
    marker=dict(size=6),
    yaxis="y2"
))
fig_monthly.update_layout(
    paper_bgcolor="#13161e",
    plot_bgcolor="#13161e",
    font=dict(color="#9ca3af", family="DM Sans"),
    legend=dict(bgcolor="#13161e", bordercolor="#1e2330",
                orientation="h", y=1.08),
    yaxis=dict(title="Total Spent ($)", gridcolor="#1e2330", color="#6b7280"),
    yaxis2=dict(title="Avg Transaction ($)", overlaying="y",
                side="right", color="#f97316", showgrid=False),
    xaxis=dict(color="#6b7280", gridcolor="#1e2330"),
    margin=dict(l=0, r=0, t=40, b=0),
    height=300,
    hovermode="x unified",
)
st.plotly_chart(fig_monthly, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two column: Product breakdown + Day of week ───────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        🛍️ Spending by Category
    </p>
    """, unsafe_allow_html=True)

    prod = card_df.groupby("product_label").agg(
        total=("TransactionAmt", "sum"),
        count=("TransactionAmt", "count"),
        avg=("TransactionAmt", "mean")
    ).reset_index().sort_values("total", ascending=True)

    fig_prod = go.Figure()
    fig_prod.add_trace(go.Bar(
        x=prod["total"],
        y=prod["product_label"],
        orientation="h",
        marker=dict(
            color=prod["total"],
            colorscale=[[0, "#1e2330"], [1, color]],
        ),
        text=[f"${v:,.0f}" for v in prod["total"]],
        textposition="outside",
        textfont=dict(color="#9ca3af", size=11),
        hovertemplate="%{y}<br>Total: $%{x:,.0f}<br>Transactions: %{customdata}<extra></extra>",
        customdata=prod["count"],
    ))
    fig_prod.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330", title="Total Spent ($)"),
        yaxis=dict(color="#6b7280"),
        margin=dict(l=0, r=60, t=10, b=0),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_prod, use_container_width=True)

with right:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        📆 Spending by Day of Week
    </p>
    """, unsafe_allow_html=True)

    day_order = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]
    dow = card_df.groupby("dayofweek").agg(
        total=("TransactionAmt", "sum"),
        count=("TransactionAmt", "count")
    ).reindex(day_order).fillna(0).reset_index()

    fig_dow = go.Figure(go.Bar(
        x=dow["dayofweek"],
        y=dow["total"],
        marker=dict(
            color=dow["total"],
            colorscale=[[0, "#1e2330"], [1, color]],
        ),
        text=[f"${v:,.0f}" for v in dow["total"]],
        textposition="outside",
        textfont=dict(color="#9ca3af", size=10),
        hovertemplate="%{x}<br>Total: $%{y:,.0f}<extra></extra>",
    ))
    fig_dow.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330"),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330", title="Total Spent ($)"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_dow, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two column: Hour heatmap + Transaction size distribution ──────────────
left2, right2 = st.columns([1, 1], gap="large")

with left2:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        🕐 Spending by Hour of Day
    </p>
    """, unsafe_allow_html=True)

    hourly = card_df.groupby("hour").agg(
        total=("TransactionAmt", "sum"),
        count=("TransactionAmt", "count")
    ).reset_index()

    fig_hour = go.Figure(go.Bar(
        x=hourly["hour"],
        y=hourly["total"],
        marker=dict(
            color=hourly["total"],
            colorscale=[[0, "#1e2330"], [1, color]],
        ),
        hovertemplate="Hour %{x}:00<br>Total: $%{y:,.0f}<extra></extra>",
    ))
    fig_hour.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="Hour of Day", tickmode="linear", dtick=2),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330", title="Total Spent ($)"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        showlegend=False,
    )
    st.plotly_chart(fig_hour, use_container_width=True)

with right2:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        💵 Transaction Size Distribution
    </p>
    """, unsafe_allow_html=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=card_df[card_df["isFraud"] == 0]["TransactionAmt"],
        name="Legitimate",
        marker_color=color,
        opacity=0.7,
        nbinsx=50,
    ))
    fig_dist.add_trace(go.Histogram(
        x=card_df[card_df["isFraud"] == 1]["TransactionAmt"],
        name="Fraudulent",
        marker_color="#ef4444",
        opacity=0.7,
        nbinsx=50,
    ))
    fig_dist.update_layout(
        barmode="overlay",
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        legend=dict(bgcolor="#13161e", bordercolor="#1e2330",
                    orientation="h", y=1.12),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="Transaction Amount ($)"),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330", title="Count"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=260,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Spending Insights Summary ──────────────────────────────────────────────
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    💡 Spending Insights
</p>
""", unsafe_allow_html=True)

# Calculate insights
top_category  = str(card_df.groupby("product_label")["TransactionAmt"].sum().idxmax())
top_day       = str(card_df.groupby("dayofweek")["TransactionAmt"].sum().idxmax())
peak_hour     = int(card_df.groupby("hour")["TransactionAmt"].sum().idxmax())
fraud_amt_avg = card_df[card_df["isFraud"] == 1]["TransactionAmt"].mean()
legit_amt_avg = card_df[card_df["isFraud"] == 0]["TransactionAmt"].mean()
amt_diff      = ((fraud_amt_avg - legit_amt_avg) / legit_amt_avg * 100) if legit_amt_avg > 0 else 0

i1, i2, i3, i4 = st.columns(4)
with i1:
    st.markdown(f"""
    <div style='background:#13161e; border:1px solid #1e2330;
                border-left:3px solid {color};
                border-radius:10px; padding:16px;'>
        <p style='margin:0; font-size:11px; color:#6b7280; 
                  text-transform:uppercase; letter-spacing:1px;'>
            Top Category
        </p>
        <p style='margin:6px 0 0 0; font-size:15px; font-weight:600;'>
            {top_category.split("—")[0].strip()}
        </p>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div style='background:#13161e; border:1px solid #1e2330;
                border-left:3px solid {color};
                border-radius:10px; padding:16px;'>
        <p style='margin:0; font-size:11px; color:#6b7280;
                  text-transform:uppercase; letter-spacing:1px;'>
            Busiest Day
        </p>
        <p style='margin:6px 0 0 0; font-size:15px; font-weight:600;'>
            {top_day}
        </p>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div style='background:#13161e; border:1px solid #1e2330;
                border-left:3px solid {color};
                border-radius:10px; padding:16px;'>
        <p style='margin:0; font-size:11px; color:#6b7280;
                  text-transform:uppercase; letter-spacing:1px;'>
            Peak Hour
        </p>
        <p style='margin:6px 0 0 0; font-size:15px; font-weight:600;'>
            {peak_hour}:00 — {peak_hour+1}:00
        </p>
    </div>
    """, unsafe_allow_html=True)

with i4:
    direction = "higher" if amt_diff > 0 else "lower"
    insight_color = "#ef4444" if amt_diff > 0 else "#22c55e"
    st.markdown(f"""
    <div style='background:#13161e; border:1px solid #1e2330;
                border-left:3px solid {insight_color};
                border-radius:10px; padding:16px;'>
        <p style='margin:0; font-size:11px; color:#6b7280;
                  text-transform:uppercase; letter-spacing:1px;'>
            Fraud vs Legit Amount
        </p>
        <p style='margin:6px 0 0 0; font-size:15px; font-weight:600;
                  color:{insight_color};'>
            {abs(amt_diff):.1f}% {direction}
        </p>
    </div>
    """, unsafe_allow_html=True)