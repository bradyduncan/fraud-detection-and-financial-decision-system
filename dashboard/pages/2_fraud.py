# dashboard/pages/2_fraud.py — Fraud Alerts Page
# Shows model predictions vs actual fraud labels

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from dashboard.utils import (
    load_predictions, get_user_card_data,
    render_sidebar, metric_card, fraud_badge, USER_COLORS
)

# ── Sidebar ────────────────────────────────────────────────────────────────
user, card = render_sidebar()

# ── Load data ──────────────────────────────────────────────────────────────
df      = load_predictions()
card_df = get_user_card_data(df, user, card).copy()
color   = USER_COLORS.get(user, "#4f9cf9")

# ── Derived columns ────────────────────────────────────────────────────────
# Classification outcomes
card_df["outcome"] = "True Negative"  # default — legit, predicted legit
card_df.loc[(card_df["isFraud"] == 1) & (card_df["model_decision"] == 1), "outcome"] = "True Positive"
card_df.loc[(card_df["isFraud"] == 1) & (card_df["model_decision"] == 0), "outcome"] = "False Negative"
card_df.loc[(card_df["isFraud"] == 0) & (card_df["model_decision"] == 1), "outcome"] = "False Positive"

# ── Confusion matrix values ────────────────────────────────────────────────
tp = int((card_df["outcome"] == "True Positive").sum())
tn = int((card_df["outcome"] == "True Negative").sum())
fp = int((card_df["outcome"] == "False Positive").sum())
fn = int((card_df["outcome"] == "False Negative").sum())
total = len(card_df)

recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0

# ── Page Header ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 32px;'>
    <p style='font-size: 12px; color: #6b7280; text-transform: uppercase;
              letter-spacing: 2px; margin: 0;'>Fraud Detection</p>
    <h1 style='font-family: Syne, sans-serif; font-size: 36px;
               font-weight: 800; margin: 4px 0; color: #e8eaf0;'>
        🚨 Fraud Alerts
    </h1>
    <p style='color: #6b7280; margin: 0;'>
        Model predictions vs actual fraud — 
        <strong style='color:{color};'>{user} · {card}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Fraud Caught (Recall)",
                f"{recall*100:.1f}%",
                delta=f"↑ {tp} of {tp+fn} actual fraud cases",
                icon="✅", color="#22c55e")
with c2:
    metric_card("False Alarms (FPR)",
                f"{fpr*100:.1f}%",
                delta=f"↑ {fp} legitimate flagged as fraud" if fp > 0 else "0 false alarms",
                icon="⚠️", color="#f97316")
with c3:
    metric_card("Precision",
                f"{precision*100:.1f}%",
                delta=f"Of flagged, {precision*100:.1f}% were real fraud",
                icon="🎯", color=color)
with c4:
    metric_card("F1 Score",
                f"{f1*100:.1f}%",
                icon="📊", color="#a855f7")

st.markdown("<br>", unsafe_allow_html=True)

# ── Two column: Confusion Matrix + Outcome Breakdown ──────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        🔲 Confusion Matrix
    </p>
    """, unsafe_allow_html=True)

    # Styled confusion matrix using plotly heatmap
    z     = [[tn, fp], [fn, tp]]
    x_lbl = ["Predicted: Legit", "Predicted: Fraud"]
    y_lbl = ["Actual: Legit", "Actual: Fraud"]

    annotations = []
    labels = [["True Negative", "False Positive"],
              ["False Negative", "True Positive"]]
    colors_ann = [["#22c55e", "#ef4444"], ["#f97316", "#22c55e"]]

    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=x_lbl[j], y=y_lbl[i],
                text=f"<b>{z[i][j]:,}</b><br><span style='font-size:10px'>{labels[i][j]}</span>",
                showarrow=False,
                font=dict(color=colors_ann[i][j], size=14)
            ))

    fig_cm = go.Figure(go.Heatmap(
        z=z, x=x_lbl, y=y_lbl,
        colorscale=[[0, "#13161e"], [1, "#1e2330"]],
        showscale=False,
        hovertemplate="%{x} / %{y}: %{z:,}<extra></extra>"
    ))
    fig_cm.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        annotations=annotations,
        xaxis=dict(color="#6b7280", side="bottom"),
        yaxis=dict(color="#6b7280"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Plain English explanation
    st.markdown(f"""
    <div style='background:#13161e; border:1px solid #1e2330;
                border-radius:10px; padding:16px; font-size:13px;'>
        <p style='margin:0 0 8px 0; color:#9ca3af;'>
            <span style='color:#22c55e;'>✅ True Positives ({tp:,})</span> — 
            Fraud correctly caught by model
        </p>
        <p style='margin:0 0 8px 0; color:#9ca3af;'>
            <span style='color:#22c55e;'>✅ True Negatives ({tn:,})</span> — 
            Legitimate transactions correctly cleared
        </p>
        <p style='margin:0 0 8px 0; color:#9ca3af;'>
            <span style='color:#f97316;'>⚠️ False Positives ({fp:,})</span> — 
            Legitimate transactions wrongly flagged
        </p>
        <p style='margin:0; color:#9ca3af;'>
            <span style='color:#ef4444;'>❌ False Negatives ({fn:,})</span> — 
            Fraud cases the model missed
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 16px;
              font-weight: 700; margin-bottom: 12px;'>
        📊 Prediction vs Actual Over Time
    </p>
    """, unsafe_allow_html=True)

    # Monthly actual fraud vs model flagged
    card_df["month"] = card_df["transaction_date"].dt.to_period("M").astype(str)
    monthly = card_df.groupby("month").agg(
        actual_fraud=("isFraud", "sum"),
        model_flagged=("model_decision", "sum")
    ).reset_index()

    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        x=monthly["month"], y=monthly["actual_fraud"],
        name="Actual Fraud",
        marker_color="#ef4444",
        opacity=0.7,
    ))
    fig_time.add_trace(go.Bar(
        x=monthly["month"], y=monthly["model_flagged"],
        name="Model Flagged",
        marker_color=color,
        opacity=0.7,
    ))
    fig_time.update_layout(
        barmode="group",
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        legend=dict(bgcolor="#13161e", bordercolor="#1e2330",
                    orientation="h", y=1.12),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330"),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="Transaction Count"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Fraud probability distribution
    st.markdown("""
    <p style='font-family: Syne, sans-serif; font-size: 14px;
              font-weight: 600; margin: 16px 0 8px 0;'>
        Fraud Probability Distribution
    </p>
    """, unsafe_allow_html=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=card_df[card_df["isFraud"] == 0]["fraud_probability"],
        name="Actual Legit",
        marker_color="#22c55e",
        opacity=0.6,
        nbinsx=40,
    ))
    fig_hist.add_trace(go.Histogram(
        x=card_df[card_df["isFraud"] == 1]["fraud_probability"],
        name="Actual Fraud",
        marker_color="#ef4444",
        opacity=0.6,
        nbinsx=40,
    ))
    fig_hist.update_layout(
        barmode="overlay",
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        legend=dict(bgcolor="#13161e", bordercolor="#1e2330",
                    orientation="h", y=1.15),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="Fraud Probability Score"),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330", title="Count"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=200,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Transaction Table ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    🔎 Transaction Detail View
</p>
""", unsafe_allow_html=True)

# Filters row
f1, f2, f3 = st.columns(3)
with f1:
    view_filter = st.selectbox(
        "Show transactions",
        ["All Transactions", "Fraud Flagged by Model",
         "Actual Fraud", "False Positives (Wrongly Flagged)",
         "False Negatives (Missed Fraud)"],
    )
with f2:
    sort_by = st.selectbox("Sort by", ["Date (newest)", "Amount (highest)", "Risk Score (highest)"])
with f3:
    n_rows = st.slider("Rows to show", 10, 100, 25)

# Apply filter
filtered = card_df.copy()
if view_filter == "Fraud Flagged by Model":
    filtered = filtered[filtered["model_decision"] == 1]
elif view_filter == "Actual Fraud":
    filtered = filtered[filtered["isFraud"] == 1]
elif view_filter == "False Positives (Wrongly Flagged)":
    filtered = filtered[filtered["outcome"] == "False Positive"]
elif view_filter == "False Negatives (Missed Fraud)":
    filtered = filtered[filtered["outcome"] == "False Negative"]

# Apply sort
if sort_by == "Date (newest)":
    filtered = filtered.sort_values("transaction_date", ascending=False)
elif sort_by == "Amount (highest)":
    filtered = filtered.sort_values("TransactionAmt", ascending=False)
else:
    filtered = filtered.sort_values("fraud_probability", ascending=False)

filtered = filtered.head(n_rows)

# Render transaction rows
for _, row in filtered.iterrows():
    outcome  = row["outcome"]
    is_model_fraud = row["model_decision"] == 1
    is_actual_fraud = row["isFraud"] == 1

    # Color coding by outcome
    if outcome == "True Positive":
        bg, border, badge_color = "#1a2a1a", "#22c55e", "#22c55e"
        badge_text = "✅ True Positive"
    elif outcome == "False Positive":
        bg, border, badge_color = "#2a2015", "#f97316", "#f97316"
        badge_text = "⚠️ False Positive"
    elif outcome == "False Negative":
        bg, border, badge_color = "#2a1515", "#ef4444", "#ef4444"
        badge_text = "❌ False Negative"
    else:
        bg, border, badge_color = "#13161e", "#1e2330", "#6b7280"
        badge_text = "✓ True Negative"

    prob_pct  = f"{row['fraud_probability']*100:.1f}%"
    date_str  = row["transaction_date"].strftime("%b %d, %Y %H:%M")
    amt       = f"${row['TransactionAmt']:.2f}"

    st.markdown(f"""
    <div style='background:{bg}; border:1px solid {border};
                border-radius:10px; padding:14px 18px;
                margin-bottom:6px;
                display:flex; justify-content:space-between; align-items:center;'>
        <div style='display:flex; gap:24px; align-items:center;'>
            <div>
                <p style='margin:0; font-size:11px; color:#6b7280;'>Date</p>
                <p style='margin:0; font-size:13px;'>{date_str}</p>
            </div>
            <div>
                <p style='margin:0; font-size:11px; color:#6b7280;'>Amount</p>
                <p style='margin:0; font-size:14px; font-weight:600;'>{amt}</p>
            </div>
            <div>
                <p style='margin:0; font-size:11px; color:#6b7280;'>Risk Score</p>
                <p style='margin:0; font-size:13px; color:{badge_color};
                          font-weight:600;'>{prob_pct}</p>
            </div>
            <div>
                <p style='margin:0; font-size:11px; color:#6b7280;'>Actual</p>
                <p style='margin:0; font-size:13px;'>
                    {"🔴 Fraud" if is_actual_fraud else "🟢 Legit"}
                </p>
            </div>
            <div>
                <p style='margin:0; font-size:11px; color:#6b7280;'>Model Said</p>
                <p style='margin:0; font-size:13px;'>
                    {"🔴 Fraud" if is_model_fraud else "🟢 Legit"}
                </p>
            </div>
        </div>
        <span style='background:{badge_color}20; color:{badge_color};
                     padding:4px 12px; border-radius:6px;
                     font-size:12px; font-weight:600; white-space:nowrap;'>
            {badge_text}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Summary footer
st.markdown(f"""
<div style='background:#13161e; border:1px solid #1e2330;
            border-radius:10px; padding:14px 18px; margin-top:12px;
            display:flex; gap:32px;'>
    <span style='font-size:13px; color:#6b7280;'>
        Showing <strong style='color:#e8eaf0;'>{len(filtered):,}</strong> transactions
    </span>
    <span style='font-size:13px; color:#6b7280;'>
        Filter: <strong style='color:{color};'>{view_filter}</strong>
    </span>
</div>
""", unsafe_allow_html=True)