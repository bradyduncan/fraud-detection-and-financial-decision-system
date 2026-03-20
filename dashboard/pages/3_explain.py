# dashboard/pages/3_explain.py — Explainability Page (SHAP)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from dashboard.utils import (
    load_predictions, get_user_card_data,
    render_sidebar, metric_card, USER_COLORS
)

# Paths
BASE_DIR       = Path(__file__).resolve().parents[2]
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
SHAP_PATH      = PROCESSED_DIR / "shap_values.csv"
IMPORTANCE_PATH= PROCESSED_DIR / "shap_feature_importance.csv"

# Sidebar selection
user, card = render_sidebar()
color      = USER_COLORS.get(user, "#4f9cf9")

# Load Data with caching
@st.cache_data
def load_shap_data():
    shap_df       = pd.read_csv(SHAP_PATH)
    importance_df = pd.read_csv(IMPORTANCE_PATH)
    return shap_df, importance_df

df              = load_predictions()
card_df         = get_user_card_data(df, user, card).copy()
shap_df, importance_df = load_shap_data()

# Filter SHAP data for selected user/card
card_shap = shap_df[
    (shap_df["demo_user"] == user) &
    (shap_df["demo_card"] == card)
].copy()

# Feature columns only (exclude metadata)
meta_cols    = ["demo_user", "demo_card", "isFraud"] + \
               [c for c in card_shap.columns if c.startswith("val_")]
feature_cols = [c for c in card_shap.columns if c not in meta_cols]

# Page Header with custom styling
st.markdown(f"""
<div style='margin-bottom: 32px;'>
    <p style='font-size: 12px; color: #6b7280; text-transform: uppercase;
              letter-spacing: 2px; margin: 0;'>Model Explainability</p>
    <h1 style='font-family: Syne, sans-serif; font-size: 36px;
               font-weight: 800; margin: 4px 0; color: #e8eaf0;'>
        🔍 Why Did the Model Flag This?
    </h1>
    <p style='color: #6b7280; margin: 0;'>
        SHAP-based feature explanations —
        <strong style='color:{color};'>{user} · {card}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Top Metrics Cards
total_features  = len(feature_cols)
top_feature     = importance_df.iloc[0]["feature"]
top_importance  = importance_df.iloc[0]["importance"]
fraud_shap      = card_shap[card_shap["isFraud"] == 1]
legit_shap      = card_shap[card_shap["isFraud"] == 0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Features Analyzed", f"{total_features}",
                icon="🔬", color=color)
with c2:
    metric_card("Top Fraud Driver", str(top_feature),
                delta=f"Importance: {top_importance:.3f}",
                icon="⚡", color="#ef4444")
with c3:
    metric_card("Fraud Samples", f"{len(fraud_shap)}",
                delta="Transactions explained",
                icon="🚨", color="#f97316")
with c4:
    metric_card("Legit Samples", f"{len(legit_shap)}",
                delta="Transactions explained",
                icon="✅", color="#22c55e")

st.markdown("<br>", unsafe_allow_html=True)

# Global Feature Importance Chart
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    🌍 Global Feature Importance
    <span style='font-size: 12px; color: #6b7280; font-weight: 400;'>
     — Which features matter most overall?
    </span>
</p>
""", unsafe_allow_html=True)

top_n      = st.slider("Number of features to show", 5, 30, 15)
top_feats  = importance_df.head(top_n).sort_values("importance", ascending=True)

fig_imp = go.Figure(go.Bar(
    x=top_feats["importance"],
    y=top_feats["feature"],
    orientation="h",
    marker=dict(
        color=top_feats["importance"],
        colorscale=[[0, "#1e2330"], [1, color]],
    ),
    text=[f"{v:.3f}" for v in top_feats["importance"]],
    textposition="outside",
    textfont=dict(color="#9ca3af", size=11),
    hovertemplate="%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>"
))
fig_imp.update_layout(
    paper_bgcolor="#13161e",
    plot_bgcolor="#13161e",
    font=dict(color="#9ca3af", family="DM Sans"),
    xaxis=dict(color="#6b7280", gridcolor="#1e2330",
               title="Mean |SHAP Value| (Average Impact on Fraud Probability)"),
    yaxis=dict(color="#6b7280"),
    margin=dict(l=0, r=60, t=10, b=0),
    height=max(300, top_n * 22),
    showlegend=False,
)
st.plotly_chart(fig_imp, use_container_width=True)

# Explanation box
st.markdown(f"""
<div style='background:#13161e; border:1px solid #1e2330;
            border-left:3px solid {color};
            border-radius:10px; padding:16px; margin-bottom:24px;'>
    <p style='margin:0 0 6px 0; font-size:14px; font-weight:600;'>
        📖 How to read this chart
    </p>
    <p style='margin:0; font-size:13px; color:#9ca3af; line-height:1.6;'>
        Each bar shows the <strong>average absolute SHAP value</strong> for that feature 
        across all {len(card_shap)} sampled transactions. A longer bar means the feature 
        had a bigger impact on pushing the fraud probability up or down. 
        <strong style='color:{color};'>{top_feature}</strong> is the most influential 
        feature for this card.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# SHAP Comparison: Fraud vs Legit
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    ⚖️ Fraud vs Legitimate — Feature Impact Comparison
    <span style='font-size: 12px; color: #6b7280; font-weight: 400;'>
     — How do features differ between fraud and legit transactions?
    </span>
</p>
""", unsafe_allow_html=True)

# Top 10 features mean SHAP for fraud vs legit
top10_features = importance_df.head(10)["feature"].tolist()
top10_features = [f for f in top10_features if f in feature_cols]

fraud_means = card_shap[card_shap["isFraud"] == 1][top10_features].mean()
legit_means = card_shap[card_shap["isFraud"] == 0][top10_features].mean()

fig_compare = go.Figure()
fig_compare.add_trace(go.Bar(
    name="Fraud Transactions",
    x=top10_features,
    y=fraud_means.values,
    marker_color="#ef4444",
    opacity=0.8,
))
fig_compare.add_trace(go.Bar(
    name="Legitimate Transactions",
    x=top10_features,
    y=legit_means.values,
    marker_color=color,
    opacity=0.8,
))
fig_compare.update_layout(
    barmode="group",
    paper_bgcolor="#13161e",
    plot_bgcolor="#13161e",
    font=dict(color="#9ca3af", family="DM Sans"),
    legend=dict(bgcolor="#13161e", bordercolor="#1e2330",
                orientation="h", y=1.08),
    xaxis=dict(color="#6b7280", gridcolor="#1e2330",
               title="Feature", tickangle=-30),
    yaxis=dict(color="#6b7280", gridcolor="#1e2330",
               title="Mean SHAP Value"),
    margin=dict(l=0, r=0, t=40, b=60),
    height=350,
    hovermode="x unified",
)
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Transaction-Level Explanation 
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    🔎 Transaction-Level Explanation
    <span style='font-size: 12px; color: #6b7280; font-weight: 400;'>
     — Why was a specific transaction flagged?
    </span>
</p>
""", unsafe_allow_html=True)

# Transaction selector
tx_filter = st.selectbox(
    "Select transaction type to explain",
    ["Fraud transactions (actual)", "Legitimate transactions",
     "False Positives (wrongly flagged)", "False Negatives (missed fraud)"]
)

# Get matching transactions from predictions
pred_card = get_user_card_data(df, user, card).copy()
pred_card["outcome"] = "True Negative"
pred_card.loc[(pred_card["isFraud"]==1) & (pred_card["model_decision"]==1), "outcome"] = "True Positive"
pred_card.loc[(pred_card["isFraud"]==1) & (pred_card["model_decision"]==0), "outcome"] = "False Negative"
pred_card.loc[(pred_card["isFraud"]==0) & (pred_card["model_decision"]==1), "outcome"] = "False Positive"

if tx_filter == "Fraud transactions (actual)":
    subset_shap = card_shap[card_shap["isFraud"] == 1]
elif tx_filter == "Legitimate transactions":
    subset_shap = card_shap[card_shap["isFraud"] == 0]
elif tx_filter == "False Positives (wrongly flagged)":
    fp_card1 = pred_card[pred_card["outcome"] == "False Positive"]["card1"].values
    subset_shap = card_shap[card_shap["isFraud"] == 0].head(20)
else:
    subset_shap = card_shap[card_shap["isFraud"] == 1].head(20)

if len(subset_shap) == 0:
    st.info("No transactions found for this filter.")
else:
    # Pick a random transaction to explain
    tx_idx = st.slider(
        "Select transaction index",
        0, min(len(subset_shap)-1, 19), 0
    )
    tx_row = subset_shap.iloc[tx_idx]

    # Get top contributing features for this transaction
    tx_shap = tx_row[feature_cols].astype(float)
    tx_top  = tx_shap.abs().nlargest(12)
    tx_vals = tx_shap[tx_top.index]

    # Waterfall chart
    colors_wf = ["#ef4444" if v > 0 else "#22c55e" for v in tx_vals.values]

    fig_wf = go.Figure(go.Bar(
        x=tx_vals.index,
        y=tx_vals.values,
        marker_color=colors_wf,
        opacity=0.85,
        text=[f"{v:+.3f}" for v in tx_vals.values],
        textposition="outside",
        textfont=dict(color="#9ca3af", size=10),
        hovertemplate="%{x}<br>SHAP: %{y:+.4f}<extra></extra>"
    ))
    fig_wf.add_hline(y=0, line_color="#4b5563", line_width=1)
    fig_wf.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        xaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="Feature", tickangle=-30),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title="SHAP Value (↑ pushes toward fraud, ↓ pushes toward legit)"),
        margin=dict(l=0, r=0, t=20, b=60),
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Plain English explanation
    is_fraud    = int(tx_row["isFraud"])
    top_push_up = tx_vals[tx_vals > 0].nlargest(3)
    top_push_dn = tx_vals[tx_vals < 0].nsmallest(3)

    fraud_label = "🔴 Actual Fraud" if is_fraud else "🟢 Legitimate"

    st.markdown(f"""
    <div style='background:#13161e; border:1px solid #1e2330;
                border-radius:10px; padding:18px; margin-top:8px;'>
        <p style='margin:0 0 10px 0; font-size:14px; font-weight:700;'>
            Transaction #{tx_idx + 1} — {fraud_label}
        </p>
        <p style='margin:0 0 8px 0; font-size:13px; color:#9ca3af;'>
            <strong style='color:#ef4444;'>Features pushing TOWARD fraud (red ↑):</strong>
        </p>
        {"".join([f"<p style='margin:2px 0 2px 16px; font-size:13px; color:#ef4444;'>• {feat}: +{val:.3f}</p>" for feat, val in top_push_up.items()])}
        <p style='margin:10px 0 8px 0; font-size:13px; color:#9ca3af;'>
            <strong style='color:#22c55e;'>Features pushing TOWARD legit (green ↓):</strong>
        </p>
        {"".join([f"<p style='margin:2px 0 2px 16px; font-size:13px; color:#22c55e;'>• {feat}: {val:.3f}</p>" for feat, val in top_push_dn.items()])}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# SHAP Summary: Fraud Rate by Top Feature
st.markdown("""
<p style='font-family: Syne, sans-serif; font-size: 16px;
          font-weight: 700; margin-bottom: 12px;'>
    📊 SHAP Distribution — Fraud vs Legit
    <span style='font-size: 12px; color: #6b7280; font-weight: 400;'>
     — How SHAP values are spread for top features
    </span>
</p>
""", unsafe_allow_html=True)

selected_feature = st.selectbox(
    "Select feature to inspect",
    options=importance_df.head(15)["feature"].tolist()
)

if selected_feature in feature_cols:
    fig_violin = go.Figure()
    fig_violin.add_trace(go.Violin(
        x=["Fraud"] * len(fraud_shap),
        y=fraud_shap[selected_feature].astype(float),
        name="Fraud",
        fillcolor="rgba(239,68,68,0.3)",
        line_color="#ef4444",
        box_visible=True,
        meanline_visible=True,
    ))
    fig_violin.add_trace(go.Violin(
        x=["Legitimate"] * len(legit_shap),
        y=legit_shap[selected_feature].astype(float),
        name="Legitimate",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}",
        line_color=color,
        box_visible=True,
        meanline_visible=True,
    ))
    fig_violin.update_layout(
        paper_bgcolor="#13161e",
        plot_bgcolor="#13161e",
        font=dict(color="#9ca3af", family="DM Sans"),
        xaxis=dict(color="#6b7280"),
        yaxis=dict(color="#6b7280", gridcolor="#1e2330",
                   title=f"SHAP Value for {selected_feature}"),
        legend=dict(bgcolor="#13161e", bordercolor="#1e2330"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
        violingap=0.3,
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    # Explanation
    fraud_mean_shap = fraud_shap[selected_feature].astype(float).mean()
    legit_mean_shap = legit_shap[selected_feature].astype(float).mean()
    direction = "higher" if fraud_mean_shap > legit_mean_shap else "lower"
    impact    = "pushes toward fraud" if fraud_mean_shap > legit_mean_shap \
                else "pushes toward legitimate"

    st.markdown(f"""
    <div style='background:#13161e; border:1px solid #1e2330;
                border-left:3px solid {color};
                border-radius:10px; padding:14px; margin-top:8px;'>
        <p style='margin:0; font-size:13px; color:#9ca3af; line-height:1.6;'>
            For <strong style='color:{color};'>{selected_feature}</strong>: 
            fraud transactions have a <strong>{direction}</strong> mean SHAP value 
            ({fraud_mean_shap:+.3f}) compared to legitimate transactions 
            ({legit_mean_shap:+.3f}). This means higher values of this feature 
            tend to <strong>{impact}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)