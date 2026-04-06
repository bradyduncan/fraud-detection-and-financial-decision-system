# dashboard/utils.py — Shared data loader and helpers for all pages
# Place at: dashboard/utils.py

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = BASE_DIR / "data" / "processed" / "predictions.csv"
MODELS_DIR       = BASE_DIR / "data" / "processed" / "models"

# Label decoders (label encoding reversal for display) 
CARD4_MAP = {1.0: "American Express", 2.0: "Discover", 3.0: "Mastercard", 4.0: "Visa"}
CARD6_MAP = {1.0: "Charge Card", 2.0: "Credit", 3.0: "Debit", 4.0: "Debit or Credit"}

# Demo user definitions
DEMO_USERS = {
    "Alice": {"Card 1": 7919,  "Card 2": 15066},
    "Bob":   {"Card 1": 9500,  "Card 2": 6019},
    "Carol": {"Card 1": 15885, "Card 2": 7585},
    "Dave":  {"Card 1": 9633,  "Card 2": 12695},
}

# User avatar colors for UI
USER_COLORS = {
    "Alice": "#4f9cf9",
    "Bob":   "#f97316",
    "Carol": "#a855f7",
    "Dave":  "#22c55e",
}

@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDICTIONS_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Decode label-encoded card columns for display
    df["card4_label"] = df["card4"].map(CARD4_MAP).fillna("Unknown")
    df["card6_label"] = df["card6"].map(CARD6_MAP).fillna("Unknown")

    return df


def get_user_card_data(df, user: str, card: str) -> pd.DataFrame:
    card1_val = DEMO_USERS[user][card]
    return df[df["card1"] == card1_val].copy()


def render_sidebar() -> tuple[str, str]:
    st.sidebar.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <p style='font-family: Syne, sans-serif; font-size: 22px; 
                  font-weight: 800; margin: 0; color: #e8eaf0;'>
            🛡️ FraudGuard
        </p>
        <p style='font-size: 11px; color: #6b7280; margin: 2px 0 0 0;'>
            Fraud Detection & Credit Tracker
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 👤 Select Account")
    user = st.sidebar.selectbox(
        "Demo User",
        options=list(DEMO_USERS.keys()),
        key="selected_user",
        label_visibility="collapsed"
    )

    st.sidebar.markdown("### 💳 Select Card")
    card = st.sidebar.selectbox(
        "Card",
        options=list(DEMO_USERS[user].keys()),
        key="selected_card",
        label_visibility="collapsed"
    )

    # Show card details
    df = load_predictions()
    card_data = get_user_card_data(df, user, card)
    if len(card_data) > 0:
        card4 = card_data["card4_label"].iloc[0]
        card6 = card_data["card6_label"].iloc[0]
        color = USER_COLORS.get(user, "#4f9cf9")
        st.sidebar.markdown(f"""
        <div style='background: #1a1d27; border: 1px solid #1e2330; 
                    border-left: 3px solid {color};
                    border-radius: 8px; padding: 12px; margin-top: 8px;'>
            <p style='margin: 0; font-size: 13px; color: #9ca3af;'>Card Type</p>
            <p style='margin: 2px 0 8px 0; font-size: 14px; font-weight: 600;'>{card4} {card6}</p>
            <p style='margin: 0; font-size: 13px; color: #9ca3af;'>Transactions</p>
            <p style='margin: 2px 0 0 0; font-size: 14px; font-weight: 600;'>{len(card_data):,}</p>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='font-size: 11px; color: #4b5563; text-align: center;'>"
        "Powered by LightGBM Ensemble<br>IEEE-CIS Fraud Dataset</p>",
        unsafe_allow_html=True
    )

    return user, card


def metric_card(label: str, value: str, delta: str | None = None,
                color: str = "#4f9cf9", icon: str = ""):
    delta_html = ""
    if delta:
        delta_color = "#ef4444" if "↑" in delta else "#22c55e"
        delta_html = f"<p style='margin:4px 0 0 0; font-size:12px; color:{delta_color};'>{delta}</p>"

    st.markdown(f"""
    <div style='background: #13161e; border: 1px solid #1e2330;
                border-top: 3px solid {color};
                border-radius: 12px; padding: 20px;'>
        <p style='margin: 0; font-size: 12px; color: #6b7280; 
                  text-transform: uppercase; letter-spacing: 1px;'>{icon} {label}</p>
        <p style='margin: 6px 0 0 0; font-size: 28px; font-weight: 700;
                  font-family: Syne, sans-serif; color: #e8eaf0;'>{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def fraud_badge(probability: float) -> str:
    if probability >= 0.7:
        return "<span style='background:#ef444420; color:#ef4444; padding:2px 8px; border-radius:4px; font-size:12px;'>🔴 High Risk</span>"
    elif probability >= 0.3:
        return "<span style='background:#f9731620; color:#f97316; padding:2px 8px; border-radius:4px; font-size:12px;'>🟠 Medium Risk</span>"
    else:
        return "<span style='background:#22c55e20; color:#22c55e; padding:2px 8px; border-radius:4px; font-size:12px;'>🟢 Safe</span>"










