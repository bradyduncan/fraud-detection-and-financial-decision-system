# app.py — Main entry point for Fraud Detection Dashboard
# Place this at the ROOT of your project directory

import streamlit as st

st.set_page_config(
    page_title="FraudGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] * {
    color: #e8eaf0 !important;
}

/* Headings */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 16px;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #13161e !important;
    border: 1px solid #1e2330 !important;
    color: #e8eaf0 !important;
}

/* Hide default Streamlit footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

home    = st.Page("dashboard/pages/1_home.py", title="Overview",          icon="🏠")
fraud   = st.Page("dashboard/pages/2_fraud.py", title="Fraud Alerts",      icon="🚨")
explain = st.Page("dashboard/pages/3_explain.py", title="Explainability",  icon="🔍")
spend   = st.Page("dashboard/pages/4_spending.py", title="Spending Patterns", icon="📊")
credit  = st.Page("dashboard/pages/5_credit.py", title="Risk Score",       icon="💳")

pg = st.navigation([home, fraud, explain, spend, credit])
pg.run()