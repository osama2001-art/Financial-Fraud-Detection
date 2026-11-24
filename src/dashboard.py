
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("Financial Fraud Detection & Risk Intelligence")
st.caption("AI-powered post-transaction fraud monitoring")

# Load alert data
file_path = Path(__file__).resolve().parent / "fraud_alerts.csv"

if not file_path.exists():
    st.error("fraud_alerts.csv not found. Run main.py first.")
    st.stop()

alerts = pd.read_csv(file_path)

# Dashboard metrics

total_alerts = len(alerts)
high_risk = (alerts["Risk_Level"] == "HIGH").sum()
medium_risk = (alerts["Risk_Level"] == "MEDIUM").sum()
actual_fraud = alerts["Actual_Fraud"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Flagged Transactions", f"{total_alerts:,}")
col2.metric("High Risk", f"{high_risk:,}")
col3.metric("Medium Risk", f"{medium_risk:,}")
col4.metric("Actual Fraud in Alerts", f"{actual_fraud:,}")

st.divider()

# Model comparison

st.subheader("Machine Learning Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression"
    ],
    "Precision": [98.77, 99.27, 98.88],
    "Recall": [92.44, 82.59, 96.13],
    "F1 Score": [95.50, 90.17, 97.49]
})

st.dataframe(comparison, use_container_width=True)

fig = px.bar(
    comparison,
    x="Model",
    y=["Precision", "Recall", "F1 Score"],
    barmode="group",
    title="Model Performance Comparison",
    range_y=[0, 100]
)

st.plotly_chart(fig, use_container_width=True)


# Risk distribution
st.subheader("Transaction Risk Distribution")

risk_counts = alerts["Risk_Level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Transactions"]

fig = px.pie(
    risk_counts,
    names="Risk Level",
    values="Transactions",
    hole=0.45
)

st.plotly_chart(fig, use_container_width=True)

# Suspicious transactions
st.subheader("Suspicious Transaction Monitoring")

risk_filter = st.selectbox(
    "Filter by risk level",
    ["ALL", "HIGH", "MEDIUM"]
)

if risk_filter != "ALL":
    displayed_alerts = alerts[
        alerts["Risk_Level"] == risk_filter
    ]
else:
    displayed_alerts = alerts

st.dataframe(
    displayed_alerts,
    use_container_width=True
)

# Download report
st.download_button(
    "Download Fraud Alert Report",
    data=alerts.to_csv(index=False),
    file_name="fraud_alerts.csv",
    mime="text/csv"
)


# Interactive Transaction Alerts
st.subheader("🚨 Transaction Alert Explorer")

# Filter by risk level
risk_filter = st.selectbox(
    "Filter by Risk Level",
    ["ALL", "HIGH", "MEDIUM"]
)

if risk_filter == "ALL":
    filtered_alerts = alerts.copy()
else:
    filtered_alerts = alerts[
        alerts["Risk_Level"] == risk_filter
    ].copy()

# Sort by fraud risk score
filtered_alerts = filtered_alerts.sort_values(
    by="Fraud_Risk_Score",
    ascending=False
)

st.caption(
    f"Showing {len(filtered_alerts):,} flagged transactions"
)

st.dataframe(
    filtered_alerts,
    use_container_width=True,
    hide_index=True
)

