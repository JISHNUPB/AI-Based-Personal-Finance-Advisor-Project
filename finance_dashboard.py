 # -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re
from datetime import datetime
import io

# -------------------------------------------
# 1️⃣ Load trained model and vectorizer
# -------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("category_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------------------
# 2️⃣ Page Setup
# -------------------------------------------
st.set_page_config(page_title="💰 AI Personal Finance Dashboard", layout="wide")

# -------------------------------------------
# 💎 Custom CSS for Styling
# -------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #f5f7fa, #c3cfe2);
        font-family: 'Poppins', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 48px !important;
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 20px !important;
        color: #475569;
        margin-top: 0;
        margin-bottom: 40px;
    }
    .feature-card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        text-align: center;
        transition: 0.3s;
        color: black;
    }
    .feature-card h3 {
        color: black;
    }
    .feature-card p {
        color: black;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    }
    .input-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------
# 🏠 Landing Page
# -------------------------------------------
st.markdown("<h1 class='title'>💰 AI-Based Personal Finance Advisor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Automatically analyze your SMS transactions, categorize expenses, and discover saving opportunities with intelligent insights.</p>", unsafe_allow_html=True)

# -------------------------------------------
# 🌟 Feature Highlights
# -------------------------------------------
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("""
        <div class='feature-card'>
            <h3>📊 Smart Categorization</h3>
            <p>Our AI model intelligently classifies your expenses — from food to shopping to travel — within seconds.</p>
        </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown("""
        <div class='feature-card'>
            <h3>💡 Personalized Insights</h3>
            <p>Get actionable recommendations on how to save money based on your spending habits.</p>
        </div>
    """, unsafe_allow_html=True)

with colC:
    st.markdown("""
        <div class='feature-card'>
            <h3>📈 Interactive Dashboard</h3>
            <p>Visualize your financial health with dynamic charts, spending trends, and downloadable reports.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------

# -------------------------------------------
# 📤 Input Option (CSV Upload Only)
# -------------------------------------------
st.markdown("### 📂 Upload Your SMS Transaction File")

df = None  # Initialize df variable

uploaded_file = st.file_uploader(
    "Choose a CSV file (must contain a column named 'SMS')",
    type=["csv"],
    key="csv_uploader"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "SMS" not in df.columns:
            st.error("❌ The uploaded file must have a column named 'SMS'.")
            df = None
        else:
            st.success(f"✅ File uploaded successfully! {len(df)} records loaded.")
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

else:
    st.info("👆 Please upload a CSV file containing your SMS transaction data to begin analysis.")

# -------------------------------------------
# 🔍 Data Processing & Visualization
# -------------------------------------------
if df is not None:
    # Preprocessing
    df["SMS"] = df["SMS"].astype(str).str.lower().str.strip()

    # Extract amount from SMS text
    def extract_amount(text):
        match = re.search(r"(?i)(?:inr|aed|₹)\s?([\d,]+\.?\d*)", text)
        if match:
            return float(match.group(1).replace(",", ""))
        return None

    df["Amount"] = df["SMS"].apply(extract_amount)
    
    # Handle cases where no amount is found
    if df["Amount"].isna().all():
        st.warning("⚠️ No transaction amounts detected in the messages. Analysis will continue without amount data.")
        df["Amount"] = 0  # Default to 0 for visualization
    else:
        df = df.dropna(subset=["Amount"])

    # Predict category using trained model
    X_new = vectorizer.transform(df["SMS"])
    df["Predicted_Category"] = model.predict(X_new)

    # Detect credit/debit
    def detect_type(text):
        text = str(text).lower()
        if any(x in text for x in ["debited", "spent", "purchase", "paid", "withdrawal"]):
            return "Debit"
        elif any(x in text for x in ["credited", "received", "refund", "deposit", "salary"]):
            return "Credit"
        else:
            return "Unknown"

    df["Type"] = df["SMS"].apply(detect_type)

    st.success("✅ Transactions analyzed successfully!")

    # -------------------------------------------
    # 🎛️ Sidebar Filters
    # -------------------------------------------
    st.sidebar.header("🔎 Filters")
    selected_type = st.sidebar.multiselect(
        "Transaction Type", df["Type"].unique(), default=df["Type"].unique()
    )
    selected_cat = st.sidebar.multiselect(
        "Category", df["Predicted_Category"].unique(), default=df["Predicted_Category"].unique()
    )

    filtered_df = df[
        (df["Type"].isin(selected_type)) &
        (df["Predicted_Category"].isin(selected_cat))
    ]

    # -------------------------------------------
    # 💎 Key Metrics (Dynamic with Filter)
    # -------------------------------------------
    col1, col2, col3 = st.columns(3)
    total_spent = filtered_df[filtered_df["Type"] == "Debit"]["Amount"].sum()
    total_credited = filtered_df[filtered_df["Type"] == "Credit"]["Amount"].sum()
    top_cat = (
        filtered_df.groupby("Predicted_Category")["Amount"].sum().idxmax()
        if not filtered_df.empty else "N/A"
    )

    col1.metric("💸 Total Spent", f"₹{total_spent:,.2f}")
    col2.metric("💰 Total Credited", f"₹{total_credited:,.2f}")
    col3.metric("🏆 Top Category", top_cat)

    st.divider()

    # -------------------------------------------
    # 📊 Visualizations
    # -------------------------------------------
    st.subheader("📊 Spending Insights")

    if not filtered_df.empty:
        category_spend = (
            filtered_df.groupby("Predicted_Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        # Pie Chart
        fig_pie = px.pie(
            names=category_spend.index,
            values=category_spend.values,
            title="Expense Breakdown by Category",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Bar Chart
        fig_bar = px.bar(
            category_spend,
            x=category_spend.index,
            y=category_spend.values,
            text=category_spend.values,
            title="Total Spending by Category",
            color=category_spend.index,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_bar.update_traces(texttemplate="₹%{text:.2s}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # -------------------------------------------
        # 💡 Smart Savings Advisor (Now Dynamic)
        # -------------------------------------------
        st.subheader("💡 Smart Savings Advisor")

        if total_credited > 0:
            spend_ratio = total_spent / total_credited
            st.progress(min(spend_ratio, 1.0))
            st.caption(f"Spending to Income Ratio: {spend_ratio*100:.1f}%")

            if total_spent > total_credited * 0.8:
                st.warning("⚠️ Spending more than 80% of your income! Time to tighten the budget.")
            elif total_spent > total_credited * 0.6:
                st.info("💡 You're spending 60-80% of your income. Consider increasing your savings rate.")
            else:
                st.success("✅ Great! You're maintaining a healthy spending-to-income ratio.")

        # Use filtered data to determine top category
        if top_cat == "Food":
            st.info("🍔 Reduce food deliveries — meal prepping can save up to 30%.")
        elif top_cat == "Shopping":
            st.info("🛍️ Set a shopping cap for the month to control impulse buys.")
        elif top_cat == "Bills":
            st.info("💡 Audit your recurring subscriptions — hidden savings often lie there.")
        elif top_cat == "Travel":
            st.info("✈️ Consider planning trips in off-peak seasons to cut travel costs.")
        else:
            st.info("💰 Keep tracking your spending — consistency is the key to savings!")

        # -------------------------------------------
        # 🧾 Data Table + Download Option
        # -------------------------------------------
        with st.expander("📋 View All Transactions"):
            st.dataframe(filtered_df[["SMS", "Amount", "Type", "Predicted_Category"]])

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Categorized Transactions (CSV)",
            data=csv,
            file_name="categorized_transactions.csv",
            mime="text/csv",
        )
    else:
        st.warning("No transactions match the selected filters.")

else:
    # Instructions when no data is loaded
    st.info("👆 Choose an input method above to start analyzing your transactions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align:center; color:#475569;'>
            <p>© 2025 AI Personal Finance Advisor | Built with &#10084;&#65039; using Streamlit & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
