"""
Retail Intelligence System: Customer Segmentation & Purchasing Behavior Prediction
Complete ML Prototype Dashboard built with Streamlit, Plotly, Pandas, and Scikit-Learn.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ============================================================

st.set_page_config(
    page_title="Retail Intelligence System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark navy & blue professional dashboard styling
CUSTOM_CSS = """
<style>
    /* Dark Navy & Slate Theme Colors */
    :root {
        --bg-navy-dark: #0B1120;
        --card-navy: #131E32;
        --card-border: #1E293B;
        --accent-blue: #3B82F6;
        --accent-sky: #38BDF8;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
    }

    /* Force dark background & readable text on overall app */
    .stApp {
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li, [data-testid="stSidebar"] label {
        color: #94A3B8 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #F8FAFC !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Headings and Text */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    p, span, label {
        color: #CBD5E1;
    }

    /* Card styling for Home and KPI sections */
    .feature-card {
        background: linear-gradient(145deg, #131E32 0%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.12);
    }

    /* Badges */
    .badge-blue {
        background-color: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* KPI metric container */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #38BDF8 !important; /* Vibrant Sky Blue */
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }

    /* Streamlit borders & card containers */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #131E32 !important;
        border-color: #1E293B !important;
        border-radius: 12px !important;
    }

    /* Number inputs and text fields */
    input, select, textarea {
        color: #F8FAFC !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

# Color Palette mapping for consistent styling across charts
SEGMENT_COLORS: Dict[str, str] = {
    "Best Customers": "#3B82F6",             # Electric Blue
    "Loyal Customers": "#10B981",            # Emerald Green
    "At-Risk Valuable Customers": "#F59E0B", # Amber / Alert
    "Regular Customers": "#8B5CF6",          # Violet / Purple
    "Inactive Customers": "#64748B",         # Slate Grey
    "Recent Customers": "#06B6D4",           # Cyan
}

DEFAULT_PALETTE = px.colors.qualitative.Plotly

def apply_chart_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Applies a consistent dark navy layout theme to Plotly figures."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="Inter, system-ui, sans-serif"),
        title=dict(font=dict(size=15, color="#F8FAFC")),
        legend=dict(
            font=dict(color="#CBD5E1"),
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(30, 41, 59, 0.8)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155")
    )
    return fig


# ============================================================
# 2. DATA AND MODEL LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Loads all three project CSV files with validation and caching."""
    errs = []
    df_pred, df_seg, df_models = None, None, None

    pred_path = BASE_DIR / "customer_segmentation_and_purchase_predictions.csv"
    seg_path = BASE_DIR / "purchase_prediction_by_customer_segment.csv"
    models_path = BASE_DIR / "purchasing_model_comparison_time_based.csv"

    if not pred_path.exists():
        errs.append(f"Missing customer predictions file: {pred_path.name}")
    else:
        try:
            df_pred = pd.read_csv(pred_path)
        except Exception as e:
            errs.append(f"Error reading {pred_path.name}: {e}")

    if not seg_path.exists():
        errs.append(f"Missing segment summary file: {seg_path.name}")
    else:
        try:
            df_seg = pd.read_csv(seg_path)
        except Exception as e:
            errs.append(f"Error reading {seg_path.name}: {e}")

    if not models_path.exists():
        errs.append(f"Missing model comparison file: {models_path.name}")
    else:
        try:
            df_models = pd.read_csv(models_path)
        except Exception as e:
            errs.append(f"Error reading {models_path.name}: {e}")

    error_msg = "; ".join(errs) if errs else None
    return df_pred, df_seg, df_models, error_msg


@st.cache_resource(show_spinner=False)
def load_trained_model() -> Tuple[Optional[object], Optional[str]]:
    """Loads the trained purchasing prediction model pipeline."""
    model_path = BASE_DIR / "logistic_regression_purchasing_model.pkl"
    if not model_path.exists():
        return None, f"Model file not found at {model_path.name}"

    try:
        # Suppress scikit-learn cross-version unpickling warnings cleanly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded_model = joblib.load(model_path)
        return loaded_model, None
    except Exception as e:
        return None, f"Exception while loading model: {e}"


# Load assets
predictions_df, segments_df, models_df, load_error = load_datasets()
trained_model, model_load_error = load_trained_model()

if load_error:
    st.error(f"⚠️ Dataset Loading Warning: {load_error}")
    st.info("Ensure all CSV files are located in the project root directory.")


# ============================================================
# 3. NAVIGATION STATE MANAGEMENT
# ============================================================

PAGE_HOME = "🏠 Home"
PAGE_SEGMENTATION = "👥 Customer Segmentation"
PAGE_PREDICTION = "🛒 Purchase Prediction"
PAGE_EVALUATION = "📈 Model Evaluation"
PAGE_INSIGHTS = "💡 Business Insights"

PAGE_OPTIONS = [
    PAGE_HOME,
    PAGE_SEGMENTATION,
    PAGE_PREDICTION,
    PAGE_EVALUATION,
    PAGE_INSIGHTS
]

if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = PAGE_HOME

def navigate_to(page_name: str) -> None:
    """Callback to switch pages when feature card action buttons are clicked."""
    st.session_state.nav_selection = page_name


# Sidebar Navigation
st.sidebar.markdown("### 🛍️ Retail Intelligence")
st.sidebar.caption(" A Decision Support System")

current_page = st.sidebar.radio(
    "Navigation Menu",
    PAGE_OPTIONS,
    key="nav_selection"
)

st.sidebar.divider()

# Sidebar Metadata & System Specs
st.sidebar.markdown("#### ⚙️ System Specifications")
st.sidebar.markdown(
    """
    - **Prediction Horizon:** Next 30 Days
    - **Selected Prediction Model:** Logistic Regression
    - **Evaluation Snapshot:** September 1, 2010
    - **Customers in September Evaluation:** 3,175 profiles (as reported by the existing evaluation output)
    - **Feature Engineering:** RFM + Basket Features
    - **Prediction Output:** Purchase probability and predicted class (Purchase / No Purchase)
    """
)
st.sidebar.divider()
st.sidebar.caption("Retail Intelligence | ML Project")


# ============================================================
# 4. PAGE 0: HOMEPAGE (DASHBOARD HUB)
# ============================================================

if current_page == PAGE_HOME:

    # Hero Title Banner
    st.title("🛍️ Retail Intelligence System")
    st.markdown(
        "##### **Customer Segmentation & Purchasing Behavior Prediction System**"
    )
    st.markdown(
        "Transform retail transaction data into actionable insights with RFM-based customer segmentation, 30-day purchase prediction, and data-driven marketing strategies."
    )

    st.write("")

    # Summary KPI Metric Banner
    if predictions_df is not None:
        total_customers = predictions_df["Customer ID"].nunique() if "Customer ID" in predictions_df.columns else len(predictions_df)
        total_segments = predictions_df["Customer_Segment"].nunique() if "Customer_Segment" in predictions_df.columns else 6
        actual_purchasers = int(predictions_df["Purchased_Next_30_Days"].sum()) if "Purchased_Next_30_Days" in predictions_df.columns else 846
        predicted_purchasers = int(predictions_df["Predicted_Purchase"].sum()) if "Predicted_Purchase" in predictions_df.columns else 1178
    else:
        total_customers, total_segments, actual_purchasers, predicted_purchasers = 3175, 6, 846, 1178

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        with st.container(border=True):
            st.metric("Total Customers Evaluated", f"{total_customers:,}")
            st.caption("September 2010 snapshot")
    with col_kpi2:
        with st.container(border=True):
            st.metric("Customer Segments", f"{total_segments}")
            st.caption("Behavioral RFM clusters")
    with col_kpi3:
        with col_kpi3:
            with st.container(border=True):
                st.metric("Model Accuracy", "70.27%")
                st.caption("Logistic Regression ")
    with col_kpi4:
        with st.container(border=True):
            st.metric("Purchase Horizon", "30 Days")
            st.caption("Future Prediction")

    st.write("")
    st.markdown("### Core System Capabilities")

    # Four Clickable Feature Cards (2x2 Grid)
    row1_col1, row1_col2 = st.columns(2, gap="medium")

    with row1_col1:
        with st.container(border=True):
            st.markdown("#### 👥 1. Customer Segmentation")
            st.markdown(
                "**Discover meaningful customer groups using RFM analysis.**\n\n"
                "Analyze customer purchasing history to identify valuable, loyal, recent, and at-risk customers. Explore segment distributions, spending patterns, and customer profiles to support targeted marketing decisions."
            )
            st.markdown('<span class="badge-blue">RFM Analysis</span> &nbsp; <span class="badge-green">Customer Insights</span>', unsafe_allow_html=True)
            st.write("")
            st.button(
                "Explore Customer Segments →",
                key="btn_home_seg",
                width="stretch",
                type="primary",
                on_click=navigate_to,
                args=(PAGE_SEGMENTATION,)
            )

    with row1_col2:
        with st.container(border=True):
            st.markdown("#### 🛒 2. Purchase Prediction")
            st.markdown(
                "**Predict customer purchase likelihood over the next 30 days.**\n\n"
                "Use a trained Logistic Regression model to estimate purchase probability from customer behavior. Enter customer metrics or select a persona to explore predictions and relevant marketing actions."
            )
            st.markdown('<span class="badge-blue">Logistic Regression</span> &nbsp; <span class="badge-amber">30-Day Forecast</span>', unsafe_allow_html=True)
            st.write("")
            st.button(
                "Launch Purchase Predictor →",
                key="btn_home_pred",
                width="stretch",
                type="primary",
                on_click=navigate_to,
                args=(PAGE_PREDICTION,)
            )

    row2_col1, row2_col2 = st.columns(2, gap="medium")

    with row2_col1:
        with st.container(border=True):
            st.markdown("#### 📈 3. Model Evaluation")
            st.markdown(
                "**Compare machine learning models using time-based validation.**\n\n"
                "Evaluate Logistic Regression, Decision Tree, and Random Forest using Accuracy, Precision, Recall, and F1-score. Understand model trade-offs and why Logistic Regression was selected for the purchase prediction prototype."
            )
            st.markdown('<span class="badge-blue">TTemporal Validation</span> &nbsp; <span class="badge-green">3 Models Compared</span>', unsafe_allow_html=True)
            st.write("")
            st.button(
                "Compare Evaluation Metrics →",
                key="btn_home_eval",
                width="stretch",
                type="primary",
                on_click=navigate_to,
                args=(PAGE_EVALUATION,)
            )

    with row2_col2:
        with st.container(border=True):
            st.markdown("#### 💡 4. Business Insights")
            st.markdown(
                "**Turn customer behavior into actionable business insights.**\n\n"
                "Explore customer value, purchase patterns, and segment-level performance. Use purchasing trends and tailored marketing recommendations to support customer retention, targeted campaigns, and revenue growth."
            )
            st.markdown('<span class="badge-blue">Customer Analytics</span> &nbsp; <span class="badge-emerald">Marketing Strategies</span>', unsafe_allow_html=True)
            st.write("")
            st.button(
                "View Strategic Insights →",
                key="btn_home_insights",
                width="stretch",
                type="primary",
                on_click=navigate_to,
                args=(PAGE_INSIGHTS,)
            )

    st.write("")
    st.divider()



# ============================================================
# 5. PAGE 1: CUSTOMER SEGMENTATION EXPLORER
# ============================================================

elif current_page == PAGE_SEGMENTATION:

    st.header("👥 Customer Segmentation Explorer")
    st.caption("Discover meaningful customer cohorts using RFM (Recency, Frequency, Monetary) behavioral analysis.")

    if predictions_df is None or predictions_df.empty:
        st.error("Customer segmentation dataset is unavailable.")
        st.stop()

    df_customers = predictions_df.copy()

    # A. Summary Metrics
    total_cust = df_customers["Customer ID"].nunique() if "Customer ID" in df_customers.columns else len(df_customers)
    unique_segments = sorted(df_customers["Customer_Segment"].dropna().unique().tolist()) if "Customer_Segment" in df_customers.columns else []
    num_segments = len(unique_segments)

    # Calculate largest segment & highest spend segment accurately
    if "Customer_Segment" in df_customers.columns:
        seg_counts = df_customers["Customer_Segment"].value_counts()
        largest_seg_name = seg_counts.index[0]
        largest_seg_count = seg_counts.iloc[0]
        largest_seg_pct = (largest_seg_count / total_cust) * 100

        if "Monetary" in df_customers.columns:
            seg_spend = df_customers.groupby("Customer_Segment")["Monetary"].sum().sort_values(ascending=False)
            highest_val_name = seg_spend.index[0]
            highest_val_total = seg_spend.iloc[0]
            highest_val_avg = df_customers[df_customers["Customer_Segment"] == highest_val_name]["Monetary"].mean()
        else:
            highest_val_name = "Best Customers"
            highest_val_total = 1564371.62
            highest_val_avg = 1921.83
    else:
        largest_seg_name, largest_seg_count, largest_seg_pct = "Inactive Customers", 854, 26.9
        highest_val_name, highest_val_total, highest_val_avg = "Best Customers", 1564371.62, 1921.83

    # Summary metrics row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        with st.container(border=True):
            st.metric("Customers Represented", f"{total_cust:,}")
            st.caption("Evaluated snapshot population")
    with m_col2:
        with st.container(border=True):
            st.metric("Customer Segments", f"{num_segments}")
            st.caption("RFM behavioral groupings")
    with m_col3:
        with st.container(border=True):
            st.metric("Largest Segment", f"{largest_seg_name}")
            st.caption(f"{largest_seg_count:,} customers ({largest_seg_pct:.1f}% of total)")
    with m_col4:
        with st.container(border=True):
            st.metric("Highest-Value Segment", f"{highest_val_name}")
            st.caption(f"£{highest_val_total:,.0f} spend (avg £{highest_val_avg:,.2f})")


    st.write("")

    # B. Segment Distribution Visuals
    st.markdown("### 📊 Segment Distribution & Revenue Contribution")

    seg_summary_counts = (
        df_customers["Customer_Segment"]
        .value_counts()
        .reset_index()
    )
    seg_summary_counts.columns = ["Customer_Segment", "Customer_Count"]
    seg_summary_counts["Percentage"] = (seg_summary_counts["Customer_Count"] / total_cust) * 100

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Donut Chart
        donut_colors = [SEGMENT_COLORS.get(seg, "#3B82F6") for seg in seg_summary_counts["Customer_Segment"]]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=seg_summary_counts["Customer_Segment"],
                    values=seg_summary_counts["Customer_Count"],
                    hole=0.30,
                    marker=dict(colors=donut_colors),
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>"
                )
            ]
        )
        fig_donut.update_layout(title="Customer Distribution by Segment")
        st.plotly_chart(apply_chart_theme(fig_donut, height=400), width="stretch")

    with col_chart2:
        # Bar Chart of Customer Counts
        bar_colors = [SEGMENT_COLORS.get(seg, "#3B82F6") for seg in seg_summary_counts["Customer_Segment"]]
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=seg_summary_counts["Customer_Segment"],
                    y=seg_summary_counts["Customer_Count"],
                    marker=dict(color=bar_colors),
                    text=seg_summary_counts["Customer_Count"].map(lambda x: f"{x:,}"),
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Customer Count: %{y:,}<extra></extra>"
                )
            ]
        )
        fig_bar.update_layout(
            title="Customer Count in Each Segment",
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers",
            yaxis=dict(range=[0, seg_summary_counts["Customer_Count"].max() * 1.15])
        )
        st.plotly_chart(apply_chart_theme(fig_bar, height=380), width="stretch")

    # Segment Spend Breakdown Chart
    if "Monetary" in df_customers.columns:
        spend_by_seg = (
            df_customers.groupby("Customer_Segment")
            .agg(Total_Spend=("Monetary", "sum"), Avg_Spend=("Monetary", "mean"))
            .reset_index()
            .sort_values("Total_Spend", ascending=False)
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_spend = go.Figure(
                data=[
                    go.Bar(
                        x=spend_by_seg["Customer_Segment"],
                        y=spend_by_seg["Total_Spend"],
                        marker=dict(color=[SEGMENT_COLORS.get(s, "#3B82F6") for s in spend_by_seg["Customer_Segment"]]),
                        text=spend_by_seg["Total_Spend"].map(lambda x: f"£{x:,.0f}"),
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Total Spend: £%{y:,.2f}<extra></extra>"
                    )
                ]
            )
            fig_spend.update_layout(
                title="Total Revenue Contribution by Segment (£)",
                xaxis_title="Customer Segment",
                yaxis_title="Total Spend ()",
                yaxis=dict(range=[0, spend_by_seg["Total_Spend"].max() * 1.15])
            )
            st.plotly_chart(apply_chart_theme(fig_spend, height=360), width="stretch")

        with col_s2:
            fig_avg_spend = go.Figure(
                data=[
                    go.Bar(
                        x=spend_by_seg["Customer_Segment"],
                        y=spend_by_seg["Avg_Spend"],
                        marker=dict(color=[SEGMENT_COLORS.get(s, "#3B82F6") for s in spend_by_seg["Customer_Segment"]]),
                        text=spend_by_seg["Avg_Spend"].map(lambda x: f"£{x:,.2f}"),
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Average Spend: £%{y:,.2f}<extra></extra>"
                    )
                ]
            )
            fig_avg_spend.update_layout(
                title="Average Spend per Customer by Segment (£)",
                xaxis_title="Customer Segment",
                yaxis_title="Average Spend (£)",
                yaxis=dict(range=[0, spend_by_seg["Avg_Spend"].max() * 1.15])
            )
            st.plotly_chart(apply_chart_theme(fig_avg_spend, height=360), width="stretch")

    st.write("")

    # C. Segment Profiles Summary Table
    st.markdown("### 📋 Segment Behavioral Profiles")
    st.caption("Aggregated RFM metrics and actual vs. predicted purchase propensity across segments.")

    profile_df = df_customers.groupby("Customer_Segment").agg(
        Customers=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Spend=("Monetary", "mean"),
        Avg_Order_Value=("Average_Order_Value", "mean"),
        Avg_Total_Qty=("Total_Quantity", "mean"),
        Actual_Purchase_Rate=("Purchased_Next_30_Days", "mean"),
        Avg_Predicted_Probability=("Purchase_Probability", "mean")
    ).reset_index()

    profile_df["Customer_Share"] = (profile_df["Customers"] / total_cust)

    # Format table for display
    display_profile = profile_df[[
        "Customer_Segment",
        "Customers",
        "Customer_Share",
        "Avg_Recency",
        "Avg_Frequency",
        "Avg_Spend",
        "Avg_Order_Value",
        "Avg_Total_Qty",
        "Actual_Purchase_Rate",
        "Avg_Predicted_Probability"
    ]].copy()

    display_profile.columns = [
        "Segment Name",
        "Customers",
        "Share (%)",
        "Avg Recency (Days)",
        "Avg Frequency",
        "Avg Spend (£)",
        "Avg Line Spend (£)",
        "Avg Quantity",
        "Actual 30d Buy Rate",
        "Avg Predicted Prob"
    ]

    st.dataframe(
        display_profile.style.format({
            "Customers": "{:,}",
            "Share (%)": "{:.1%}",
            "Avg Recency (Days)": "{:.1f}",
            "Avg Frequency": "{:.2f}",
            "Avg Spend (£)": "£{:,.2f}",
            "Avg Line Spend (£)": "£{:,.2f}",
            "Avg Quantity": "{:.1f}",
            "Actual 30d Buy Rate": "{:.1%}",
            "Avg Predicted Prob": "{:.1%}"
        }),
        width="stretch",
        hide_index=True
    )

    with st.expander("📖 Segment Definitions &  Interpretations"):
        st.markdown(
            """
            - **Best Customers (814 customers | £1,921.83 avg spend | 45.9 days recency):** High-frequency, high-value repeat shoppers accounting for **56.8%** of cumulative revenue. Demonstrates a strong **54.7%** actual 30-day repeat purchase rate. Top priority for VIP retention.
            - **Loyal Customers (456 customers | £781.83 avg spend | 74.9 days recency):** Steady, dependable shoppers with consistent transaction frequency (3.5 orders avg) and a **32.0%** actual repurchase rate. Prime cohort for cross-selling.
            - **At-Risk Valuable Customers (483 customers | £972.41 avg spend | 78.9 days recency):** High historical monetary value now showing increasing inactivity. Despite lapsing, **35.8%** still repurchased in the 30-day window, indicating high recoverable revenue potential if re-engaged.
            - **Regular Customers (523 customers | £295.94 avg spend | 112.3 days recency):** Moderate spenders with lower frequency (1.6 orders) and an **11.3%** repurchase rate. Benefit from threshold incentives ("Spend £50, get £10 off").
            - **Inactive Customers (854 customers | £251.41 avg spend | 134.4 days recency):** Largest group (26.9% of customer base) with high recency and dormant engagement. Displays a low **2.7%** actual repurchase rate.
            - **Recent Customers (45 customers | £251.16 avg spend | 122.3 days recency):** Single-order first-time purchasers (frequency = 1.0) with **0.0%** observed 30-day repeat purchases in the historical window. Need structured onboarding nurture sequences.
            """
        )

    st.write("")
    st.divider()

    # D. Interactive Customer Explorer
    st.markdown("### 🔍 Interactive Customer Explorer")
    st.caption("Search, filter, inspect, and export individual customer profiles.")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        seg_filter_opts = ["All Segments"] + unique_segments
        selected_seg_filter = st.selectbox("Filter by Segment", seg_filter_opts)

    with ctrl_col2:
        search_cust_id = st.text_input("Search Customer ID", placeholder="e.g. 12346")

    with ctrl_col3:
        pred_filter_opts = ["All Predictions", "Likely to Purchase (1)", "Not Predicted (0)"]
        selected_pred_filter = st.selectbox("Model Prediction", pred_filter_opts)

    with ctrl_col4:
        outcome_filter_opts = ["All Outcomes", "Actually Purchased (1)", "Did Not Purchase (0)"]
        selected_outcome_filter = st.selectbox("Actual 30d Outcome", outcome_filter_opts)

    # Apply filters
    filtered_df = df_customers.copy()

    if selected_seg_filter != "All Segments":
        filtered_df = filtered_df[filtered_df["Customer_Segment"] == selected_seg_filter]

    if search_cust_id.strip():
        search_term = search_cust_id.strip()
        filtered_df = filtered_df[filtered_df["Customer ID"].astype(str).str.contains(search_term, na=False)]

    if selected_pred_filter == "Likely to Purchase (1)":
        filtered_df = filtered_df[filtered_df["Predicted_Purchase"] == 1]
    elif selected_pred_filter == "Not Predicted (0)":
        filtered_df = filtered_df[filtered_df["Predicted_Purchase"] == 0]

    if selected_outcome_filter == "Actually Purchased (1)":
        filtered_df = filtered_df[filtered_df["Purchased_Next_30_Days"] == 1]
    elif selected_outcome_filter == "Did Not Purchase (0)":
        filtered_df = filtered_df[filtered_df["Purchased_Next_30_Days"] == 0]

    # Sorting controls
    sort_col1, sort_col2 = st.columns([3, 1])
    with sort_col1:
        sortable_columns = ["Customer ID", "Recency", "Frequency", "Monetary", "Average_Order_Value", "Total_Quantity", "Purchase_Probability"]
        selected_sort_col = st.selectbox("Sort Table By", sortable_columns, index=sortable_columns.index("Purchase_Probability"))
    with sort_col2:
        sort_ascending = st.checkbox("Ascending Order", value=False)

    filtered_df = filtered_df.sort_values(by=selected_sort_col, ascending=sort_ascending)

    st.write(f"Showing **{len(filtered_df):,}** of **{total_cust:,}** matching customer profiles:")

    if filtered_df.empty:
        st.warning("⚠️ No customer records match the selected filter criteria. Please adjust your search or segment filter.")
    else:
        # Prepare display dataframe
        export_cols = [
            "Customer ID",
            "Customer_Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "Average_Order_Value",
            "Total_Quantity",
            "Purchase_Probability",
            "Predicted_Purchase",
            "Purchased_Next_30_Days"
        ]
        available_cols = [c for c in export_cols if c in filtered_df.columns]
        display_cust_table = filtered_df[available_cols].copy()

        # Format columns nicely
        format_mapping = {}
        if "Monetary" in display_cust_table.columns:
            format_mapping["Monetary"] = "£{:,.2f}"
        if "Average_Order_Value" in display_cust_table.columns:
            format_mapping["Average_Order_Value"] = "£{:,.2f}"
        if "Purchase_Probability" in display_cust_table.columns:
            format_mapping["Purchase_Probability"] = "{:.1%}"

        st.dataframe(
            display_cust_table.style.format(format_mapping),
            width="stretch",
            hide_index=True
        )

        # Download button
        csv_download = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Filtered Customer Records (CSV)",
            data=csv_download,
            file_name=f"retail_customers_filtered_{selected_seg_filter.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )


# ============================================================
# 6. PAGE 2: PURCHASE PREDICTION INTERFACE
# ============================================================

elif current_page == PAGE_PREDICTION:

    st.header("🛒 Customer Purchase Prediction")
    st.caption("Estimate whether an individual customer is likely to purchase within the next 30 days using the trained Machine Learning pipeline.")

    if trained_model is None:
        st.error("⚠️ The trained Logistic Regression model could not be loaded.")
        st.code(model_load_error or "Unknown error loading logistic_regression_purchasing_model.pkl")
        st.stop()

    # Model input feature confirmation
    expected_features = ["Recency", "Frequency", "Monetary", "Average_Order_Value", "Total_Quantity"]

    st.markdown(
        """
        <div style="background-color: rgba(30, 41, 59, 0.4); border: 1px solid #1E293B; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;">
            <span style="color: #60A5FA; font-weight: 600;">Active Model Architecture:</span>
            <span style="color: #94A3B8;"> Scikit-Learn Pipeline (StandardScaler → LogisticRegression). Expected input vector: <code>Recency</code>, <code>Frequency</code>, <code>Monetary</code>, <code>Average_Order_Value</code>, <code>Total_Quantity</code>.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Customer Preset Selection for convenience during demonstration
    st.markdown("####  Quick Customer Persona Presets")
    st.caption("Select a persona preset to populate realistic customer inputs or enter custom figures below:")

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    # Store form state defaults
    if "input_recency" not in st.session_state:
        st.session_state.input_recency = 53
        st.session_state.input_freq = 5
        st.session_state.input_monetary = 600.0
        st.session_state.input_aov = 10.0
        st.session_state.input_qty = 15

    def apply_preset(r: int, f: int, m: float, aov: float, q: int):
        st.session_state.input_recency = r
        st.session_state.input_freq = f
        st.session_state.input_monetary = m
        st.session_state.input_aov = aov
        st.session_state.input_qty = q

    with preset_col1:
        if st.button("🌟 Best Customer Preset", width="stretch"):
            apply_preset(15, 12, 2450.0, 24.5, 210)
            st.rerun()
    with preset_col2:
        if st.button("💎 Loyal Customer Preset", width="stretch"):
            apply_preset(45, 6, 820.0, 16.0, 65)
            st.rerun()
    with preset_col3:
        if st.button("⚠️ At-Risk Customer Preset", width="stretch"):
            apply_preset(88, 4, 950.0, 18.5, 52)
            st.rerun()
    with preset_col4:
        if st.button("💤 Inactive Customer Preset", width="stretch"):
            apply_preset(165, 1, 140.0, 14.0, 8)
            st.rerun()

    st.write("")

    # A. Customer Input Form
    st.markdown("#### 📝 Customer Input Attributes")

    with st.container(border=True):
        form_col1, form_col2 = st.columns(2, gap="large")

        with form_col1:
            recency = st.number_input(
                "Recency (Days since last purchase)",
                min_value=0,
                max_value=1500,
                value=int(st.session_state.input_recency),
                step=1,
                help="Number of elapsed days between the customer's last purchase and the snapshot date."
            )

            frequency = st.number_input(
                "Frequency (Total distinct orders)",
                min_value=1,
                max_value=500,
                value=int(st.session_state.input_freq),
                step=1,
                help="Total count of distinct transaction sessions/invoices recorded for this customer."
            )

            monetary = st.number_input(
                "Monetary Value (£ Cumulative Spend)",
                min_value=0.0,
                max_value=500000.0,
                value=float(st.session_state.input_monetary),
                step=25.0,
                help="Total cumulative monetary spend (£) across all transactions."
            )

        with form_col2:
            avg_order_value = st.number_input(
                "Average Order Value (£)",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.input_aov),
                step=1.0,
                help="Note: In this project's feature engineering, Average_Order_Value represents the mean transaction-line spend rather than total invoice basket amount."
            )

            total_quantity = st.number_input(
                "Total Quantity (Aggregate product units purchased)",
                min_value=0,
                max_value=250000,
                value=int(st.session_state.input_qty),
                step=5,
                help="Aggregate quantity of items purchased across all line items."
            )

            # st.caption(
            #     "ℹ️ **Feature Definition Note:** `Average_Order_Value` in the trained model pipeline reflects the mean "
            #     "transaction-line item expenditure (spend per item line) as engineered during dataset cleaning."
            # )

        predict_clicked = st.button("Predict 30-Day Purchase Probability", type="primary", width="stretch")

    # B. Prediction Execution
    if predict_clicked:
        try:
            # Construct feature vector with exact names and order expected by Pipeline
            input_df = pd.DataFrame([{
                "Recency": int(recency),
                "Frequency": int(frequency),
                "Monetary": float(monetary),
                "Average_Order_Value": float(avg_order_value),
                "Total_Quantity": int(total_quantity)
            }])[expected_features]

            # Model inference
            pred_class = int(trained_model.predict(input_df)[0])

            if hasattr(trained_model, "predict_proba"):
                probabilities = trained_model.predict_proba(input_df)[0]
                purchase_prob = float(probabilities[1])
            else:
                purchase_prob = float(pred_class)

            st.write("")
            st.markdown("### 🎯 Model Prediction Results")

            res_col1, res_col2 = st.columns([1.2, 1], gap="medium")

            with res_col1:
                with st.container(border=True):
                    st.markdown("##### Predicted Outcome")
                    if pred_class == 1:
                        st.markdown(
                            """
                            <div style="background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 16px; margin: 10px 0;">
                                <h3 style="color: #34D399; margin: 0;">🟢 Likely to Purchase within 30 Days</h3>
                                <p style="color: #CBD5E1; margin: 8px 0 0 0; font-size: 0.95rem;">
                                    The model predicts this customer has a <b>high propensity</b> to make a transaction in the upcoming 30-day window.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 16px; margin: 10px 0;">
                                <h3 style="color: #FBBF24; margin: 0;">🟠 Less Likely to Purchase within 30 Days</h3>
                                <p style="color: #CBD5E1; margin: 8px 0 0 0; font-size: 0.95rem;">
                                    The model predicts this customer has a <b>low propensity</b> to purchase within 30 days without targeted re-engagement.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.write("")
                    st.metric("Estimated Purchase Probability", f"{purchase_prob:.1%}")

                    # Styled progress bar with boundary marker
                    st.progress(
                        max(0.0, min(1.0, purchase_prob)),
                        text=f"Model Probability: {purchase_prob:.1%} (Classification Threshold: 50.0%)"
                    )

            with res_col2:
                # Gauge representation of purchase probability
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=purchase_prob * 100,
                        number={"suffix": "%", "font": {"size": 38, "color": "#38BDF8", "family": "Inter, sans-serif"}},
                        title={"text": "Purchase Propensity Score", "font": {"size": 15, "color": "#CBD5E1"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                            "bar": {"color": "#3B82F6"},
                            "bgcolor": "#131E32",
                            "borderwidth": 1,
                            "bordercolor": "#1E293B",
                            "steps": [
                                {"range": [0, 40], "color": "rgba(100, 116, 139, 0.25)"},
                                {"range": [40, 60], "color": "rgba(245, 158, 11, 0.3)"},
                                {"range": [60, 100], "color": "rgba(16, 185, 129, 0.35)"}
                            ],
                            "threshold": {
                                "line": {"color": "#EF4444", "width": 3},
                                "thickness": 0.75,
                                "value": 50
                            }
                        }
                    )
                )
                fig_gauge.update_layout(
                    height=240,
                    margin=dict(l=20, r=20, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_gauge, width="stretch")

            # C. Suggested Business Action
            st.write("")
            st.markdown("### 💡 Recommended Business Strategy")
            

            rec_box_style = (
                "background-color: #131E32; border: 1px solid #1E293B; border-left: 4px solid #3B82F6; "
                "border-radius: 8px; padding: 18px 20px; margin-top: 8px;"
            )

            # Rule-based strategic recommendations
            strategies = []
            if recency > 90:
                strategies.append(
                    "Re-Engagement & Win-Back Campaign : The customer has not purchased in over 90 days. "
                    "Deploy an automated reactivation sequence featuring a personalized discount, 'We Miss You' incentive, "
                    "or reminders of popular products from categories they previously purchased."
                )
            if frequency >= 8 and monetary >= 1000:
                strategies.append(
                    "VIP Loyalty Program Invitation : High cumulative spend (£{:,.0f}) and high frequency ({:d} orders) "
                    "place this customer in the top commercial tier. Provide exclusive early access to new catalog items, dedicated account support, "
                    "and tier-based loyalty perks to safeguard against competitive churn.".format(monetary, frequency)
                )
            if purchase_prob >= 0.50:
                strategies.append(
                    "High-Propensity Replenishment Offer : The customer displays strong buying momentum ({:.1%} likelihood). "
                    "Trigger replenishment notifications, recommend complementary cross-sell accessories, or offer free-shipping order-value boosters "
                    "within the next 14 days.".format(purchase_prob)
                )
            else:
                strategies.append(
                    "Low-Cost Content Nurture & Feedback Survey : Predicted propensity is under 50% ({:.1%}). "
                    "Avoid deep margin-eroding discounts; instead, deliver educational category content, product discovery newsletters, "
                    "or a short customer feedback survey to diagnose drop-off reasons.".format(purchase_prob)
                )

            st.markdown(
                f"""
                <div style="{rec_box_style}">
                    <h5 style="color: #60A5FA; margin-top: 0;">Suggested Commercial Strategy (Business Guidance):</h5>
                    <ul style="color: #E2E8F0; margin-bottom: 0; line-height: 1.6;">
                        {''.join(f'<li>{s}</li>' for s in strategies)}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as e:
            st.error(f"Prediction failed with error: {e}")
            st.caption("Please verify that all inputs contain valid numeric values.")

    st.write("")
    st.divider()

    # D. Historical Validation Explorer
    st.markdown("### 📜 Historical Evaluation Predictions Inspector")
    st.caption("Inspect how the trained Logistic Regression model performed across the 3,175 historical holdout customer records.")

    if predictions_df is not None and "Predicted_Purchase" in predictions_df.columns:
        with st.expander("🔍 View Historical Prediction Confusion Matrix & Sample Records"):
            hist_total = len(predictions_df)
            tp = int(((predictions_df["Predicted_Purchase"] == 1) & (predictions_df["Purchased_Next_30_Days"] == 1)).sum())
            tn = int(((predictions_df["Predicted_Purchase"] == 0) & (predictions_df["Purchased_Next_30_Days"] == 0)).sum())
            fp = int(((predictions_df["Predicted_Purchase"] == 1) & (predictions_df["Purchased_Next_30_Days"] == 0)).sum())
            fn = int(((predictions_df["Predicted_Purchase"] == 0) & (predictions_df["Purchased_Next_30_Days"] == 1)).sum())

            cm_c1, cm_c2, cm_c3, cm_c4 = st.columns(4)
            with cm_c1:
                st.metric("True Positives (TP)", f"{tp:,}", help="Predicted to purchase and actually purchased")
            with cm_c2:
                st.metric("True Negatives (TN)", f"{tn:,}", help="Predicted not to purchase and did not purchase")
            with cm_c3:
                st.metric("False Positives (FP)", f"{fp:,}", help="Predicted to purchase but did not purchase")
            with cm_c4:
                st.metric("False Negatives (FN)", f"{fn:,}", help="Predicted not to purchase but actually purchased")

            st.write("")
            st.dataframe(
                predictions_df[[
                    "Customer ID", "Customer_Segment", "Recency", "Frequency", "Monetary",
                    "Purchase_Probability", "Predicted_Purchase", "Purchased_Next_30_Days"
                ]].head(50).style.format({
                    "Monetary": "£{:,.2f}",
                    "Purchase_Probability": "{:.1%}"
                }),
                width="stretch",
                hide_index=True
            )


# ============================================================
# 7. PAGE 3: MODEL PERFORMANCE & EVALUATION
# ============================================================

elif current_page == PAGE_EVALUATION:

    st.header("📈 Machine Learning Model Evaluation")
    st.caption("Temporal holdout performance comparison across Logistic Regression, Decision Tree, and Random Forest.")

    if models_df is None or models_df.empty:
        st.error("⚠️ Model comparison results file (purchasing_model_comparison_time_based.csv) is missing.")
        st.stop()

    df_models_clean = models_df.copy()

    # Expected metric columns
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    for m in metric_cols:
        if m in df_models_clean.columns:
            df_models_clean[m] = pd.to_numeric(df_models_clean[m], errors="coerce")

    # A. Metrics Table
    st.markdown("### 🏆 Classifier Comparison Table")
    st.caption("Time-based holdout evaluation results recorded from the September 1, 2010 temporal split.")

    # Format table for clean percentage display
    table_formatter = {m: "{:.2%}" for m in metric_cols if m in df_models_clean.columns}

    st.dataframe(
        df_models_clean.style.format(table_formatter),
        width="stretch",
        hide_index=True
    )

    st.write("")

    # B. Interactive Model Comparison Chart
    st.markdown("### 📊 Interactive Metric Comparison")

    chart_select_col, _ = st.columns([2, 2])
    with chart_select_col:
        selected_eval_metric = st.selectbox(
            "Select Metric to Visualize",
            ["All Metrics (Side-by-Side)", "Accuracy", "Precision", "Recall", "F1 Score"],
            index=0
        )

    model_col_name = "Model" if "Model" in df_models_clean.columns else df_models_clean.columns[0]

    if selected_eval_metric == "All Metrics (Side-by-Side)":
        # Grouped bar chart comparing all 4 metrics across models
        melted_df = df_models_clean.melt(
            id_vars=[model_col_name],
            value_vars=[c for c in metric_cols if c in df_models_clean.columns],
            var_name="Evaluation Metric",
            value_name="Score"
        )
        melted_df["Score_Pct"] = melted_df["Score"] * 100
        melted_df["Text_Label"] = melted_df["Score_Pct"].map(lambda x: f"{x:.2f}%")

        fig_multi = px.bar(
            melted_df,
            x="Evaluation Metric",
            y="Score_Pct",
            color=model_col_name,
            barmode="group",
            text="Text_Label",
            title="Comprehensive Classifier Benchmark Across All Metrics",
            labels={"Score_Pct": "Metric Score (%)", "Evaluation Metric": "Metric"},
            color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B"]
        )
        fig_multi.update_traces(textposition="outside")
        fig_multi.update_layout(yaxis=dict(range=[0, 90]))
        st.plotly_chart(apply_chart_theme(fig_multi, height=420), width="stretch")

    else:
        # Single metric drilldown chart
        metric_chart_df = df_models_clean[[model_col_name, selected_eval_metric]].dropna().copy()
        metric_chart_df["Score_Pct"] = metric_chart_df[selected_eval_metric] * 100
        metric_chart_df["Text_Label"] = metric_chart_df["Score_Pct"].map(lambda x: f"{x:.2f}%")

        best_row = metric_chart_df.sort_values(by="Score_Pct", ascending=False).iloc[0]

        fig_single = px.bar(
            metric_chart_df,
            x=model_col_name,
            y="Score_Pct",
            text="Text_Label",
            color=model_col_name,
            title=f"Classifier Comparison: {selected_eval_metric}",
            labels={"Score_Pct": f"{selected_eval_metric} (%)", model_col_name: "Classifier Model"},
            color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6"]
        )
        fig_single.update_traces(textposition="outside")
        fig_single.update_layout(yaxis=dict(range=[0, metric_chart_df["Score_Pct"].max() * 1.2]))
        st.plotly_chart(apply_chart_theme(fig_single, height=380), width="stretch")

        st.markdown(
            f"**Best Performer on {selected_eval_metric}:** `{best_row[model_col_name]}` with **{best_row['Score_Pct']:.2f}%**."
        )

    st.write("")
    st.divider()

    # C. Model Interpretation & Selection Rationale
    st.markdown("### 🧠 Metric Interpretation & Model Selection")

    interp_col1, interp_col2 = st.columns(2, gap="large")

    with interp_col1:
        with st.container(border=True):
            st.markdown("#### 🎯 Understanding the Metrics")
            st.markdown(
                """
                - **Accuracy — 70.27% - Logistic Regression** Overall percentage of correct predictions. Accuracy alone may hide poor performance on a less common class.
                - **Precision (45.84% - Logistic Regression):** Of customers predicted to purchase, the percentage who actually purchased.
                - **Recall (72.81% - Decision Tree):** Of actual purchasers, the percentage correctly identified by the model.
                - **F1 Score (53.36% - Logistic Regression):** The harmonic mean of precision and recall. Logistic Regression achieved the highest F1-score in your comparison.
                """
            )

    with interp_col2:
        with st.container(border=True):
            st.markdown("####⚖️ Why Logistic Regression?")
            st.markdown(
                """
                In our time-based temporal evaluation:
                1. **1. Highest F1-score** 53.36%, the best balance of precision and recall among the tested models.
                2. **2. Highest accuracy and precision:** 70.27% accuracy and 45.84% precision, outperforming the other two models on these metrics.
                3. **3. Recall trade-off:** Decision Tree achieved higher recall (72.81% vs. 63.83%), but had lower precision and F1-score.
                4. **4. Interpretability** Logistic Regression is generally easier to interpret through its coefficients. Probability calibration must be verified separately.
                """
            )

    st.write("")



# ============================================================
# 8. PAGE 4: BUSINESS INSIGHTS & STRATEGIC RECOMMENDATIONS
# ============================================================

elif current_page == PAGE_INSIGHTS:

    st.header("💡 Strategic Business Insights")
    st.caption("Actionable commercial intelligence synthesized from RFM customer segmentation and predictive purchase modeling.")

    if predictions_df is None or predictions_df.empty or segments_df is None:
        st.error("⚠️ Business insights require valid customer and segment datasets.")
        st.stop()

    # A. Business KPIs
    total_customers = len(predictions_df)
    actual_purchasers = int(predictions_df["Purchased_Next_30_Days"].sum())
    pred_purchasers = int(predictions_df["Predicted_Purchase"].sum())
    actual_rate = (actual_purchasers / total_customers) * 100
    pred_rate = (pred_purchasers / total_customers) * 100
    avg_pred_prob = float(predictions_df["Purchase_Probability"].mean()) * 100
    total_revenue = float(predictions_df["Monetary"].sum())

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        with st.container(border=True):
            st.metric("Total Customers", f"{total_customers:,}")
            st.caption("Evaluated customer base")
    with kpi2:
        with st.container(border=True):
            st.metric("Actual Purchasers", f"{actual_purchasers:,}")
            st.caption(f"{actual_rate:.1f}% baseline conversion")
    with kpi3:
        with st.container(border=True):
            st.metric("Predicted Purchasers", f"{pred_purchasers:,}")
            st.caption(f"{pred_rate:.1f}% model prediction rate")
    with kpi4:
        with st.container(border=True):
            st.metric("Mean Purchase Likelihood", f"{avg_pred_prob:.1f}%")
            st.caption("Population-wide probability")
    with kpi5:
        with st.container(border=True):
            st.metric("Total Customer Spend", f"£{total_revenue:,.0f}")
            st.caption("Cumulative historical revenue")

    st.write("")

    # B. Actual vs. Predicted Purchases by Segment
    st.markdown("### 📊 Actual vs. Predicted Purchasers by Customer Segment")
    st.caption("Comparison of observed historical purchasers versus model-identified purchaser cohorts across segments.")

    # Calculate actual vs predicted aggregated by segment
    seg_perf = predictions_df.groupby("Customer_Segment").agg(
        Total_Customers=("Customer ID", "count"),
        Actual_Purchasers=("Purchased_Next_30_Days", "sum"),
        Predicted_Purchasers=("Predicted_Purchase", "sum"),
        Avg_Probability=("Purchase_Probability", "mean")
    ).reset_index()

    seg_perf["Actual_Rate_Pct"] = (seg_perf["Actual_Purchasers"] / seg_perf["Total_Customers"]) * 100
    seg_perf["Pred_Rate_Pct"] = (seg_perf["Predicted_Purchasers"] / seg_perf["Total_Customers"]) * 100

    # Melt for grouped bar chart
    melted_comparison = seg_perf.melt(
        id_vars=["Customer_Segment"],
        value_vars=["Actual_Purchasers", "Predicted_Purchasers"],
        var_name="Cohort",
        value_name="Count"
    )
    melted_comparison["Cohort"] = melted_comparison["Cohort"].replace({
        "Actual_Purchasers": "Actual Purchasers (Observed)",
        "Predicted_Purchasers": "Predicted Purchasers (Model)"
    })

    fig_comp = px.bar(
        melted_comparison,
        x="Customer_Segment",
        y="Count",
        color="Cohort",
        barmode="group",
        text="Count",
        title="Actual vs. Predicted 30-Day Purchasers across Segments",
        labels={"Count": "Number of Customers", "Customer_Segment": "Customer Segment"},
        color_discrete_map={
            "Actual Purchasers (Observed)": "#10B981",   # Emerald
            "Predicted Purchasers (Model)": "#3B82F6"    # Electric Blue
        }
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(yaxis=dict(range=[0, melted_comparison["Count"].max() * 1.18]))
    st.plotly_chart(apply_chart_theme(fig_comp, height=420), width="stretch")

    # Detailed conversion breakdown table
    with st.expander("📊 View Detailed Segment Purchase Conversion Statistics"):
        conv_display = seg_perf[[
            "Customer_Segment", "Total_Customers", "Actual_Purchasers", "Actual_Rate_Pct",
            "Predicted_Purchasers", "Pred_Rate_Pct", "Avg_Probability"
        ]].copy()
        conv_display.columns = [
            "Segment Name", "Total Customers", "Actual Buyers", "Actual Buy Rate (%)",
            "Predicted Buyers", "Predicted Buy Rate (%)", "Mean Predicted Prob (%)"
        ]
        st.dataframe(
            conv_display.style.format({
                "Total Customers": "{:,}",
                "Actual Buyers": "{:,}",
                "Actual Buy Rate (%)": "{:.1f}%",
                "Predicted Buyers": "{:,}",
                "Predicted Buy Rate (%)": "{:.1f}%",
                "Mean Predicted Prob (%)": "{:.1%}"
            }),
            width="stretch",
            hide_index=True
        )

    st.write("")

    # C. Purchase Likelihood by Segment
    st.markdown("### 📈 Purchase Likelihood (Propensity) by Customer Segment")
    st.caption("Average model-estimated probability of purchasing within the next 30 days.")

    seg_prob_sorted = seg_perf.sort_values(by="Avg_Probability", ascending=False)
    prob_colors = [SEGMENT_COLORS.get(s, "#3B82F6") for s in seg_prob_sorted["Customer_Segment"]]

    fig_prob = go.Figure(
        data=[
            go.Bar(
                x=seg_prob_sorted["Customer_Segment"],
                y=seg_prob_sorted["Avg_Probability"] * 100,
                marker=dict(color=prob_colors),
                text=seg_prob_sorted["Avg_Probability"].map(lambda x: f"{x:.1%}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Mean Purchase Likelihood: %{y:.1f}%<extra></extra>"
            )
        ]
    )
    fig_prob.update_layout(
        title="Average Purchase Probability by Customer Segment (%)",
        xaxis_title="Customer Segment",
        yaxis_title="Purchase Probability (%)",
        yaxis=dict(range=[0, 85])
    )
    st.plotly_chart(apply_chart_theme(fig_prob, height=360), width="stretch")

    st.write("")
    st.divider()

    # D. Strategic Business Recommendations
    st.markdown("### 🎯 Data-Driven Business Recommendations by Segment")
    st.caption("Actionable marketing strategies grounded in observed RFM profiles and predictive likelihoods.")

    rec_c1, rec_c2 = st.columns(2, gap="medium")

    with rec_c1:
        with st.container(border=True):
            st.markdown("#### 🌟 1. Best Customers (814 Profiles | £1.56M Spend | 69.2% Prob)")
            st.markdown(
                """
                - **Strategic Focus:** VIP Retention & Advocacy.
                - **Actionable Playbook:**
                  - Enroll into exclusive Tier-1 loyalty club with early access to limited releases and priority support.
                  - Implement a high-value customer concierge program.
                  - Provide threshold incentives on high-margin lines rather than broad discount codes to preserve gross margins.
                """
            )

        with st.container(border=True):
            st.markdown("#### 💎 2. Loyal Customers (456 Profiles | £356k Spend | 52.3% Prob)")
            st.markdown(
                """
                - **Strategic Focus:** Basket Expansion & Category Cross-Selling.
                - **Actionable Playbook:**
                  - Promote multi-item bundles and complementary category suggestions based on purchase history.
                  - Introduce milestone loyalty points ("Complete 5 orders this quarter to earn a £25 voucher").
                  - Trigger automated replenishment prompts calibrated to individual order cycle frequencies.
                """
            )

        with st.container(border=True):
            st.markdown("#### ⚠️ 3. At-Risk Valuable Customers (483 Profiles | £470k Spend | 54.1% Prob)")
            st.markdown(
                """
                - **Strategic Focus:** High-Priority Win-Back & Churn Mitigation.
                - **Actionable Playbook:**
                  - Urgent reactivation campaign: Send personalized 'We Miss You' emails with time-delimited vouchers.
                  - Deploy survey to discover service issues or unmet product needs before customer permanently churns.
                  - Re-engage across multi-channel touchpoints (SMS / email) with compelling incentives.
                """
            )

    with rec_c2:
        with st.container(border=True):
            st.markdown("#### 🛍️ 4. Regular Customers (523 Profiles | £155k Spend | 36.4% Prob)")
            st.markdown(
                """
                - **Strategic Focus:** Frequency Acceleration.
                - **Actionable Playbook:**
                  - Implement spending tier thresholds ("Spend £50, receive £10 off your next invoice").
                  - Gamify the purchasing journey to encourage transitions into the *Loyal* cohort.
                  - Highlight seasonal promotions and best-selling categories to spark repeat orders.
                """
            )

        with st.container(border=True):
            st.markdown("#### 💤 5. Inactive Customers (854 Profiles | £215k Spend | 31.5% Prob)")
            st.markdown(
                """
                - **Strategic Focus:** Low-Cost Automated Reactivation.
                - **Actionable Playbook:**
                  - Minimize expensive outbound ad spend; utilize automated email drip sequences.
                  - Highlight clearance events, seasonal discounts, and brand updates.
                  - If unengaged after 180 days, suppress from paid advertising to reduce customer acquisition/retention waste.
                """
            )

        with st.container(border=True):
            st.markdown("#### 🌱 6. Recent Customers (45 Profiles | £11.3k Spend | 31.7% Prob)")
            st.markdown(
                """
                - **Strategic Focus:** Post-Purchase Onboarding & Second-Order Conversion.
                - **Actionable Playbook:**
                  - Send automated post-purchase care and onboarding series within 7–14 days of receipt.
                  - Provide a time-sensitive second-order incentive to convert one-off shoppers into habitual buyers.
                  - Solicit product reviews and invite social community engagement.
                """
            )

    st.write("")

    # E. Historical Data Warning
    st.warning(
        "📌 **Historical Data Advisory:** These insights are based on historical transaction and model evaluation data. "
        "They do not represent live retail activity. Business recommendations are proposed operational strategies, "
        "not experimentally validated outcomes."
    )


# ============================================================
# END OF APPLICATION SCRIPT
# ============================================================
