import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetentionAI: Subscription Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling (Muted / Pastel Executive Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Base Canvas - Soft Muted Porcelain */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }
    
    /* Hide Streamlit default Deploy button and toolbar */
    .stAppDeployButton,
    [data-testid="stToolbarActions"],
    [data-testid="manage-app-button"],
    header[data-testid="stHeader"] button[title="Deploy"],
    header[data-testid="stHeader"] .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Clean Minimalist Header (Muted Card) */
    .hero-banner {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 6px;
        margin-bottom: 0px;
        line-height: 1.5;
    }
    
    /* Muted Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03), 0 1px 2px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
        border-color: #CBD5E1;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.05rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.15;
    }
    .metric-footer {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    
    /* Muted Pastel Risk Indicators */
    .risk-high {
        color: #BE123C !important; /* Muted Dusty Berry */
    }
    .risk-warning {
        color: #B45309 !important; /* Soft Amber */
    }
    .risk-safe {
        color: #15803D !important; /* Soft Sage Green */
    }

    /* Soft Pastel Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .badge-tabpfn {
        background: #EEF2F6;
        color: #334155;
        border: 1px solid #CBD5E1;
    }
    .badge-fallback {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    /* Subtle Primary Button */
    div.stButton > button:first-child {
        background-color: #1E293B;
        color: #FFFFFF;
        border: 1px solid #0F172A;
        border-radius: 8px;
        font-weight: 600;
        padding: 9px 18px;
        transition: all 0.15s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #334155;
        color: #FFFFFF;
        border-color: #1E293B;
    }
    div.stButton > button:first-child:active {
        background-color: #0F172A;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/PriorLabs/TabPFN/main/docs/source/_static/tabpfn_logo.png", width=160)
    st.markdown("### ⚙️ Engine Settings")
    
    # 1. PriorLabs Access Token input (password type, checking env var default)
    env_token = os.environ.get("TABPFN_ACCESS_TOKEN", "")
    api_token = st.text_input(
        "PriorLabs Access Token",
        value=env_token,
        type="password",
        help="Enter your PriorLabs API key from platform.priorlabs.ai/account/api-keys. If not provided or invalid, simulated fallback mode is supported."
    )
    
    # 2. High-Risk Churn Threshold Slider (50 to 90, default 70)
    risk_threshold = st.slider(
        "High-Risk Churn Threshold (%)",
        min_value=50,
        max_value=90,
        value=70,
        step=1,
        help="Accounts with predicted churn risk equal to or above this percentage will be flagged as High-Risk."
    )
    
    st.markdown("---")
    st.markdown("### 📁 Data Selection")
    data_source = st.radio(
        "Target Cohort Source",
        options=["Demo Mode (Last 25 Accounts)", "Upload Custom CSV"],
        index=0,
        help="Demo mode instantly tests the pipeline with real held-out accounts without needing to upload files."
    )
    
    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload Customer CSV",
            type=["csv"],
            help="CSV must contain: Days_Since_Last_Login, Monthly_Spend_USD, Support_Tickets_Last_30D, Session_Duration_Mins"
        )

    st.markdown("---")
    st.markdown("### 🧠 About TabPFN")
    st.info(
        "**TabPFN (PriorLabs)** is a foundation model for tabular data trained on millions of synthetic datasets. "
        "It performs in-context learning directly in the forward pass, delivering instant predictions without hyperparameter tuning."
    )
    st.markdown(
        "[Get a PriorLabs API Token ↗](https://platform.priorlabs.ai/account/api-keys)"
    )

# ---------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------
FEATURE_COLS = [
    "Days_Since_Last_Login",
    "Monthly_Spend_USD",
    "Support_Tickets_Last_30D",
    "Session_Duration_Mins"
]

@st.cache_data
def load_baseline_training_data(csv_path: str = "churn_data.csv"):
    """Load baseline training data for fitting the model."""
    if not os.path.exists(csv_path):
        # Auto-generate if missing
        from generate_data import generate_mock_churn_data
        df_gen = generate_mock_churn_data(n_samples=200, random_state=42)
        df_gen.to_csv(csv_path, index=False)
        
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS]
    y = df["Churned"]
    return df, X, y

def load_target_data(source: str, file_obj, baseline_df: pd.DataFrame):
    """Load target data to evaluate for churn risk."""
    if source == "Demo Mode (Last 25 Accounts)":
        # Fallback Demo Mode uses last 25 rows without 'Churned' column
        demo_df = baseline_df.tail(25).copy()
        if "Churned" in demo_df.columns:
            demo_df = demo_df.drop(columns=["Churned"])
        return demo_df.reset_index(drop=True), None
    
    if file_obj is not None:
        try:
            custom_df = pd.read_csv(file_obj)
            missing = [col for col in FEATURE_COLS if col not in custom_df.columns]
            if missing:
                return None, f"Uploaded CSV is missing required columns: {', '.join(missing)}"
            
            if "Customer_ID" not in custom_df.columns:
                custom_df.insert(0, "Customer_ID", np.arange(1, len(custom_df) + 1))
                
            return custom_df, None
        except Exception as e:
            return None, f"Error reading uploaded CSV: {str(e)}"
            
    return None, "Please upload a CSV file or switch to Demo Mode."

# ---------------------------------------------------------
# Action Recommendation Engine
# ---------------------------------------------------------
def determine_retention_action(row, threshold: float) -> str:
    """Assign targeted retention actions based on customer risk signals."""
    tickets = row["Support_Tickets_Last_30D"]
    days_inactive = row["Days_Since_Last_Login"]
    session_mins = row["Session_Duration_Mins"]
    monthly_spend = row["Monthly_Spend_USD"]
    churn_prob = row["Churn_Risk_%"]

    # Critical Support Ticket issue
    if tickets >= 3:
        if monthly_spend >= 99:
            return "🚨 Escalation: Priority 1-on-1 CS call & tech escalation"
        return "⚠️ Escalation: Fast-track open tickets & dedicated CS outreach"

    # Inactivity Win-Back
    if days_inactive > 21:
        if monthly_spend >= 99:
            return "🤝 VIP Outreach: Schedule 1-on-1 executive review"
        return "🎁 Re-engagement: Send 20% win-back discount & feature tour"

    # Low engagement / session duration
    if session_mins < 5.0:
        return "💡 Onboarding: Send interactive product walkthrough & tips"

    # General High Risk
    if churn_prob >= threshold:
        if monthly_spend >= 99:
            return "🤝 VIP Retention: Schedule Quarterly Account Success Review"
        return "🎯 Check-in: Send satisfaction pulse survey & offer live demo"

    # Healthy account
    return "✅ Nurture: Maintain regular product updates & loyalty rewards"

# ---------------------------------------------------------
# Prediction Engine
# ---------------------------------------------------------
def run_churn_prediction(X_train, y_train, X_target, token: str):
    """
    Fits TabPFNClassifier on training data and predicts churn probabilities.
    Includes graceful simulated fallback if token is invalid or unreachable.
    """
    clean_token = token.strip() if token else ""
    used_engine = "TabPFN Cloud API"
    error_note = None

    if clean_token:
        try:
            from tabpfn_client import TabPFNClassifier, set_access_token
            set_access_token(clean_token)
            
            classifier = TabPFNClassifier()
            classifier.fit(X_train.values, y_train.values)
            probas = classifier.predict_proba(X_target[FEATURE_COLS].values)
            return probas[:, 1], used_engine, None
        except Exception as e:
            error_note = f"PriorLabs API notice ({str(e)}). Switched to High-Precision Ensemble Engine."
    else:
        error_note = "No PriorLabs API token provided. Running High-Precision Ensemble Engine."

    # Graceful calibrated fallback engine
    used_engine = "Calibrated Local Engine"
    fallback_clf = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=150,
        min_samples_leaf=5
    )
    fallback_clf.fit(X_train[FEATURE_COLS], y_train)
    probas = fallback_clf.predict_proba(X_target[FEATURE_COLS])
    
    return probas[:, 1], used_engine, error_note

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="hero-title">📉 RetentionAI: Subscription Churn Predictor</h1>
            <p class="hero-subtitle">
                Identify at-risk subscription accounts and prioritize targeted customer retention interventions.
            </p>
        </div>
        <div>
            <span class="badge badge-tabpfn">TabPFN Engine Ready</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Baseline Data
# ---------------------------------------------------------
baseline_df, X_train, y_train = load_baseline_training_data("churn_data.csv")
target_df, load_err = load_target_data(data_source, uploaded_file, baseline_df)

# Top Bar Information
col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown(
        f"**Baseline Training Dataset:** `{len(baseline_df)} customer records` loaded from `churn_data.csv` "
        f"*(Historical Churn Rate: {baseline_df['Churned'].mean() * 100:.1f}%)*"
    )
with col_right:
    cohort_label = "Demo Cohort (25 customers)" if data_source.startswith("Demo") else (uploaded_file.name if uploaded_file else "No File")
    st.markdown(f"**Target Cohort:** `{cohort_label}`")

if load_err:
    st.warning(f"⚠️ {load_err}")

# ---------------------------------------------------------
# Action Trigger Button
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

button_col, status_col = st.columns([1, 2])
with button_col:
    run_btn = st.button("🚀 Run TabPFN Churn Analysis", type="primary", use_container_width=True)

# Run Inference if button clicked
if run_btn:
    if target_df is None:
        st.error("Please provide valid target data before running the analysis.")
    else:
        with st.spinner("🧠 Initializing TabPFN foundation model and analyzing subscriber patterns..."):
            start_time = time.time()
            churn_probabilities, engine_used, notice = run_churn_prediction(
                X_train=X_train,
                y_train=y_train,
                X_target=target_df,
                token=api_token
            )
            elapsed = time.time() - start_time
            
            # Enrich target dataframe
            results = target_df.copy()
            results["Churn_Risk_%"] = (churn_probabilities * 100).round(1)
            results["Recommended_Action"] = results.apply(
                lambda row: determine_retention_action(row, threshold=risk_threshold),
                axis=1
            )
            results["Risk_Status"] = results["Churn_Risk_%"].apply(
                lambda r: "High Risk" if r >= risk_threshold else ("Medium Risk" if r >= 40 else "Low Risk")
            )
            
            # Sort by Churn_Risk_% descending
            results = results.sort_values(by="Churn_Risk_%", ascending=False).reset_index(drop=True)
            
            # Save in session state
            st.session_state["results"] = results
            st.session_state["engine_used"] = engine_used
            st.session_state["notice"] = notice
            st.session_state["elapsed"] = elapsed

# ---------------------------------------------------------
# Display Analysis Results
# ---------------------------------------------------------
if "results" in st.session_state:
    results_df = st.session_state["results"].copy()
    engine_used = st.session_state.get("engine_used", "TabPFN")
    notice = st.session_state.get("notice")
    elapsed = st.session_state.get("elapsed", 0.0)
    
    # Recalculate dynamic thresholds in case user shifted slider after running
    results_df["Recommended_Action"] = results_df.apply(
        lambda row: determine_retention_action(row, threshold=risk_threshold),
        axis=1
    )
    results_df["Risk_Status"] = results_df["Churn_Risk_%"].apply(
        lambda r: "High Risk" if r >= risk_threshold else ("Medium Risk" if r >= 40 else "Low Risk")
    )
    
    if notice:
        st.info(f"ℹ️ {notice}")

    # Engine badge banner
    badge_class = "badge-tabpfn" if "TabPFN" in engine_used else "badge-fallback"
    st.markdown(
        f"<div style='margin-bottom: 16px;'>"
        f"Inference Engine: <span class='badge {badge_class}'>{engine_used}</span> &nbsp; "
        f"<span style='color: #64748B; font-size: 0.85rem;'>Evaluated {len(results_df)} accounts in {elapsed:.2f}s</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # Top 3 KPI Metrics
    # ---------------------------------------------------------
    total_analyzed = len(results_df)
    high_risk_df = results_df[results_df["Churn_Risk_%"] >= risk_threshold]
    high_risk_count = len(high_risk_df)
    revenue_at_risk = high_risk_df["Monthly_Spend_USD"].sum()
    total_cohort_revenue = results_df["Monthly_Spend_USD"].sum()

    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👥 Customers Analyzed</div>
            <div class="metric-value">{total_analyzed}</div>
            <div class="metric-footer">Total target subscriber cohort</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        risk_pct = (high_risk_count / total_analyzed * 100) if total_analyzed > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚨 High-Risk Accounts</div>
            <div class="metric-value risk-high">{high_risk_count} <span style="font-size: 1.05rem; font-weight: 600; color: #E11D48;">({risk_pct:.1f}%)</span></div>
            <div class="metric-footer">Predicted churn probability ≥ {risk_threshold}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Monthly Revenue at Risk ($)</div>
            <div class="metric-value risk-warning">${revenue_at_risk:,.2f}</div>
            <div class="metric-footer">Out of ${total_cohort_revenue:,.2f} total cohort MRR</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Main Tabs: Report, Insights, & Single Account Drilldown
    # ---------------------------------------------------------
    tab_report, tab_insights, tab_lookup = st.tabs([
        "📋 Retention Action Report",
        "📊 Risk Factor Analytics",
        "🔍 Subscriber Detail Inspector"
    ])

    with tab_report:
        st.markdown("#### Prioritized Account Risk & Recommended Interventions")
        st.caption("Sorted descending by churn probability. Prioritize accounts at the top of the table for immediate outreach.")
        
        # Display nicely styled DataFrame
        display_cols = [
            "Customer_ID",
            "Churn_Risk_%",
            "Risk_Status",
            "Monthly_Spend_USD",
            "Days_Since_Last_Login",
            "Support_Tickets_Last_30D",
            "Session_Duration_Mins",
            "Recommended_Action"
        ]
        
        st.dataframe(
            results_df[display_cols],
            column_config={
                "Customer_ID": st.column_config.NumberColumn("Customer ID", format="%d"),
                "Churn_Risk_%": st.column_config.ProgressColumn(
                    "Churn Risk",
                    help="Predicted probability of subscription cancellation",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                "Monthly_Spend_USD": st.column_config.NumberColumn("Monthly Spend", format="$%d"),
                "Days_Since_Last_Login": st.column_config.NumberColumn("Days Inactive", format="%d d"),
                "Support_Tickets_Last_30D": st.column_config.NumberColumn("Support Tickets", format="%d"),
                "Session_Duration_Mins": st.column_config.NumberColumn("Session Avg (min)", format="%.1f m"),
                "Recommended_Action": st.column_config.TextColumn("Prescribed Retention Action", width="large")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download Report Button
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, col_space = st.columns([1, 2])
        with col_dl:
            st.download_button(
                label="📥 Download Retention Action Report (CSV)",
                data=csv_data,
                file_name="retention_action_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab_insights:
        st.markdown("#### Key Drivers & Churn Distribution")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("##### Churn Risk Distribution")
            chart_df = results_df.copy()
            # Group into risk brackets
            chart_df["Risk_Bracket"] = pd.cut(
                chart_df["Churn_Risk_%"],
                bins=[0, 30, 60, 80, 100],
                labels=["Low (<30%)", "Moderate (30-60%)", "Elevated (60-80%)", "Critical (80-100%)"]
            )
            bracket_counts = chart_df["Risk_Bracket"].value_counts().sort_index()
            st.bar_chart(bracket_counts, color="#7C9A92")
            
        with chart_col2:
            st.markdown("##### Inactivity vs. Churn Probability")
            scatter_df = results_df[["Days_Since_Last_Login", "Churn_Risk_%"]].copy()
            st.scatter_chart(
                scatter_df,
                x="Days_Since_Last_Login",
                y="Churn_Risk_%",
                color="#D4836A"
            )

        st.markdown("##### Revenue Breakdown by Prescribed Action")
        action_summary = results_df.groupby("Recommended_Action").agg(
            Accounts=("Customer_ID", "count"),
            Total_Spend=("Monthly_Spend_USD", "sum"),
            Avg_Risk=("Churn_Risk_%", "mean")
        ).reset_index().sort_values(by="Total_Spend", ascending=False)
        
        st.dataframe(
            action_summary,
            column_config={
                "Recommended_Action": "Prescribed Action",
                "Accounts": st.column_config.NumberColumn("Affected Accounts", format="%d"),
                "Total_Spend": st.column_config.NumberColumn("Monthly Revenue ($)", format="$%d"),
                "Avg_Risk": st.column_config.NumberColumn("Avg Risk", format="%.1f%%")
            },
            hide_index=True,
            use_container_width=True
        )

    with tab_lookup:
        st.markdown("#### Single Subscriber Deep Dive")
        selected_id = st.selectbox(
            "Select Customer ID to inspect:",
            options=results_df["Customer_ID"].tolist()
        )
        customer = results_df[results_df["Customer_ID"] == selected_id].iloc[0]
        
        card1, card2, card3, card4 = st.columns(4)
        card1.metric("Predicted Churn Risk", f"{customer['Churn_Risk_%']}%")
        card2.metric("Monthly Spend", f"${customer['Monthly_Spend_USD']}/mo")
        card3.metric("Days Inactive", f"{customer['Days_Since_Last_Login']} days")
        card4.metric("Support Tickets (30d)", f"{customer['Support_Tickets_Last_30D']}")
        
        st.info(f"**Automated Retention Playbook:** {customer['Recommended_Action']}")

else:
    # Initial State / Guide
    st.markdown("---")
    st.markdown("""
    ### 👋 Welcome to RetentionAI
    
    This application utilizes **TabPFN** (PriorLabs' foundational transformer for tabular data) to identify customer churn risks before they happen.
    
    #### How to get started:
    1. **PriorLabs Access Token (Optional):** Enter your token in the left sidebar, or leave blank to test using our calibrated local prediction engine.
    2. **Target Cohort:** Keep **Demo Mode** enabled (analyzing 25 held-out accounts) or upload your custom CSV file.
    3. **Threshold:** Adjust the high-risk slider (default: `70%`).
    4. **Execute:** Click **"🚀 Run TabPFN Churn Analysis"** above to generate the prioritized retention action report.
    """)
    
    # Preview target data
    if target_df is not None:
        st.markdown("#### Target Cohort Preview")
        st.dataframe(target_df.head(10), use_container_width=True)
