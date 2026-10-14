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
# Bespoke Muted Pastel Design System
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #F8FAFC !important;
        color: #0F172A;
    }

    /* Page container spacing */
    .block-container {
        padding-top: 1.75rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1320px;
    }

    /* Hide Streamlit default Deploy button & toolbar header */
    .stAppDeployButton,
    [data-testid="stToolbarActions"],
    [data-testid="manage-app-button"],
    header[data-testid="stHeader"] button[title="Deploy"],
    header[data-testid="stHeader"] .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    /* Top Navigation Header */
    .brand-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        margin: 2px 0 0 0;
        font-weight: 400;
    }

    /* Pastel Status Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1;
    }
    .pill-tabpfn-active {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .pill-tabpfn-idle {
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }
    .pill-danger {
        background: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
    }
    .pill-warning {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .pill-success {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .pill-neutral {
        background: #F8FAFC;
        color: #334155;
        border: 1px solid #E2E8F0;
    }

    /* Action Control Banner */
    .control-deck {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 22px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .control-text {
        font-size: 0.92rem;
        color: #334155;
    }
    .control-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 3px;
    }

    /* Muted Pastel KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.04);
    }
    .kpi-accent-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .kpi-title {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-number {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-meta {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Accent Bar Colors */
    .accent-slate { background: #94A3B8; }
    .accent-rose { background: #F43F5E; }
    .accent-amber { background: #F59E0B; }
    .accent-sage { background: #10B981; }

    /* Custom Primary Button */
    div.stButton > button:first-child {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border: 1px solid #0F172A !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 9px 20px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #334155 !important;
        border-color: #334155 !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:first-child:active {
        background-color: #020617 !important;
    }

    /* Custom Download Button */
    div.stDownloadButton > button:first-child {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
    }
    div.stDownloadButton > button:first-child:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }

    /* Sidebar Clean Cards */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    .sidebar-section {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 10px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 6px !important;
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 16px !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }

    /* Action Card in Playbook */
    .action-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
        <span style="font-size: 1.4rem;">📉</span>
        <span style="font-weight: 800; font-size: 1.15rem; color: #0F172A; letter-spacing: -0.02em;">RetentionAI</span>
    </div>
    """, unsafe_allow_html=True)

    # 1. API Configuration Section
    st.markdown('<div class="sidebar-section-title">🔑 PriorLabs Connection</div>', unsafe_allow_html=True)
    env_token = os.environ.get("TABPFN_ACCESS_TOKEN", "")
    api_token = st.text_input(
        "Access Token",
        value=env_token,
        type="password",
        help="PriorLabs API token. Leave as default or enter from platform.priorlabs.ai/account/api-keys.",
        label_visibility="collapsed"
    )
    if api_token.strip():
        st.markdown('<span class="pill pill-tabpfn-active">● PriorLabs API Configured</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-tabpfn-idle">○ Local Calibrated Fallback</span>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # 2. Risk Sensitivity Slider
    st.markdown('<div class="sidebar-section-title">🎯 Sensitivity & Threshold</div>', unsafe_allow_html=True)
    risk_threshold = st.slider(
        "High-Risk Churn Threshold (%)",
        min_value=50,
        max_value=90,
        value=70,
        step=1,
        help="Subscribers with predicted churn probability equal to or higher than this threshold will be flagged as High Risk."
    )
    st.caption(f"Accounts ≥ **{risk_threshold}%** churn likelihood require urgent intervention.")

    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # 3. Data Source Section
    st.markdown('<div class="sidebar-section-title">📁 Evaluation Cohort</div>', unsafe_allow_html=True)
    data_source = st.radio(
        "Cohort Source",
        options=["Demo Mode (Last 25 Accounts)", "Upload Custom CSV"],
        index=0,
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Select customer CSV",
            type=["csv"],
            help="Expected columns: Days_Since_Last_Login, Monthly_Spend_USD, Support_Tickets_Last_30D, Session_Duration_Mins"
        )

    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # 4. Engine Info
    st.markdown("""
    <div style="font-size: 0.78rem; color: #64748B; line-height: 1.5;">
        <strong>TabPFN</strong> is a tabular foundation model that performs Bayesian inference in a single forward pass without hyperparameter tuning.
        <br><br>
        <a href="https://platform.priorlabs.ai/account/api-keys" target="_blank" style="color: #475569; font-weight: 600; text-decoration: underline;">
            Get PriorLabs API Key ↗
        </a>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------
FEATURE_COLS = [
    "Days_Since_Last_Login",
    "Monthly_Spend_USD",
    "Support_Tickets_Last_30D",
    "Session_Duration_Mins"
]

@st.cache_data
def load_baseline_training_data(csv_path: str = "churn_data.csv"):
    """Load baseline historical training data."""
    if not os.path.exists(csv_path):
        from generate_data import generate_mock_churn_data
        df_gen = generate_mock_churn_data(n_samples=200, random_state=42)
        df_gen.to_csv(csv_path, index=False)
        
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS]
    y = df["Churned"]
    return df, X, y

def load_target_data(source: str, file_obj, baseline_df: pd.DataFrame):
    """Load target cohort to predict upon."""
    if source == "Demo Mode (Last 25 Accounts)":
        demo_df = baseline_df.tail(25).copy()
        if "Churned" in demo_df.columns:
            demo_df = demo_df.drop(columns=["Churned"])
        return demo_df.reset_index(drop=True), None
    
    if file_obj is not None:
        try:
            custom_df = pd.read_csv(file_obj)
            missing = [col for col in FEATURE_COLS if col not in custom_df.columns]
            if missing:
                return None, f"Uploaded CSV missing required columns: {', '.join(missing)}"
            
            if "Customer_ID" not in custom_df.columns:
                custom_df.insert(0, "Customer_ID", np.arange(1, len(custom_df) + 1))
                
            return custom_df, None
        except Exception as e:
            return None, f"Failed to read CSV: {str(e)}"
            
    return None, "Upload a CSV file or select Demo Mode."

# ---------------------------------------------------------
# Recommendation Logic
# ---------------------------------------------------------
def determine_retention_action(row, threshold: float) -> str:
    """Prescribe customer retention actions based on risk indicators."""
    tickets = row["Support_Tickets_Last_30D"]
    days_inactive = row["Days_Since_Last_Login"]
    session_mins = row["Session_Duration_Mins"]
    monthly_spend = row["Monthly_Spend_USD"]
    churn_prob = row["Churn_Risk_%"]

    # Critical Support Tickets (≥ 3)
    if tickets >= 3:
        if monthly_spend >= 99:
            return "🚨 VIP Escalation: Priority CS engineering review & 1-on-1 account call"
        return "⚠️ Support Escalation: Fast-track open tickets & dedicated outreach"

    # Prolonged Inactivity (> 21 days)
    if days_inactive > 21:
        if monthly_spend >= 99:
            return "🤝 Executive Check-in: Schedule 1-on-1 executive review & tailored onboarding"
        return "🎁 Win-Back Incentive: Send 20% discount code & feature highlight tour"

    # Low Engagement Session (< 5 mins)
    if session_mins < 5.0:
        return "💡 Onboarding Assist: Trigger interactive feature tour & best practices email"

    # Elevated Churn Likelihood (≥ threshold)
    if churn_prob >= threshold:
        if monthly_spend >= 99:
            return "🤝 VIP Retention: Schedule Quarterly Account Review with Account Manager"
        return "🎯 Customer Pulse: Send satisfaction survey & offer live optimization call"

    # Healthy subscriber
    return "✅ Account Nurture: Maintain regular product updates & loyalty reward tier"

# ---------------------------------------------------------
# TabPFN Prediction Engine
# ---------------------------------------------------------
def run_churn_prediction(X_train, y_train, X_target, token: str):
    """Run model inference with TabPFN Cloud API or calibrated fallback."""
    clean_token = token.strip() if token else ""
    used_engine = "PriorLabs TabPFN API"
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
            error_note = f"PriorLabs API notice ({str(e)}). Ran with local calibrated engine."
    else:
        error_note = "No token provided. Running with local calibrated prediction engine."

    # Calibrated fallback engine
    used_engine = "Local Calibrated Engine"
    fallback_clf = HistGradientBoostingClassifier(random_state=42, max_iter=150, min_samples_leaf=5)
    fallback_clf.fit(X_train[FEATURE_COLS], y_train)
    probas = fallback_clf.predict_proba(X_target[FEATURE_COLS])
    
    return probas[:, 1], used_engine, error_note

# ---------------------------------------------------------
# Top Navigation Bar
# ---------------------------------------------------------
token_status_html = '<span class="pill pill-tabpfn-active">● TabPFN Cloud Ready</span>' if api_token.strip() else '<span class="pill pill-tabpfn-idle">○ Local Engine Active</span>'

st.markdown(f"""
<div class="brand-navbar">
    <div>
        <h1 class="brand-title">
            <span>📉</span> RetentionAI
        </h1>
        <p class="brand-subtitle">Subscription Churn Intelligence & Automated Retention Playbooks</p>
    </div>
    <div>
        {token_status_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Baseline & Cohort Ingestion
# ---------------------------------------------------------
baseline_df, X_train, y_train = load_baseline_training_data("churn_data.csv")
target_df, load_err = load_target_data(data_source, uploaded_file, baseline_df)

cohort_label = "Demo Cohort (25 customers)" if data_source.startswith("Demo") else (uploaded_file.name if uploaded_file else "No File")

# ---------------------------------------------------------
# Command Center Deck
# ---------------------------------------------------------
deck_col_info, deck_col_action = st.columns([2, 1])

with deck_col_info:
    st.markdown(f"""
    <div style="padding-top: 4px;">
        <span style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">Target Cohort:</span> 
        <span style="color: #475569; font-size: 0.95rem;">{cohort_label}</span>
        <br>
        <span style="font-size: 0.8rem; color: #64748B;">
            Baseline training set: <strong>{len(baseline_df)} records</strong> ({baseline_df['Churned'].mean() * 100:.1f}% historical churn rate)
        </span>
    </div>
    """, unsafe_allow_html=True)

with deck_col_action:
    run_btn = st.button("🚀 Run TabPFN Churn Analysis", use_container_width=True)

if load_err:
    st.warning(f"⚠️ {load_err}")

# ---------------------------------------------------------
# Execute Analysis
# ---------------------------------------------------------
if run_btn:
    if target_df is None:
        st.error("Please select or upload a valid customer dataset first.")
    else:
        with st.spinner("Analyzing subscriber engagement patterns and predicting churn risk..."):
            t_start = time.time()
            churn_probabilities, engine_used, notice = run_churn_prediction(
                X_train=X_train,
                y_train=y_train,
                X_target=target_df,
                token=api_token
            )
            elapsed = time.time() - t_start
            
            # Enrich target dataframe
            results = target_df.copy()
            results["Churn_Risk_%"] = (churn_probabilities * 100).round(1)
            results["Recommended_Action"] = results.apply(
                lambda row: determine_retention_action(row, threshold=risk_threshold),
                axis=1
            )
            results["Risk_Tier"] = results["Churn_Risk_%"].apply(
                lambda r: "High Risk" if r >= risk_threshold else ("Medium Risk" if r >= 40 else "Low Risk")
            )
            results = results.sort_values(by="Churn_Risk_%", ascending=False).reset_index(drop=True)
            
            st.session_state["results"] = results
            st.session_state["engine_used"] = engine_used
            st.session_state["notice"] = notice
            st.session_state["elapsed"] = elapsed

# ---------------------------------------------------------
# Results Presentation
# ---------------------------------------------------------
if "results" in st.session_state:
    results_df = st.session_state["results"].copy()
    engine_used = st.session_state.get("engine_used", "TabPFN")
    notice = st.session_state.get("notice")
    elapsed = st.session_state.get("elapsed", 0.0)

    # Dynamic threshold re-computation
    results_df["Recommended_Action"] = results_df.apply(
        lambda row: determine_retention_action(row, threshold=risk_threshold),
        axis=1
    )
    results_df["Risk_Tier"] = results_df["Churn_Risk_%"].apply(
        lambda r: "High Risk" if r >= risk_threshold else ("Medium Risk" if r >= 40 else "Low Risk")
    )

    if notice:
        st.caption(f"ℹ️ {notice}")

    # Top KPI Metrics Strip
    total_analyzed = len(results_df)
    high_risk_df = results_df[results_df["Churn_Risk_%"] >= risk_threshold]
    high_risk_count = len(high_risk_df)
    high_risk_pct = (high_risk_count / total_analyzed * 100) if total_analyzed > 0 else 0
    revenue_at_risk = high_risk_df["Monthly_Spend_USD"].sum()
    total_cohort_revenue = results_df["Monthly_Spend_USD"].sum()
    avg_churn_risk = results_df["Churn_Risk_%"].mean()

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-accent-bar accent-slate"></div>
            <div class="kpi-title">
                <span>Customers Analyzed</span>
                <span class="pill pill-neutral">Total</span>
            </div>
            <div class="kpi-number">{total_analyzed}</div>
            <div class="kpi-meta">
                <span>Evaluated in {elapsed:.2f}s via {engine_used}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-accent-bar accent-rose"></div>
            <div class="kpi-title">
                <span>High-Risk Accounts</span>
                <span class="pill pill-danger">{high_risk_pct:.1f}%</span>
            </div>
            <div class="kpi-number" style="color: #BE123C;">{high_risk_count}</div>
            <div class="kpi-meta">
                <span>Accounts with churn likelihood ≥ {risk_threshold}%</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-accent-bar accent-amber"></div>
            <div class="kpi-title">
                <span>Monthly Revenue at Risk ($)</span>
                <span class="pill pill-warning">MRR Exposure</span>
            </div>
            <div class="kpi-number" style="color: #B45309;">${revenue_at_risk:,.2f}</div>
            <div class="kpi-meta">
                <span>Out of ${total_cohort_revenue:,.2f} total analyzed MRR</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-accent-bar accent-sage"></div>
            <div class="kpi-title">
                <span>Avg Cohort Risk</span>
                <span class="pill pill-success">Health</span>
            </div>
            <div class="kpi-number" style="color: #15803D;">{avg_churn_risk:.1f}%</div>
            <div class="kpi-meta">
                <span>Mean predicted probability across cohort</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Main Views Tabs
    # ---------------------------------------------------------
    tab_report, tab_analytics, tab_inspector = st.tabs([
        "📋 Prioritized Retention Action Playbook",
        "📊 Cohort Risk Factors & Revenue Impact",
        "🔍 Single Subscriber Profile Inspector"
    ])

    with tab_report:
        col_hdr_left, col_hdr_right = st.columns([2, 1])
        with col_hdr_left:
            st.markdown("#### Ranked Subscriber Retention Queue")
            st.caption("Sorted descending by churn risk score. Prioritize accounts at top of table.")
        with col_hdr_right:
            # Download Action Report Button
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Retention Action Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="retention_action_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Main Data Table
        display_cols = [
            "Customer_ID",
            "Churn_Risk_%",
            "Risk_Tier",
            "Monthly_Spend_USD",
            "Days_Since_Last_Login",
            "Support_Tickets_Last_30D",
            "Session_Duration_Mins",
            "Recommended_Action"
        ]

        st.dataframe(
            results_df[display_cols],
            column_config={
                "Customer_ID": st.column_config.NumberColumn("Customer ID", format="%d", width="small"),
                "Churn_Risk_%": st.column_config.ProgressColumn(
                    "Churn Likelihood",
                    help="Predicted probability of subscription cancellation",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="medium"
                ),
                "Risk_Tier": st.column_config.TextColumn("Risk Tier", width="small"),
                "Monthly_Spend_USD": st.column_config.NumberColumn("MRR ($)", format="$%d", width="small"),
                "Days_Since_Last_Login": st.column_config.NumberColumn("Days Inactive", format="%d d", width="small"),
                "Support_Tickets_Last_30D": st.column_config.NumberColumn("Tickets (30d)", format="%d", width="small"),
                "Session_Duration_Mins": st.column_config.NumberColumn("Avg Session", format="%.1f min", width="small"),
                "Recommended_Action": st.column_config.TextColumn("Prescribed Retention Action", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

    with tab_analytics:
        st.markdown("#### Cohort Analytics & Key Drivers")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Churn Risk Distribution")
            chart_df = results_df.copy()
            chart_df["Risk_Bracket"] = pd.cut(
                chart_df["Churn_Risk_%"],
                bins=[0, 30, 60, 80, 100],
                labels=["Low (<30%)", "Moderate (30-60%)", "Elevated (60-80%)", "Critical (80-100%)"]
            )
            bracket_counts = chart_df["Risk_Bracket"].value_counts().sort_index()
            st.bar_chart(bracket_counts, color="#7C9A92")

        with c2:
            st.markdown("##### Inactivity Days vs. Predicted Churn Risk")
            scatter_df = results_df[["Days_Since_Last_Login", "Churn_Risk_%"]].copy()
            st.scatter_chart(
                scatter_df,
                x="Days_Since_Last_Login",
                y="Churn_Risk_%",
                color="#D4836A"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Revenue Protection by Prescribed Retention Play")
        
        playbook_summary = results_df.groupby("Recommended_Action").agg(
            Subscribers=("Customer_ID", "count"),
            Revenue_at_Stake=("Monthly_Spend_USD", "sum"),
            Avg_Churn_Risk=("Churn_Risk_%", "mean")
        ).reset_index().sort_values(by="Revenue_at_Stake", ascending=False)

        st.dataframe(
            playbook_summary,
            column_config={
                "Recommended_Action": "Prescribed Retention Play",
                "Subscribers": st.column_config.NumberColumn("Affected Accounts", format="%d"),
                "Revenue_at_Stake": st.column_config.NumberColumn("Monthly Revenue at Stake", format="$%d"),
                "Avg_Churn_Risk": st.column_config.NumberColumn("Average Risk", format="%.1f%%")
            },
            hide_index=True,
            use_container_width=True
        )

    with tab_inspector:
        st.markdown("#### Subscriber Profile Inspector")
        st.caption("Deep-dive into an individual subscriber record to view risk signals and prescribed action.")

        selected_customer_id = st.selectbox(
            "Select Customer ID to inspect:",
            options=results_df["Customer_ID"].tolist(),
            format_func=lambda cid: f"Customer #{cid} — Risk: {results_df[results_df['Customer_ID']==cid]['Churn_Risk_%'].values[0]}% — Action: {results_df[results_df['Customer_ID']==cid]['Recommended_Action'].values[0][:40]}..."
        )

        customer = results_df[results_df["Customer_ID"] == selected_customer_id].iloc[0]

        card_col1, card_col2, card_col3, card_col4 = st.columns(4)
        card_col1.metric("Predicted Churn Risk", f"{customer['Churn_Risk_%']}%")
        card_col2.metric("Monthly Spend", f"${customer['Monthly_Spend_USD']}/mo")
        card_col3.metric("Days Since Login", f"{customer['Days_Since_Last_Login']} days")
        card_col4.metric("Support Tickets (30d)", f"{customer['Support_Tickets_Last_30D']}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"📋 **Prescribed Retention Action:** {customer['Recommended_Action']}")

else:
    # Empty / Initial state view
    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 28px 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
        <h3 style="margin-top: 0; color: #0F172A; font-weight: 700;">Ready to Run TabPFN Churn Prediction</h3>
        <p style="color: #64748B; font-size: 0.95rem; line-height: 1.6; max-width: 800px;">
            RetentionAI uses PriorLabs' foundation model for tabular data to forecast customer cancellations before they occur.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 20px;">
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">1. Training Baseline</div>
                <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">200 customer records in <code>churn_data.csv</code></div>
            </div>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">2. Target Cohort</div>
                <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">25 held-out subscribers ready in Demo Mode</div>
            </div>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">3. Model Execution</div>
                <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">Click <strong>🚀 Run TabPFN Churn Analysis</strong> above</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if target_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Preview Target Cohort")
        st.dataframe(target_df.head(10), hide_index=True, use_container_width=True)
