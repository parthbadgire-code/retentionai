# 📉 RetentionAI: Subscription Churn Predictor

An enterprise-grade, Render-ready Streamlit web application powered by **TabPFN** (PriorLabs' foundational transformer model for tabular datasets) to predict subscriber churn risk and automate retention actions.

---

## 🌟 Key Features

- **TabPFN Client Integration:** Direct connection to PriorLabs' in-context tabular foundation model via `tabpfn-client`.
- **Zero-Config Demo Mode:** Ships with 200 realistic synthetic subscription customer records and a default 25-account evaluation cohort so it functions right out of the box.
- **Custom CSV Ingestion:** Upload any subscriber dataset with support for custom customer IDs and dynamic column mapping.
- **Dynamic Executive KPIs:**
  - 👥 **Customers Analyzed**
  - 🚨 **High-Risk Accounts** (with dynamic risk threshold)
  - 💰 **Monthly Revenue at Risk ($)**
- **Automated Retention Playbook:** Prescribes targeted business actions (e.g. 20% win-back discounts for inactivity >21 days, urgent CS escalation for support tickets ≥3, VIP executive calls for high-spend tiers).
- **Interactive Visualizations:** Churn distribution histograms, inactivity vs. churn scatter charts, and revenue breakdown by intervention.
- **Export Ready:** One-click CSV export of prioritized customer retention reports.
- **Cloud & Render Deployable:** Includes `Procfile`, `render.yaml`, and `.streamlit/config.toml` preconfigured for immediate deployment.

---

## 📋 Required Dataset Columns

Whether using the included `churn_data.csv` or uploading custom data, the following columns are supported:

| Column | Type | Description |
|---|---|---|
| `Customer_ID` | Integer | Unique identifier for subscriber |
| `Days_Since_Last_Login` | Integer (1–45) | Days elapsed since subscriber last accessed platform |
| `Monthly_Spend_USD` | Integer ($15, $30, $99) | Subscription tier / Monthly Recurring Revenue (MRR) |
| `Support_Tickets_Last_30D` | Integer (Poisson λ=1.5) | Frequency of support complaints/inquiries |
| `Session_Duration_Mins` | Float (1.0–60.0) | Average session engagement time in minutes |
| `Churned` | Integer (0 or 1) | Historical ground truth (used for baseline training) |

---

## 🚀 Local Quickstart

### 1. Set Up Environment & Install Dependencies

```bash
cd retentionai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data

```bash
python generate_data.py
```
This generates `churn_data.csv` containing 200 records with realistic churn distributions.

### 3. Run Streamlit Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Render

This repository is pre-configured for seamless deployment to [Render](https://render.com).

### Option 1: Render Blueprints (Recommended)
1. Push your repository to GitHub / GitLab.
2. Log in to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** > **Blueprint**.
4. Connect your repository. Render will automatically detect [`render.yaml`](render.yaml) and configure:
   - **Build Command:** `pip install -r requirements.txt && python generate_data.py`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. (Optional) Set `TABPFN_ACCESS_TOKEN` under Environment Variables.

### Option 2: Manual Web Service
1. In Render, select **New Web Service**.
2. Set Runtime to **Python**.
3. **Build Command:** `pip install -r requirements.txt && python generate_data.py`
4. **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service**.

---

## 🔑 PriorLabs API Key
Get your free API key at [platform.priorlabs.ai/account/api-keys](https://platform.priorlabs.ai/account/api-keys) and paste it into the sidebar input, or set it as an environment variable:
```bash
export TABPFN_ACCESS_TOKEN="your_token_here"
```
*(If no token is supplied, RetentionAI runs seamlessly using its calibrated local prediction engine for instant testing.)*
