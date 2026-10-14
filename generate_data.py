import numpy as np
import pandas as pd

def generate_mock_churn_data(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic mock subscription customer records for churn prediction."""
    np.random.seed(random_state)
    
    customer_ids = np.arange(1, n_samples + 1)
    days_since_last_login = np.random.randint(1, 46, size=n_samples)
    monthly_spend = np.random.choice([15, 30, 99], size=n_samples, p=[0.50, 0.35, 0.15])
    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    session_duration = np.clip(np.random.normal(loc=15, scale=5, size=n_samples), 1, 60).round(1)
    
    # Calculate realistic churn probability based on risk indicators
    # Base risk around 8%
    base_logits = -2.4
    
    # Risk factor weights
    inactive_factor = (days_since_last_login > 21).astype(float) * 1.8
    ticket_factor = (support_tickets > 3).astype(float) * 1.5
    low_session_factor = (session_duration < 5.0).astype(float) * 1.3
    
    # Slight interaction for very low engagement + high days
    extra_risk = ((days_since_last_login > 30) & (support_tickets >= 2)).astype(float) * 0.8
    
    # Random noise to simulate real-world variance
    noise = np.random.normal(loc=0, scale=0.3, size=n_samples)
    
    logits = base_logits + inactive_factor + ticket_factor + low_session_factor + extra_risk + noise
    churn_prob = 1.0 / (1.0 + np.exp(-logits))
    
    # Clip probability between 0.02 and 0.95 for realistic distributions
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    
    churned = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Days_Since_Last_Login": days_since_last_login,
        "Monthly_Spend_USD": monthly_spend,
        "Support_Tickets_Last_30D": support_tickets,
        "Session_Duration_Mins": session_duration,
        "Churned": churned
    })
    
    return df

if __name__ == "__main__":
    df = generate_mock_churn_data(n_samples=200, random_state=42)
    output_path = "churn_data.csv"
    df.to_csv(output_path, index=False)
    churn_rate = df["Churned"].mean() * 100
    print(f"Generated {len(df)} records saved to {output_path}")
    print(f"Overall Churn Rate: {churn_rate:.1f}% ({df['Churned'].sum()} churned)")
    print(df.head())
