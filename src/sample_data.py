"""Synthetic behavioral/transactional event log used for the 'try it with sample data' mode."""

import numpy as np
import pandas as pd

PLANS = np.array(["Basic", "Pro", "Enterprise"])
PLAN_WEIGHTS = np.array([0.55, 0.35, 0.10])
PLAN_MONTHLY_CHURN = {"Basic": 0.09, "Pro": 0.05, "Enterprise": 0.02}
PLAN_AMOUNT_RANGE = {"Basic": (9, 29), "Pro": (49, 99), "Enterprise": (199, 499)}


def generate_sample_data(n_customers: int = 1500, months_of_history: int = 18, seed: int = 42) -> pd.DataFrame:
    """Return an event-level log: one row per usage/transaction event, with a realistic churn pattern baked in."""
    rng = np.random.default_rng(seed)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(months=months_of_history)
    horizon_days = (end_date - start_date).days

    signup_offsets = rng.integers(0, horizon_days, size=n_customers)
    customer_plans = rng.choice(PLANS, size=n_customers, p=PLAN_WEIGHTS)

    records = []
    for i in range(n_customers):
        customer_id = f"CUST{i:05d}"
        signup_date = start_date + pd.Timedelta(days=int(signup_offsets[i]))
        plan = customer_plans[i]
        monthly_churn_p = PLAN_MONTHLY_CHURN[plan]
        amount_lo, amount_hi = PLAN_AMOUNT_RANGE[plan]

        for month_start in pd.date_range(signup_date, end_date, freq="MS"):
            if month_start > signup_date and rng.random() < monthly_churn_p:
                break  # customer churns: no further activity generated
            n_events = rng.integers(1, 9)
            for day_offset in rng.integers(0, 28, size=n_events):
                event_date = month_start + pd.Timedelta(days=int(day_offset))
                if event_date > end_date:
                    continue
                amount = round(float(rng.uniform(amount_lo, amount_hi)), 2)
                records.append((customer_id, event_date, amount, plan))

    df = pd.DataFrame(records, columns=["customer_id", "event_date", "amount", "plan_tier"])
    return df.sort_values("event_date").reset_index(drop=True)
