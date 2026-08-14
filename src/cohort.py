"""Cohort/retention analysis computed with DuckDB SQL over the raw event log."""

import duckdb
import pandas as pd

_QUERY = """
    with cohorts as (
        select customer_id, date_trunc('month', min(event_date)) as cohort_month
        from events
        group by customer_id
    ),
    activity as (
        select e.customer_id, date_trunc('month', e.event_date) as activity_month, c.cohort_month
        from events e
        join cohorts c using (customer_id)
    ),
    periods as (
        select cohort_month, activity_month,
               datediff('month', cohort_month, activity_month) as period_number,
               count(distinct customer_id) as active_customers
        from activity
        group by cohort_month, activity_month
    ),
    cohort_sizes as (
        select cohort_month, count(distinct customer_id) as cohort_size
        from cohorts
        group by cohort_month
    )
    select p.cohort_month, p.period_number, p.active_customers, s.cohort_size,
           p.active_customers::double / s.cohort_size as retention_rate
    from periods p
    join cohort_sizes s using (cohort_month)
    order by p.cohort_month, p.period_number
"""


def build_cohort_retention_table(df: pd.DataFrame, id_col: str, date_col: str) -> pd.DataFrame:
    events = df[[id_col, date_col]].rename(columns={id_col: "customer_id", date_col: "event_date"}).copy()
    events["event_date"] = pd.to_datetime(events["event_date"])

    con = duckdb.connect()
    con.register("events", events)
    result = con.execute(_QUERY).df()
    con.close()
    return result
