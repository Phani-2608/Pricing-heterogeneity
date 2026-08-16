"""SQL layer (sqlite3): realistic multi-table schema and query patterns.

Models the problem the way it actually looks in a warehouse: separate
customers, transactions, pricing_experiments, and treatment_history tables,
joined and aggregated with CTEs and window functions rather than one flat
pandas DataFrame pretending to be a database.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


def build_star_schema(df: pd.DataFrame, rng: np.random.Generator) -> sqlite3.Connection:
    """Decompose the flat customer table into a small realistic star schema."""
    conn = sqlite3.connect(":memory:")
    n = len(df)

    customers = df[
        ["age", "ltv", "purchase_freq", "tenure_months", "category", "region"]
    ].copy()
    customers.insert(0, "customer_id", range(1, n + 1))
    customers.to_sql("customers", conn, index=False)

    treatment_history = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "cohort_month": df["cohort_month"].values,
            "treated": df["treated"].values,
            "prior_discounts": df["prior_discounts"].values,
        }
    )
    treatment_history.to_sql("treatment_history", conn, index=False)

    pricing_experiments = pd.DataFrame(
        {
            "experiment_id": 1,
            "customer_id": range(1, n + 1),
            "discount_depth": np.where(df["treated"].values == 1, 0.15, 0.0),
            "propensity": df.get("propensity", pd.Series(np.nan, index=df.index)).values,
        }
    )
    pricing_experiments.to_sql("pricing_experiments", conn, index=False)

    n_txn = int(n * 1.4)
    transactions = pd.DataFrame(
        {
            "txn_id": range(1, n_txn + 1),
            "customer_id": rng.integers(1, n + 1, n_txn),
            "converted": rng.binomial(1, df["converted"].mean(), n_txn),
            "order_value": rng.lognormal(4.2, 0.6, n_txn).round(2),
            "txn_month": rng.integers(1, 13, n_txn),
        }
    )
    transactions.to_sql("transactions", conn, index=False)

    return conn


SQL_COHORT_LIFT = """
SELECT
    th.cohort_month,
    COUNT(DISTINCT c.customer_id)                                 AS n_customers,
    SUM(th.treated)                                                AS n_treated,
    ROUND(AVG(CASE WHEN th.treated = 1 THEN t.converted END), 4)   AS conv_treated,
    ROUND(AVG(CASE WHEN th.treated = 0 THEN t.converted END), 4)   AS conv_control,
    ROUND(
        AVG(CASE WHEN th.treated = 1 THEN t.converted END)
        - AVG(CASE WHEN th.treated = 0 THEN t.converted END), 4
    )                                                               AS naive_lift
FROM customers c
JOIN treatment_history th ON th.customer_id = c.customer_id
JOIN transactions t       ON t.customer_id = c.customer_id
GROUP BY th.cohort_month
HAVING COUNT(*) > 100
ORDER BY th.cohort_month;
"""

SQL_CATEGORY_REGION = """
SELECT
    category,
    region,
    COUNT(*)                     AS n_customers,
    ROUND(AVG(ltv), 2)           AS avg_ltv,
    ROUND(AVG(purchase_freq), 2) AS avg_purchase_freq
FROM customers
GROUP BY category, region
ORDER BY avg_ltv DESC
LIMIT 8;
"""

# CTE + window function: customer-level running order count and rank by LTV
# within their region, joined back to treatment status.
SQL_CUSTOMER_WINDOW = """
WITH ranked_customers AS (
    SELECT
        c.customer_id,
        c.region,
        c.ltv,
        th.treated,
        ROW_NUMBER() OVER (PARTITION BY c.region ORDER BY c.ltv DESC) AS ltv_rank_in_region,
        NTILE(4) OVER (ORDER BY c.ltv)                                 AS ltv_quartile
    FROM customers c
    JOIN treatment_history th ON th.customer_id = c.customer_id
),
txn_counts AS (
    SELECT customer_id, COUNT(*) AS n_transactions, SUM(order_value) AS total_spend
    FROM transactions
    GROUP BY customer_id
)
SELECT
    rc.region,
    rc.ltv_quartile,
    COUNT(*)                                AS n_customers,
    ROUND(AVG(rc.treated), 3)               AS treat_rate,
    ROUND(AVG(COALESCE(tc.n_transactions, 0)), 2) AS avg_transactions,
    ROUND(AVG(COALESCE(tc.total_spend, 0)), 2)    AS avg_spend
FROM ranked_customers rc
LEFT JOIN txn_counts tc ON tc.customer_id = rc.customer_id
GROUP BY rc.region, rc.ltv_quartile
ORDER BY rc.region, rc.ltv_quartile;
"""


def run_aggregations(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Run the cohort, category/region, and windowed customer-value queries."""
    return {
        "cohort_lift": pd.read_sql_query(SQL_COHORT_LIFT, conn),
        "category_region": pd.read_sql_query(SQL_CATEGORY_REGION, conn),
        "customer_window": pd.read_sql_query(SQL_CUSTOMER_WINDOW, conn),
    }
