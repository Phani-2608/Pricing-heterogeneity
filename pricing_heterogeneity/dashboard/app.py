"""Streamlit dashboard: interactive decision simulator.

Run with:  streamlit run pricing_heterogeneity/dashboard/app.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from ..config import DEFAULT_CONFIG
from ..models.registry import ModelRegistry
from ..optimization.policy_engine import decide_for_customer

st.set_page_config(page_title="Pricing Heterogeneity", layout="wide")


@st.cache_data
def load_results(path: str = "outputs/results.json") -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_registered_model(registry_dir: str = "outputs/registry"):
    reg = ModelRegistry(registry_dir)
    v = reg.latest_version()
    if not v:
        return None, None, None
    art, meta = reg.load(v)
    return art["cate_model"], art["feature_builder"], meta


def _kpi(col, label, value, delta=None):
    col.metric(label=label, value=value, delta=delta)


def _sidebar_customer_form(fb) -> pd.DataFrame:
    st.sidebar.header("Decision Simulator")
    st.sidebar.caption("Score a hypothetical customer and see the policy recommendation.")
    return pd.DataFrame([{
        "customer_id": "SIM-001",
        "age": st.sidebar.slider("Age", 18, 78, 34),
        "ltv": st.sidebar.slider("LTV ($)", 20, 1500, 320),
        "purchase_freq": st.sidebar.slider("Purchase frequency", 0, 20, 5),
        "tenure_months": st.sidebar.slider("Tenure (months)", 0, 120, 18),
        "prior_discounts": st.sidebar.slider("Prior discounts used", 0, 6, 2),
        "sessions_30d": st.sidebar.slider("Sessions (30d)", 0, 30, 8),
        "cohort_month": st.sidebar.slider("Cohort month", 1, 12, 6),
        "category": st.sidebar.selectbox("Category", ["Electronics", "Apparel", "Home", "Beauty"]),
        "region": st.sidebar.selectbox("Region", ["North", "South", "East", "West"]),
        "ltv_was_missing": 0,
        "tenure_months_was_missing": 0,
        "sessions_30d_was_missing": 0,
    }])


def main():
    st.title("Heterogeneous Treatment Effects in Customer Pricing")
    results = load_results()
    if results is None:
        st.error("`outputs/results.json` not found. Run `python -m pricing_heterogeneity.pipeline` first.")
        return

    strat_df = pd.DataFrame(results["strategy_comparison"])
    seg_df = pd.DataFrame(results["segments"])
    fair_df = pd.DataFrame(results["fairness_by_group"])
    fi_df = pd.DataFrame(results.get("feature_importance", []))
    upl_df = load_csv("outputs/uplift_curve.csv")

    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, "Customers", f"{results['n_customers']:,}")
    _kpi(c2, "DR ATE", f"{results['ate_dr']:+.4f}",
         delta=f"true {results['true_ate']:+.4f}")
    _kpi(c3, "Best strategy lift vs treat-all",
         f"{results['best_strategy_lift_vs_treat_all']:+.1%}")
    _kpi(c4, "Validation", f"{results['tests_passed']}/{results['tests_total']}")

    tab_overview, tab_strategy, tab_segments, tab_fairness, tab_sim = st.tabs(
        ["Overview", "Strategy comparison", "Segments", "Fairness", "Decision simulator"]
    )

    with tab_overview:
        st.subheader("ATE and confidence intervals")
        ci = results["ate_dr_ci95"]
        boot = results.get("ate_dr_bootstrap_ci95", [None, None])
        st.write(
            f"DR ATE **{results['ate_dr']:+.4f}**  ·  "
            f"analytic 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  ·  "
            f"bootstrap 95% CI [{boot[0]:+.4f}, {boot[1]:+.4f}]"
        )
        st.subheader("Model quality")
        st.write({
            "CATE vs truth correlation": round(results["cate_corr_with_truth_dr"], 3),
            "GATES calibration slope": round(results["calibration_slope"], 3),
            "GATES rank correlation": round(results["gates_rank_correlation"], 3),
            "Qini": round(results["qini"], 4),
        })
        if not fi_df.empty:
            st.subheader("Global feature importance")
            st.bar_chart(fi_df.set_index("feature"))

    with tab_strategy:
        st.subheader("Business KPIs by targeting strategy")
        st.dataframe(strat_df, use_container_width=True)
        st.bar_chart(strat_df.set_index("strategy")["total_profit"])
        if upl_df is not None:
            st.subheader("Uplift curve (higher = better ranking)")
            st.line_chart(upl_df.set_index("targeting_depth"))

    with tab_segments:
        st.subheader("Segment-level treatment response")
        st.dataframe(seg_df, use_container_width=True)
        st.bar_chart(seg_df.set_index("segment")["est_cate"])

    with tab_fairness:
        st.subheader("Fairness audit")
        st.write(
            f"Disparate-impact ratio: **{results['fairness_disparity_ratio']:.3f}** "
            f"(four-fifths threshold 0.80)"
        )
        st.dataframe(fair_df, use_container_width=True)
        if results["fairness_disparity_ratio"] < 0.80:
            st.warning(
                "Policy fails the four-fifths rule across age groups. "
                "Requires either removing age (and proxies) from the policy or "
                "documented business necessity and legal review."
            )

    with tab_sim:
        model, fb, meta = load_registered_model()
        if model is None:
            st.info("No trained model in `outputs/registry/`. Train first to enable the simulator.")
            return
        row = _sidebar_customer_form(fb)
        X = fb.transform(row)
        pred = float(model.predict(X)[0])
        baseline = st.sidebar.slider("Baseline conversion probability", 0.05, 0.60, 0.20, 0.01)

        decision = decide_for_customer(
            customer_id=row.iloc[0]["customer_id"],
            predicted_ite=pred,
            baseline_p=baseline,
            ite_se=None,
            customer_value=float(row.iloc[0]["ltv"]),
            cfg=DEFAULT_CONFIG,
            model_version=meta.model_version,
        )
        st.subheader("Recommendation")
        st.json(decision.as_dict())


if __name__ == "__main__":
    main()
