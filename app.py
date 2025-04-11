import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

st.set_page_config(page_title="Incrementality Test", layout="wide")
st.title("🧪 Incrementality Test Power Analysis")

# Sidebar Inputs
st.sidebar.header("📥 Upload Your Metric Files")
revenue_file = st.sidebar.file_uploader("Primary Metric: New Customer Revenue", type=["csv"])
orders_file = st.sidebar.file_uploader("Secondary Metric: New Customer Orders", type=["csv"])

st.sidebar.header("⚙️ Test Configuration")
weekly_budget = st.sidebar.number_input("Weekly Budget ($)", value=500)
budget_increase_pct = st.sidebar.slider("Budget Increase (%)", 0, 100, 20)
test_weeks = st.sidebar.slider("Test Duration (weeks)", 1, 12, 4)
holdout_pct = st.sidebar.slider("Holdout Share (%)", 0, 50, 20)
alpha = st.sidebar.number_input("Significance Level (α)", value=0.1)
power = st.sidebar.number_input("Statistical Power", value=0.8)

if revenue_file and orders_file:
    # Load data
    revenue_df = pd.read_csv(revenue_file, parse_dates=["date"])
    orders_df = pd.read_csv(orders_file, parse_dates=["date"])

    # Assign test/control groups
    np.random.seed(42)
    cities = revenue_df["geo_location"].unique()
    holdout_count = int(len(cities) * (holdout_pct / 100))
    control_cities = np.random.choice(cities, size=holdout_count, replace=False)
    revenue_df["group"] = revenue_df["geo_location"].apply(lambda x: "control" if x in control_cities else "test")

    # Show control city assignments
    st.sidebar.markdown("### 🗺️ Control Cities")
    st.sidebar.write(sorted(control_cities.tolist()))

    # Filter for test period
    days = test_weeks * 7
    test_start = revenue_df["date"].max() - pd.Timedelta(days=days)
    test_df = revenue_df[revenue_df["date"] > test_start]

    # Power analysis
    baseline = test_df["revenue"].mean()
    std_dev = test_df["revenue"].std()
    lift_amt = (budget_increase_pct / 100) * baseline
    effect_size = lift_amt / std_dev

    analysis = TTestIndPower()
    sample_needed = int(np.ceil(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)))

    # Observed metrics
    test_group = test_df[test_df["group"] == "test"]["revenue"]
    control_group = test_df[test_df["group"] == "control"]["revenue"]

    # Warn if control group is too small
    if len(control_group) < 5:
        st.warning("⚠️ Control group is too small for valid statistical analysis. Increase holdout share or test duration.")
    
    # Warn if underpowered
    if len(test_group) < sample_needed or len(control_group) < sample_needed:
        st.warning("⚠️ Your sample sizes are smaller than required for the selected power/significance level. Consider increasing test duration or holdout share.")

    # Proceed with t-test if valid
    if len(test_group) >= 2 and len(control_group) >= 2:
        t_stat, p_val = ttest_ind(test_group, control_group, equal_var=False)
        obs_lift = test_group.mean() - control_group.mean()
        percent_lift = (obs_lift / control_group.mean()) * 100
    else:
        t_stat, p_val, obs_lift, percent_lift = None, None, None, None

    # Budget calc
    total_budget = weekly_budget * (1 + (budget_increase_pct / 100)) * test_weeks

    # Summary Table
    st.subheader("📊 Summary")
    results = {
        "Required Sample Size per Group": sample_needed,
        "Actual Sample Size (Test)": len(test_group),
        "Actual Sample Size (Control)": len(control_group),
        "Baseline Revenue": round(baseline, 2),
        "Expected Lift ($)": round(lift_amt, 2),
        "Effect Size": round(effect_size, 3),
        "Observed Lift ($)": round(obs_lift, 2) if obs_lift is not None else "N/A",
        "Observed % Lift": f"{round(percent_lift, 2)}%" if percent_lift is not None else "N/A",
        "P-Value": round(p_val, 4) if p_val is not None else "N/A",
        "Total Test Budget": f"${round(total_budget, 2)}",
        "Holdout Size %": f"{round((holdout_count / len(cities)) * 100, 1)}%"
    }

    st.dataframe(pd.DataFrame(list(results.items()), columns=["Metric", "Value"]))

    # Plot time series
    st.subheader("📈 Revenue Time Series: Test vs Control")
    avg_ts = test_df.groupby(["date", "group"])["revenue"].mean().unstack()
    st.line_chart(avg_ts)

else:
    st.info("👈 Please upload both metric CSV files to get started.")
