"""
Fathom Labs Researcher Retention — interactive Streamlit app.

Loads the fitted logistic regression model and lets the viewer shift
organization-level variables (escalation response, workload, equity cliff)
to see how predicted researcher attrition moves, grounded in a real
fitted model rather than a static chart.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fathom Labs Researcher Retention",
    page_icon="🧭",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load data and model
# ----------------------------------------------------------------------


@st.cache_data
def load_data():
    return pd.read_csv("data/fathom_labs_researchers_clean.csv")


@st.cache_data
def load_model():
    with open("analysis/model_coefficients.json") as f:
        return json.load(f)


df = load_data()
model = load_model()

FEATURES = model["features"]
COEF = model["coefficients"]
INTERCEPT = model["intercept"]
MEANS = model["feature_means"]
STDS = model["feature_stds"]


def predict_attrition_rate(shifted_df: pd.DataFrame) -> float:
    """Score the (shifted) population with the fitted logistic model and
    return the average predicted probability of leaving."""
    z = INTERCEPT
    for f in FEATURES:
        x_std = (shifted_df[f] - MEANS[f]) / STDS[f]
        z = z + COEF[f] * x_std
    probs = 1 / (1 + np.exp(-z))
    return probs.mean()


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("🧭 Fathom Labs Researcher Retention")
st.caption(
    "A synthetic case study on what predicts AI safety researchers leaving — "
    "inspired by real 2026 resignations citing AI-risk concerns. All data below "
    "is synthetic; Fathom Labs is a fictional company."
)

baseline_rate = predict_attrition_rate(df)

col1, col2, col3 = st.columns(3)
col1.metric("Modeled population", f"{len(df):,} researchers")
col2.metric("Current predicted attrition", f"{baseline_rate:.1%}")
col3.metric("Model fit (ROC AUC)", f"{model['auc']:.2f}")

st.divider()

# ----------------------------------------------------------------------
# Finding: what actually predicts leaving
# ----------------------------------------------------------------------

st.subheader("What actually predicts a researcher leaving")
st.markdown(
    "The hypothesis going in: researchers who don't trust that raising a "
    "safety concern leads anywhere (**low escalation confidence**), and who "
    "believe the pace of the work itself causes harm (**high moral injury**), "
    "would be far more likely to leave than researchers who are simply "
    "overworked. The fitted model supports that — in order of standardized "
    "effect size:"
)

coef_df = pd.DataFrame(
    {"Factor": FEATURES, "Standardized effect on attrition risk": [COEF[f] for f in FEATURES]}
).assign(abs_effect=lambda d: d["Standardized effect on attrition risk"].abs())
coef_df = coef_df.sort_values("abs_effect", ascending=True)

st.bar_chart(
    coef_df.set_index("Factor")["Standardized effect on attrition risk"],
    horizontal=True,
)

st.caption(
    "Positive = raises attrition risk. Negative = lowers it. "
    "Escalation confidence and moral injury dominate; raw workload hours barely register — "
    "it isn't the hours that predict people leaving, it's whether they feel heard and whether "
    "they believe the work itself is doing harm."
)

st.divider()

# ----------------------------------------------------------------------
# Interactive simulator
# ----------------------------------------------------------------------

st.subheader("Simulate an organizational change")
st.markdown(
    "Adjust what Fathom Labs could realistically change, and see predicted "
    "attrition shift — scored by the same fitted model, applied to the whole "
    "population."
)

sim_col, result_col = st.columns([3, 2])

with sim_col:
    escalation_boost = st.slider(
        "Improve escalation confidence (e.g. a safety concerns board with "
        "visible follow-through)",
        min_value=0.0, max_value=2.0, value=0.0, step=0.1,
        help="Shifts every researcher's escalation_confidence score upward, capped at 5.",
    )
    moral_injury_relief = st.slider(
        "Reduce moral injury (e.g. slower release cadence, more say over what ships)",
        min_value=0.0, max_value=2.0, value=0.0, step=0.1,
        help="Shifts every researcher's moral_injury_index downward, floored at 1.",
    )
    manager_support_boost = st.slider(
        "Strengthen manager support (e.g. manager training, smaller spans of control)",
        min_value=0.0, max_value=2.0, value=0.0, step=0.1,
    )
    workload_change = st.slider(
        "Change average weekly workload (hours)",
        min_value=-15, max_value=15, value=0, step=1,
    )
    equity_cliff_change = st.slider(
        "Shorten equity cliff (months, negative = shorter)",
        min_value=-6, max_value=6, value=0, step=1,
        help="A shorter cliff raises effective equity_vested_pct for recent hires.",
    )

shifted = df.copy()
shifted["escalation_confidence"] = np.clip(
    shifted["escalation_confidence"] + escalation_boost, 1, 5
)
shifted["moral_injury_index"] = np.clip(
    shifted["moral_injury_index"] - moral_injury_relief, 1, 5
)
shifted["manager_support"] = np.clip(
    shifted["manager_support"] + manager_support_boost, 1, 5
)
shifted["workload_hours_week"] = np.clip(
    shifted["workload_hours_week"] + workload_change, 35, 90
)
shifted["equity_vested_pct"] = np.clip(
    shifted["equity_vested_pct"] + equity_cliff_change * 3, 0, 100
)

new_rate = predict_attrition_rate(shifted)
delta = new_rate - baseline_rate

with result_col:
    st.metric(
        "Predicted attrition under this scenario",
        f"{new_rate:.1%}",
        delta=f"{delta:+.1%}",
        delta_color="inverse",
    )
    if abs(delta) < 0.001:
        st.info("Move a slider to see the model's prediction shift.")
    elif delta < 0:
        st.success(
            f"This combination is projected to reduce attrition by "
            f"{abs(delta) * len(df):.0f} researchers in this population."
        )
    else:
        st.warning(
            f"This combination is projected to increase attrition by "
            f"{abs(delta) * len(df):.0f} researchers in this population."
        )

st.divider()

# ----------------------------------------------------------------------
# Raw data
# ----------------------------------------------------------------------

with st.expander("View the underlying synthetic dataset"):
    st.dataframe(df, use_container_width=True)

st.caption(
    "Methodology: synthetic data generated with Python Faker, cleaned in SQL, "
    "modeled with a standardized logistic regression (scikit-learn). "
    "Full pipeline and write-up in the repo README."
)
