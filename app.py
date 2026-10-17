# app.py
# Model Monitoring and Drift Detection Dashboard
# ZENIK.AI — Computational Epidemiology Portfolio

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

st.set_page_config(
    page_title="Kenya Malaria Model Monitor",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Kenya Malaria Model — Drift Detection Dashboard")
st.markdown(
    "Monitors the Kenya malaria prediction model for data drift across "
    "three real-world scenarios: seasonal transmission changes, bed net "
    "campaigns, and outbreak events."
)
st.divider()

from baseline import generate_baseline_data, generate_drifted_data
from drift_detector import run_drift_detection, FEATURES

@st.cache_data
def load_baseline():
    if os.path.exists("data/baseline.csv"):
        return pd.read_csv("data/baseline.csv")
    return generate_baseline_data(1000)

baseline_df = load_baseline()

tab1, tab2, tab3 = st.tabs(["Drift Analysis", "Feature Distributions", "Run Detection"])

with tab1:
    st.markdown("### Drift Detection Results")

    if os.path.exists("outputs/drift_summary.json"):
        with open("outputs/drift_summary.json") as f:
            results = json.load(f)

        for result in results:
            drift_type = result["drift_type"].title()
            drifted = result["number_of_drifted_features"]
            share = result["drift_share"]
            alert = result["alert"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{drift_type} Scenario", f"{drifted} features drifted")
            with col2:
                st.metric("Drift Share", f"{share:.1%}")
            with col3:
                if alert:
                    st.error("ALERT — Retraining recommended")
                else:
                    st.success("Within acceptable range")

            if result["drifted_features"]:
                st.markdown(f"**Drifted features in {drift_type} scenario:**")
                for f in result["drifted_features"]:
                    st.markdown(f"- {f['feature']}: drift score {f['drift_score']}")

            st.divider()
    else:
        st.info("No drift results yet. Run detection from the Run Detection tab.")

with tab2:
    st.markdown("### Feature Distribution Comparison")
    st.markdown("Compare baseline training distributions against simulated production scenarios.")

    scenario = st.selectbox("Select scenario:", ["seasonal", "campaign", "outbreak"])
    feature = st.selectbox("Select feature:", FEATURES)

    current_data, description = generate_drifted_data(200, drift_type=scenario)
    st.markdown(f"**Scenario:** {description}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=baseline_df[feature],
        name="Baseline (Training)",
        opacity=0.6,
        nbinsx=30,
        marker_color="#1B4332"
    ))
    fig.add_trace(go.Histogram(
        x=current_data[feature],
        name=f"Production ({scenario})",
        opacity=0.6,
        nbinsx=30,
        marker_color="#BC4A1A"
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution of {feature}: Baseline vs {scenario.title()} Scenario",
        xaxis_title=feature,
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline statistics:**")
        st.dataframe(baseline_df[feature].describe().round(3))
    with col2:
        st.markdown(f"**{scenario.title()} scenario statistics:**")
        st.dataframe(current_data[feature].describe().round(3))

with tab3:
    st.markdown("### Run Live Drift Detection")
    st.markdown("Simulate production data and run Evidently drift detection in real time.")

    scenario_choice = st.selectbox(
        "Choose drift scenario:",
        ["seasonal", "campaign", "outbreak"],
        help="seasonal: rainy season increases parasite density | campaign: bed net coverage increases | outbreak: younger patients, higher parasite load"
    )

    n_samples = st.slider("Production samples to generate:", 100, 500, 200)

    if st.button("Run Drift Detection", type="primary"):
        with st.spinner("Running drift detection..."):
            current_data, description = generate_drifted_data(n_samples, drift_type=scenario_choice)
            summary, report = run_drift_detection(baseline_df, current_data, drift_type=scenario_choice)

        st.markdown(f"**Scenario:** {description}")
        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Drifted", summary["number_of_drifted_features"])
        with col2:
            st.metric("Drift Share", f"{summary['drift_share']:.1%}")
        with col3:
            if summary["alert"]:
                st.error("ALERT — Model retraining recommended")
            else:
                st.success("No alert — within acceptable range")

        if summary["drifted_features"]:
            st.markdown("### Drifted Features")
            drift_df = pd.DataFrame(summary["drifted_features"])
            fig = px.bar(
                drift_df,
                x="feature",
                y="drift_score",
                title="Drift Score by Feature",
                color="drift_score",
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No individual features drifted significantly.")

        os.makedirs("outputs", exist_ok=True)
        report.save_html(f"outputs/drift_report_{scenario_choice}_live.html")
        st.success(f"Evidently HTML report saved to outputs/drift_report_{scenario_choice}_live.html")

st.divider()
st.caption("ZENIK.AI · Kenya Malaria Model Monitoring · Evidently AI + Streamlit")