# drift_detector.py
# Detects data drift between baseline and production data using Evidently
# ZENIK.AI — Computational Epidemiology Portfolio

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.baseline import generate_baseline_data, generate_drifted_data

# ── What this does ──────────────────────────────────────────────────────────
# Evidently compares two datasets — reference (training) and current (production)
# It calculates statistical tests for each feature to detect distribution shifts
# The drift report tells us which features drifted and by how much
# ────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "age", "sex_male", "iron_deficiency", "bed_net_use",
    "hiv_positive", "prior_malaria_episode", "haemoglobin_gdl",
    "parasite_density", "days_to_treatment", "wealth_quintile",
    "high_risk_county"
]

DRIFT_THRESHOLD = 0.3  # If more than 30% of features drift, raise alert


def run_drift_detection(reference_df: pd.DataFrame,
                        current_df: pd.DataFrame,
                        drift_type: str = "unknown") -> dict:
    """
    Run Evidently drift detection between reference and current data.
    Returns a summary dict with drift results.
    """
    print(f"Running drift detection for: {drift_type}")
    print(f"Reference: {len(reference_df)} samples | Current: {len(current_df)} samples")

    ref = reference_df[FEATURES]
    cur = current_df[FEATURES]

    report = Report(metrics=[
        DataDriftPreset(),
    ])
    report.run(reference_data=ref, current_data=cur)

    report_dict = report.as_dict()

    # Extract drift results
    drift_results = {}
    drifted_features = []

    metrics = report_dict.get("metrics", [])
    for metric in metrics:
        if metric.get("metric") == "DatasetDriftMetric":
            result = metric.get("result", {})
            dataset_drift = result.get("dataset_drift", False)
            drift_share = result.get("share_of_drifted_columns", 0)
            number_of_drifted = result.get("number_of_drifted_columns", 0)

        if metric.get("metric") == "ColumnDriftMetric":
            col_result = metric.get("result", {})
            col_name = col_result.get("column_name", "")
            col_drift = col_result.get("drift_detected", False)
            drift_score = col_result.get("drift_score", 0)
            if col_drift:
                drifted_features.append({
                    "feature": col_name,
                    "drift_score": round(drift_score, 4)
                })
            drift_results[col_name] = {
                "drifted": col_drift,
                "drift_score": round(drift_score, 4)
            }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "drift_type": drift_type,
        "reference_size": len(reference_df),
        "current_size": len(current_df),
        "dataset_drift_detected": dataset_drift,
        "drift_share": round(drift_share, 3),
        "number_of_drifted_features": number_of_drifted,
        "drifted_features": drifted_features,
        "feature_results": drift_results,
        "alert": dataset_drift and drift_share > DRIFT_THRESHOLD
    }

    return summary, report


def save_report(report, filename: str):
    """Save Evidently HTML report."""
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/{filename}"
    report.save_html(path)
    print(f"Report saved: {path}")
    return path


if __name__ == "__main__":
    print("Loading baseline data...")
    baseline = pd.read_csv("data/baseline.csv")

    drift_scenarios = ["seasonal", "campaign", "outbreak"]
    all_results = []

    for scenario in drift_scenarios:
        print(f"\n{'='*50}")
        current_data, description = generate_drifted_data(200, drift_type=scenario)
        summary, report = run_drift_detection(baseline, current_data, drift_type=scenario)

        print(f"Drift detected: {summary['dataset_drift_detected']}")
        print(f"Features drifted: {summary['number_of_drifted_features']}")
        print(f"Drift share: {summary['drift_share']:.1%}")
        if summary['drifted_features']:
            print("Drifted features:")
            for f in summary['drifted_features']:
                print(f"  {f['feature']}: score {f['drift_score']}")
        print(f"Alert triggered: {summary['alert']}")

        save_report(report, f"drift_report_{scenario}.html")
        all_results.append(summary)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/drift_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll drift reports saved to outputs folder.")
    print("Day 1 of Project 7 complete.")