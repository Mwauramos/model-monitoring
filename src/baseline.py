# baseline.py
# Generates training baseline data for drift detection
# Reuses the malaria dataset from Project 3
# ZENIK.AI — Computational Epidemiology Portfolio

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

np.random.seed(42)

KENYA_COUNTIES = [
    "Kisumu", "Homa Bay", "Migori", "Siaya",
    "Nairobi", "Mombasa", "Nakuru", "Kisii",
    "Turkana", "Marsabit", "Kakamega", "Bungoma"
]

COUNTY_MALARIA_RISK = {
    "Kisumu": 0.75, "Homa Bay": 0.80, "Migori": 0.72, "Siaya": 0.78,
    "Nairobi": 0.15, "Mombasa": 0.35, "Nakuru": 0.20, "Kisii": 0.45,
    "Turkana": 0.40, "Marsabit": 0.38, "Kakamega": 0.55, "Bungoma": 0.50
}


def generate_baseline_data(n=1000):
    """Generate baseline training distribution."""
    counties = np.random.choice(KENYA_COUNTIES, size=n)
    ages = np.random.gamma(shape=2, scale=10, size=n).clip(0, 85).astype(int)
    sex = np.random.choice([0, 1], size=n, p=[0.49, 0.51])
    iron_deficiency = np.random.binomial(1, 0.45, n)
    bed_net_use = np.random.binomial(1, 0.55, n)
    hiv_positive = np.random.binomial(1, 0.06, n)
    prior_malaria = np.random.binomial(1, 0.60, n)
    haemoglobin = np.random.normal(10.5, 2.5, n).clip(4, 18)
    parasite_density = np.random.exponential(scale=5000, size=n).clip(100, 100000)
    days_to_treatment = np.random.exponential(scale=2, size=n).clip(0, 14).astype(int)
    wealth_quintile = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.25, 0.25, 0.20, 0.18, 0.12])
    high_risk_county = np.array([1 if c in ["Kisumu", "Homa Bay", "Migori", "Siaya"] else 0 for c in counties])

    county_risk = np.array([COUNTY_MALARIA_RISK[c] for c in counties])
    log_odds = (
        -2.0
        + 0.8 * (ages < 5).astype(int)
        + 0.6 * iron_deficiency
        - 0.5 * bed_net_use
        + 0.7 * hiv_positive
        + 0.4 * prior_malaria
        - 0.3 * (haemoglobin > 11).astype(int)
        + 0.5 * np.log(parasite_density / 1000)
        + 0.4 * days_to_treatment
        - 0.3 * (wealth_quintile >= 4).astype(int)
        + np.log(county_risk / (1 - county_risk))
    )
    prob_severe = 1 / (1 + np.exp(-log_odds))
    severe_malaria = np.random.binomial(1, prob_severe)

    df = pd.DataFrame({
        "age": ages,
        "sex_male": sex,
        "iron_deficiency": iron_deficiency,
        "bed_net_use": bed_net_use,
        "hiv_positive": hiv_positive,
        "prior_malaria_episode": prior_malaria,
        "haemoglobin_gdl": haemoglobin.round(1),
        "parasite_density": parasite_density.round(0).astype(int),
        "days_to_treatment": days_to_treatment,
        "wealth_quintile": wealth_quintile,
        "high_risk_county": high_risk_county,
        "severe_malaria": severe_malaria
    })
    return df


def generate_drifted_data(n=200, drift_type="seasonal"):
    """
    Generate production data with drift.
    drift_type options:
    - seasonal: rainy season increases parasite density
    - campaign: bed net campaign increases coverage
    - outbreak: new high-risk county emerges
    """
    df = generate_baseline_data(n)

    if drift_type == "seasonal":
        # Rainy season: higher parasite density across the board
        df["parasite_density"] = (df["parasite_density"] * 2.5).clip(100, 100000)
        drift_description = "Rainy season drift: parasite density increased 2.5x"

    elif drift_type == "campaign":
        # Bed net campaign: coverage increased significantly
        df["bed_net_use"] = np.random.binomial(1, 0.85, n)
        drift_description = "Bed net campaign drift: coverage increased from 55% to 85%"

    elif drift_type == "outbreak":
        # New outbreak: younger patients presenting more frequently
        df["age"] = np.random.gamma(shape=1.5, scale=5, size=n).clip(0, 85).astype(int)
        df["parasite_density"] = (df["parasite_density"] * 1.8).clip(100, 100000)
        drift_description = "Outbreak drift: younger patients, higher parasite density"

    df.attrs["drift_type"] = drift_type
    df.attrs["drift_description"] = drift_description
    return df, drift_description


if __name__ == "__main__":
    baseline = generate_baseline_data(1000)
    os.makedirs("data", exist_ok=True)
    baseline.to_csv("data/baseline.csv", index=False)
    print(f"Baseline dataset: {len(baseline)} patients")
    print(f"Severe malaria rate: {baseline['severe_malaria'].mean():.1%}")
    print(f"Avg parasite density: {baseline['parasite_density'].mean():.0f}")
    print(f"Bed net coverage: {baseline['bed_net_use'].mean():.1%}")
    print("Baseline saved to data/baseline.csv")